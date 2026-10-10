package com.samanramezani.aichattest

import android.content.ContentResolver
import android.net.Uri
import android.provider.OpenableColumns
import com.woogit.aicore.domain.ChatMessage
import com.woogit.aicore.domain.GenerationRequest
import com.woogit.aicore.domain.GenerationResult
import com.woogit.aicore.domain.ModelDescriptor
import com.woogit.aicore.domain.ModelError
import com.woogit.aicore.domain.ModelResult
import com.woogit.aicore.domain.InferenceSettings
import com.woogit.aicore.runtime.FileModelImporter
import com.woogit.aicore.runtime.FileModelRepository
import com.woogit.aicore.runtime.LocalModelService
import com.woogit.aicore.runtime.RuntimeDiagnosticsStore
import com.woogit.aicore.runtime.android.LlamaCppAndroidRuntimeAdapter
import com.woogit.aicore.runtime.RuntimeAdapter
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.file.Files
import java.nio.file.Path

/** Android model façade: SAF import -> GGUF validation -> durable registry -> llama.cpp/OpenCL. */
class AndroidModelManager(
    private val contentResolver: ContentResolver,
    private val appContext: android.content.Context,
    modelDirectory: Path,
    private val runtime: LlamaCppAndroidRuntimeAdapter,
    private val dispatchRuntime: RuntimeAdapter,
) {
    private val root = modelDirectory
    private val repository = FileModelRepository(root.resolve("models.properties"))
    private val importer = FileModelImporter(root)
    private val draftRoot = root.resolve("drafts")
    private val draftRepository = FileModelRepository(root.resolve("drafts.properties"))
    private val draftImporter = FileModelImporter(draftRoot)
    private val mnnRoot = modelDirectory.resolve("mnn")
    private val mnnRepository = FileModelRepository(modelDirectory.resolve("mnn-models.properties"))
    private val mnnPackImporter = MnnModelPackImporter()
    private val service = LocalModelService(importer, repository, dispatchRuntime)

    suspend fun import(uri: Uri): ModelResult<ModelDescriptor> = withContext(Dispatchers.IO) {
        val displayName = runCatching {
            contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { cursor ->
                if (cursor.moveToFirst()) cursor.getString(0) else null
            }
        }.getOrNull().orEmpty()
        if (displayName.endsWith(".zip", ignoreCase = true) || displayName.endsWith(".mnnpack", ignoreCase = true)) {
            return@withContext importMnnPack(uri)
        }
        try {
            Files.createDirectories(root)
            val staged = Files.createTempFile(root, "import-", ".gguf")
            try {
                val input = contentResolver.openInputStream(uri)
                    ?: return@withContext ModelResult.Failure(ModelError.FileAccess("Unable to open the selected model file"))
                input.use { source -> Files.newOutputStream(staged).use { target -> source.copyTo(target) } }
                service.importModel(staged)
            } catch (t: Throwable) {
                Files.deleteIfExists(staged)
                ModelResult.Failure(ModelError.Storage("Unable to stage the selected model", t))
            }
        } catch (t: Throwable) {
            ModelResult.Failure(ModelError.Storage("Unable to prepare model storage", t))
        }
    }

    suspend fun models(): ModelResult<List<ModelDescriptor>> {
        val gguf = service.listModels()
        if (gguf is ModelResult.Failure) return gguf
        val mnn = mnnRepository.list()
        if (mnn is ModelResult.Failure) return mnn
        return ModelResult.Success((gguf as ModelResult.Success).value + (mnn as ModelResult.Success).value)
    }

    private fun importMnnPack(uri: Uri): ModelResult<ModelDescriptor> {
        var extracted: ModelDescriptor? = null
        return try {
            Files.createDirectories(mnnRoot)
            val input = contentResolver.openInputStream(uri)
                ?: return ModelResult.Failure(ModelError.FileAccess("Unable to open the selected MNN model archive"))
            input.use { stream -> extracted = mnnPackImporter.unpack(stream, mnnRoot) }
            val model = requireNotNull(extracted)
            when (val registered = mnnRepository.register(model)) {
                is ModelResult.Success -> {
                    RuntimeDiagnosticsStore.recordNativeEvent("MNN_MODEL_PACK_IMPORTED id=${model.id} sizeBytes=${model.sizeBytes}")
                    ModelResult.Success(model)
                }
                is ModelResult.Failure -> {
                    runCatching { mnnPackImporter.deleteRecursively(model.path.parent) }
                    registered
                }
            }
        } catch (t: Throwable) {
            extracted?.let { runCatching { mnnPackImporter.deleteRecursively(it.path.parent) } }
            ModelResult.Failure(ModelError.InvalidModel("Invalid MNN model ZIP: ${t.message}"))
        }
    }

    suspend fun draftModels(): ModelResult<List<ModelDescriptor>> = withContext(Dispatchers.IO) {
        draftRepository.list()
    }

    suspend fun importDraft(uri: Uri): ModelResult<ModelDescriptor> = withContext(Dispatchers.IO) {
        try {
            Files.createDirectories(draftRoot)
            val staged = Files.createTempFile(draftRoot, "draft-import-", ".gguf")
            try {
                val input = contentResolver.openInputStream(uri)
                    ?: return@withContext ModelResult.Failure(ModelError.FileAccess("Unable to open the selected draft model file"))
                input.use { source -> Files.newOutputStream(staged).use { target -> source.copyTo(target) } }
                when (val imported = draftImporter.import(staged)) {
                    is ModelResult.Failure -> { Files.deleteIfExists(staged); imported }
                    is ModelResult.Success -> when (val registered = draftRepository.register(imported.value)) {
                        is ModelResult.Success -> imported
                        is ModelResult.Failure -> { Files.deleteIfExists(imported.value.path); registered }
                    }
                }
            } catch (t: Throwable) {
                Files.deleteIfExists(staged)
                ModelResult.Failure(ModelError.Storage("Unable to import draft model", t))
            }
        } catch (t: Throwable) {
            ModelResult.Failure(ModelError.Storage("Unable to prepare draft model storage", t))
        }
    }

    suspend fun draftForModel(modelId: String): ModelResult<ModelDescriptor?> = withContext(Dispatchers.IO) {
        val registry = root.resolve("draft-bindings.properties")
        try {
            if (!Files.isRegularFile(registry)) return@withContext ModelResult.Success(null)
            val props = java.util.Properties()
            Files.newInputStream(registry).use(props::load)
            val draftId = props.getProperty("target.$modelId.draftId").orEmpty()
            if (draftId.isBlank()) ModelResult.Success(null) else draftRepository.get(draftId)
        } catch (t: Throwable) {
            ModelResult.Failure(ModelError.Storage("Unable to read draft assignment", t))
        }
    }

    suspend fun assignDraft(modelId: String, draftId: String?): ModelResult<Unit> = withContext(Dispatchers.IO) {
        try {
            val draft = if (draftId.isNullOrBlank()) null else when (val result = draftRepository.get(draftId)) {
                is ModelResult.Success -> result.value
                is ModelResult.Failure -> return@withContext result
            }
            if (draftId != null && draft == null) {
                return@withContext ModelResult.Failure(ModelError.InvalidModel("Draft model is not registered"))
            }
            val registry = root.resolve("draft-bindings.properties")
            val props = java.util.Properties()
            if (Files.isRegularFile(registry)) Files.newInputStream(registry).use(props::load)
            val key = "target.$modelId.draftId"
            if (draft == null) props.remove(key) else props.setProperty(key, draft.id)
            Files.createDirectories(root)
            val temp = Files.createTempFile(root, "draft-bindings-", ".tmp")
            try {
                Files.newOutputStream(temp).use { props.store(it, "AI Chat speculative draft assignments") }
                runCatching { Files.move(temp, registry, java.nio.file.StandardCopyOption.ATOMIC_MOVE, java.nio.file.StandardCopyOption.REPLACE_EXISTING) }
                    .getOrElse { Files.move(temp, registry, java.nio.file.StandardCopyOption.REPLACE_EXISTING) }
            } finally {
                Files.deleteIfExists(temp)
            }
            runtime.setDraftPath(draft?.path?.toString())
            // A draft assignment is a live runtime connection, not only a persisted
            // preference. Keep target-only activation intact, then reload the already
            // active target with the selected draft. Native speculative setup happens
            // only after the target context is fully ready.
            val active = (repository.getActive() as? ModelResult.Success)?.value
            if (draft != null && active != null && active.id == modelId) {
                when (val reloaded = runtime.loadResult(active)) {
                    is ModelResult.Failure -> return@withContext reloaded
                    is ModelResult.Success -> RuntimeDiagnosticsStore.recordNativeEvent(
                        "SPECULATIVE_DRAFT_CONNECTED target=${active.id} draft=${draft.id}"
                    )
                }
            } else if (draft == null) {
                RuntimeDiagnosticsStore.recordNativeEvent("SPECULATIVE_DRAFT_DISCONNECTED target=$modelId")
            }
            ModelResult.Success(Unit)
        } catch (t: Throwable) {
            ModelResult.Failure(ModelError.Storage("Unable to save draft assignment", t))
        }
    }

    suspend fun deleteDraft(id: String): ModelResult<Unit> = withContext(Dispatchers.IO) {
        try {
            val draft = when (val result = draftRepository.get(id)) {
                is ModelResult.Success -> result.value
                is ModelResult.Failure -> return@withContext result
            } ?: return@withContext ModelResult.Failure(ModelError.InvalidModel("Draft model is not registered"))
            val active = (repository.getActive() as? ModelResult.Success)?.value
            if (active != null && (draftForModel(active.id) as? ModelResult.Success)?.value?.id == id) {
                assignDraft(active.id, null)
            }
            Files.deleteIfExists(draft.path)
            draftRepository.unregister(id)
        } catch (t: Throwable) {
            ModelResult.Failure(ModelError.Storage("Unable to delete draft model", t))
        }
    }

    suspend fun activeModel(): ModelResult<ModelDescriptor?> {
        return when (val activeMnn = mnnRepository.getActive()) {
            is ModelResult.Failure -> activeMnn
            is ModelResult.Success -> if (activeMnn.value != null) activeMnn else service.activeModel()
        }
    }

    suspend fun restoreActive(): ModelResult<ModelDescriptor?> = withContext(Dispatchers.IO) {
        val activeMnn = (mnnRepository.getActive() as? ModelResult.Success)?.value
        if (activeMnn != null) {
            return@withContext try {
                runtime.setDraftPath(null)
                dispatchRuntime.load(activeMnn)
                ModelRuntimeForegroundService.start(appContext)
                ModelResult.Success(activeMnn)
            } catch (t: Throwable) {
                RuntimeDiagnosticsStore.recordNativeEvent("MNN_RESTORE_FAILED error=${t.message}")
                ModelResult.Failure(ModelError.LoadFailed("Unable to restore MNN model: ${t.message}", t))
            }
        }
        // Target activation is deliberately independent from speculative decoding.
        // Never let a persisted/broken draft participate in target model load.
        runtime.setDraftPath(null)
        val result = service.restoreActive()
        if (result is ModelResult.Success) {
            val restored = result.value
            if (restored != null) {
                attachDraftAfterTargetLoad(restored)
                ModelRuntimeForegroundService.start(appContext)
            }
        }
        result
    }

    suspend fun activate(id: String): ModelResult<ModelDescriptor> = withContext(Dispatchers.IO) {
        val mnnModel = (mnnRepository.get(id) as? ModelResult.Success)?.value
        if (mnnModel != null) {
            runtime.setDraftPath(null)
            service.deactivate()
            mnnRepository.setActive(id)
            return@withContext try {
                dispatchRuntime.load(mnnModel)
                RuntimeDiagnosticsStore.recordNativeEvent("MNN_MODEL_ACTIVE id=${mnnModel.id} backend=${dispatchRuntime.runtimeInfo().backend}")
                ModelRuntimeForegroundService.start(appContext)
                ModelResult.Success(mnnModel)
            } catch (t: Throwable) {
                mnnRepository.setActive(null)
                runCatching { dispatchRuntime.unload() }
                ModelResult.Failure(ModelError.LoadFailed("Unable to activate MNN model: ${t.message}", t))
            }
        }
        mnnRepository.setActive(null)
        // The target must always load with speculative decoding disabled. Draft state
        // is connected only after the target runtime has reported a successful load.
        runtime.setDraftPath(null)
        val result = service.activate(id)
        if (result is ModelResult.Success) {
            attachDraftAfterTargetLoad(result.value)
            ModelRuntimeForegroundService.start(appContext)
        }
        result
    }

    private suspend fun attachDraftAfterTargetLoad(model: ModelDescriptor) {
        val modelId = model.id
        val draft = (draftForModel(modelId) as? ModelResult.Success)?.value
        val draftPath = draft?.path?.takeIf { Files.isRegularFile(it) && Files.isReadable(it) }?.toString()
        runtime.setDraftPath(draftPath)
        if (draftPath != null) {
            // Target activation already succeeded. Re-load the same target with the
            // now-selected draft so speculative decoding becomes active immediately.
            when (val connected = runtime.loadResult(model)) {
                is ModelResult.Failure -> RuntimeDiagnosticsStore.recordNativeEvent("SPECULATIVE_DRAFT_CONNECT_FAILED target=$modelId error=${connected.error.message}")
                is ModelResult.Success -> RuntimeDiagnosticsStore.recordNativeEvent("SPECULATIVE_DRAFT_CONNECTED target=$modelId draft=${draft.id}")
            }
        }
        if (draft != null && draftPath == null) {
            RuntimeDiagnosticsStore.recordNativeEvent("SPECULATIVE_DRAFT_IGNORED_INVALID file=${draft.path}")
        }
    }

    suspend fun deactivate(): ModelResult<Unit> = withContext(Dispatchers.IO) {
        runtime.setDraftPath(null)
        mnnRepository.setActive(null)
        val result = service.deactivate()
        if (result is ModelResult.Success) {
            ModelRuntimeForegroundService.stop(appContext)
        }
        result
    }

    suspend fun unload(): ModelResult<Unit> = deactivate()

    suspend fun generate(
        messages: List<ChatMessage>,
        settings: InferenceSettings,
        onToken: suspend (String) -> Unit = {}
    ): ModelResult<GenerationResult> {
        val activeMnn = (mnnRepository.getActive() as? ModelResult.Success)?.value
        return if (activeMnn != null) {
            try {
                ModelResult.Success(dispatchRuntime.generate(GenerationRequest(messages, settings), onToken))
            } catch (t: Throwable) {
                ModelResult.Failure(ModelError.Inference("MNN generation failed: ${t.message}", t))
            }
        } else service.generate(messages, settings, onToken)
    }

    suspend fun stopGeneration() = dispatchRuntime.stopGeneration()

    suspend fun delete(id: String): ModelResult<Unit> = withContext(Dispatchers.IO) {
        val mnnModel = (mnnRepository.get(id) as? ModelResult.Success)?.value
        if (mnnModel != null) {
            if ((mnnRepository.getActive() as? ModelResult.Success)?.value?.id == id) deactivate()
            runCatching { mnnPackImporter.deleteRecursively(mnnModel.path.parent) }
            mnnRepository.unregister(id)
        } else service.deleteModel(id)
    }
}
