package com.samanramezani.aichattest

import android.content.ContentResolver
import android.net.Uri
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
import kotlinx.coroutines.CancellationException
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
) {
    private val root = modelDirectory
    private val repository = FileModelRepository(root.resolve("models.properties"))
    private val importer = FileModelImporter(root)
    private val draftRoot = root.resolve("drafts")
    private val draftRepository = FileModelRepository(root.resolve("drafts.properties"))
    private val draftImporter = FileModelImporter(draftRoot)
    private val service = LocalModelService(importer, repository, runtime)

    suspend fun import(uri: Uri): ModelResult<ModelDescriptor> = withContext(Dispatchers.IO) {
        try {
            Files.createDirectories(root)
            val staged = Files.createTempFile(root, "import-", ".gguf")
            try {
                val input = contentResolver.openInputStream(uri)
                if (input == null) {
                    Files.deleteIfExists(staged)
                    return@withContext ModelResult.Failure(ModelError.FileAccess("Unable to open the selected model file"))
                }
                input.use { source -> Files.newOutputStream(staged).use { target -> source.copyTo(target) } }
                service.importModel(staged)
            } catch (cancelled: CancellationException) {
                Files.deleteIfExists(staged)
                throw cancelled
            } catch (t: Throwable) {
                Files.deleteIfExists(staged)
                ModelResult.Failure(ModelError.Storage("Unable to stage the selected model", t))
            }
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (t: Throwable) {
            ModelResult.Failure(ModelError.Storage("Unable to prepare model storage", t))
        }
    }

    suspend fun models(): ModelResult<List<ModelDescriptor>> = service.listModels()

    suspend fun draftModels(): ModelResult<List<ModelDescriptor>> = withContext(Dispatchers.IO) {
        draftRepository.list()
    }

    suspend fun importDraft(uri: Uri): ModelResult<ModelDescriptor> = withContext(Dispatchers.IO) {
        try {
            Files.createDirectories(draftRoot)
            val staged = Files.createTempFile(draftRoot, "draft-import-", ".gguf")
            try {
                val input = contentResolver.openInputStream(uri)
                if (input == null) {
                    Files.deleteIfExists(staged)
                    return@withContext ModelResult.Failure(ModelError.FileAccess("Unable to open the selected draft model file"))
                }
                input.use { source -> Files.newOutputStream(staged).use { target -> source.copyTo(target) } }
                when (val imported = draftImporter.import(staged)) {
                    is ModelResult.Failure -> { Files.deleteIfExists(staged); imported }
                    is ModelResult.Success -> when (val registered = draftRepository.register(imported.value)) {
                        is ModelResult.Success -> imported
                        is ModelResult.Failure -> { Files.deleteIfExists(imported.value.path); registered }
                    }
                }
            } catch (cancelled: CancellationException) {
                Files.deleteIfExists(staged)
                throw cancelled
            } catch (t: Throwable) {
                Files.deleteIfExists(staged)
                ModelResult.Failure(ModelError.Storage("Unable to import draft model", t))
            }
        } catch (cancelled: CancellationException) {
            throw cancelled
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
        } catch (cancelled: CancellationException) {
            throw cancelled
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
            RuntimeDiagnosticsStore.recordNativeEvent(
                if (draft == null) {
                    "SPECULATIVE_DRAFT_UNASSIGNED target=$modelId"
                } else {
                    "SPECULATIVE_DRAFT_SAVED_NATIVE_DISABLED target=$modelId draft=${draft.id}"
                },
            )
            ModelResult.Success(Unit)
        } catch (cancelled: CancellationException) {
            throw cancelled
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
                when (val detached = assignDraft(active.id, null)) {
                    is ModelResult.Failure -> return@withContext detached
                    is ModelResult.Success -> Unit
                }
            }
            Files.deleteIfExists(draft.path)
            draftRepository.unregister(id)
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (t: Throwable) {
            ModelResult.Failure(ModelError.Storage("Unable to delete draft model", t))
        }
    }

    fun requestRuntimeReload() = service.requestRuntimeReload()

    suspend fun activeModel(): ModelResult<ModelDescriptor?> = service.activeModel()

    suspend fun restoreActive(): ModelResult<ModelDescriptor?> = withContext(Dispatchers.IO) {
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
        val result = service.activate(id)
        if (result is ModelResult.Success) {
            attachDraftAfterTargetLoad(result.value)
            ModelRuntimeForegroundService.start(appContext)
        }
        result
    }

    private suspend fun attachDraftAfterTargetLoad(model: ModelDescriptor) {
        val draft = (draftForModel(model.id) as? ModelResult.Success)?.value ?: return
        if (Files.isRegularFile(draft.path) && Files.isReadable(draft.path)) {
            RuntimeDiagnosticsStore.recordNativeEvent(
                "SPECULATIVE_DRAFT_SAVED_NATIVE_DISABLED target=${model.id} draft=${draft.id}",
            )
        } else {
            RuntimeDiagnosticsStore.recordNativeEvent("SPECULATIVE_DRAFT_IGNORED_INVALID file=${draft.path}")
        }
    }

    suspend fun deactivate(): ModelResult<Unit> = withContext(Dispatchers.IO) {
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
    ): ModelResult<GenerationResult> = service.generate(messages, settings, onToken)

    suspend fun stopGeneration() = service.stopGeneration()

    suspend fun delete(id: String): ModelResult<Unit> = withContext(Dispatchers.IO) {
        service.deleteModel(id)
    }
}
