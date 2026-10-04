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
import com.woogit.aicore.runtime.android.LlamaCppAndroidRuntimeAdapter
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.file.Files
import java.nio.file.Path

/** Android model façade: SAF import -> GGUF validation -> durable registry -> llama.cpp/OpenCL. */
class AndroidModelManager(
    private val contentResolver: ContentResolver,
    modelDirectory: Path,
    private val runtime: LlamaCppAndroidRuntimeAdapter,
) {
    private val root = modelDirectory
    private val repository = FileModelRepository(root.resolve("models.properties"))
    private val importer = FileModelImporter(root)
    private val service = LocalModelService(importer, repository, runtime)

    suspend fun import(uri: Uri): ModelResult<ModelDescriptor> = withContext(Dispatchers.IO) {
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

    suspend fun models(): ModelResult<List<ModelDescriptor>> = service.listModels()

    suspend fun activeModel(): ModelResult<ModelDescriptor?> = service.activeModel()

    suspend fun restoreActive(): ModelResult<ModelDescriptor?> = withContext(Dispatchers.IO) {
        service.restoreActive()
    }

    suspend fun activate(id: String): ModelResult<ModelDescriptor> = withContext(Dispatchers.IO) {
        service.activate(id)
    }

    suspend fun deactivate(): ModelResult<Unit> = withContext(Dispatchers.IO) {
        service.deactivate()
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
