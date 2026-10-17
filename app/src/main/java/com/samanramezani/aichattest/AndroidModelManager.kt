package com.samanramezani.aichattest

import android.content.ContentResolver
import android.net.Uri
import com.woogit.aicore.domain.ChatMessage
import com.woogit.aicore.domain.GenerationResult
import com.woogit.aicore.domain.InferenceSettings
import com.woogit.aicore.domain.ModelDescriptor
import com.woogit.aicore.domain.ModelError
import com.woogit.aicore.domain.ModelResult
import com.woogit.aicore.runtime.FileModelRepository
import com.woogit.aicore.runtime.LocalModelService
import com.woogit.aicore.runtime.OnnxGenAiModelImporter
import com.woogit.aicore.runtime.android.OnnxRuntimeGenAiAndroidRuntimeAdapter
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.file.Files
import java.nio.file.Path

/** Android model façade: SAF ZIP import -> ONNX GenAI package validation -> durable registry -> ONNX Runtime GenAI. */
class AndroidModelManager(
    private val contentResolver: ContentResolver,
    private val appContext: android.content.Context,
    modelDirectory: Path,
    private val runtime: OnnxRuntimeGenAiAndroidRuntimeAdapter,
) {
    private val root = modelDirectory
    private val repository = FileModelRepository(root.resolve("models.properties"))
    private val importer = OnnxGenAiModelImporter(root)
    private val service = LocalModelService(importer, repository, runtime)

    suspend fun import(uri: Uri): ModelResult<ModelDescriptor> = withContext(Dispatchers.IO) {
        try {
            Files.createDirectories(root)
            val staged = Files.createTempFile(root, "import-", ".zip")
            try {
                val input = contentResolver.openInputStream(uri)
                    ?: return@withContext ModelResult.Failure(ModelError.FileAccess("Unable to open the selected ONNX GenAI ZIP package"))
                input.use { source -> Files.newOutputStream(staged).use { target -> source.copyTo(target) } }
                service.importModel(staged)
            } catch (t: Throwable) {
                ModelResult.Failure(ModelError.Storage("Unable to stage the selected ONNX GenAI package", t))
            } finally {
                runCatching { Files.deleteIfExists(staged) }
            }
        } catch (t: Throwable) {
            ModelResult.Failure(ModelError.Storage("Unable to prepare model storage", t))
        }
    }

    suspend fun models(): ModelResult<List<ModelDescriptor>> = service.listModels()

    suspend fun activeModel(): ModelResult<ModelDescriptor?> = service.activeModel()

    suspend fun restoreActive(): ModelResult<ModelDescriptor?> = withContext(Dispatchers.IO) {
        val result = service.restoreActive()
        if (result is ModelResult.Success && result.value != null) {
            ModelRuntimeForegroundService.start(appContext)
        }
        result
    }

    suspend fun activate(id: String): ModelResult<ModelDescriptor> = withContext(Dispatchers.IO) {
        val result = service.activate(id)
        if (result is ModelResult.Success) ModelRuntimeForegroundService.start(appContext)
        result
    }

    suspend fun deactivate(): ModelResult<Unit> = withContext(Dispatchers.IO) {
        val result = service.deactivate()
        if (result is ModelResult.Success) ModelRuntimeForegroundService.stop(appContext)
        result
    }

    suspend fun unload(): ModelResult<Unit> = deactivate()

    suspend fun generate(
        messages: List<ChatMessage>,
        settings: InferenceSettings,
        onToken: suspend (String) -> Unit = {},
    ): ModelResult<GenerationResult> = service.generate(messages, settings, onToken)

    suspend fun stopGeneration() = service.stopGeneration()

    suspend fun delete(id: String): ModelResult<Unit> = withContext(Dispatchers.IO) {
        service.deleteModel(id)
    }
}
