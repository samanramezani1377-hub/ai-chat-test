package com.samanramezani.aichattest

import android.content.ContentResolver
import android.net.Uri
import com.woogit.aicore.domain.ModelDescriptor
import com.woogit.aicore.domain.ModelError
import com.woogit.aicore.domain.ModelResult
import com.woogit.aicore.runtime.FileModelImporter
import com.woogit.aicore.runtime.FileModelRepository
import com.woogit.aicore.runtime.LocalModelService
import com.woogit.aicore.runtime.RuntimeAdapter
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption

/** Android bridge for the user-selected GGUF file. The model is copied into app-private storage. */
class AndroidModelManager(
    private val contentResolver: ContentResolver,
    private val modelDirectory: Path,
    runtime: RuntimeAdapter,
) {
    private val service = LocalModelService(
        importer = FileModelImporter(modelDirectory),
        repository = FileModelRepository(modelDirectory.resolve("registry.properties")),
        runtime = runtime,
    )

    suspend fun import(uri: Uri): ModelResult<ModelDescriptor> {
        Files.createDirectories(modelDirectory)
        val source = Files.createTempFile(modelDirectory, "selected-", ".gguf")
        return try {
            val input = contentResolver.openInputStream(uri)
                ?: return ModelResult.Failure(ModelError.FileAccess("Selected model file could not be opened"))
            input.use { Files.copy(it, source, StandardCopyOption.REPLACE_EXISTING) }
            service.importModel(source)
        } catch (t: Throwable) {
            ModelResult.Failure(ModelError.Storage("Unable to read selected model file", t))
        } finally {
            Files.deleteIfExists(source)
        }
    }

    suspend fun models(): ModelResult<List<ModelDescriptor>> = service.listModels()

    suspend fun activeModel(): ModelResult<ModelDescriptor?> {
        val activeId = service.activeModelId() ?: return ModelResult.Success(null)
        return service.getModel(activeId)
    }

    suspend fun activate(id: String): ModelResult<ModelDescriptor> = service.activate(id)

    suspend fun deactivate(): ModelResult<Unit> = service.deactivate()
}
