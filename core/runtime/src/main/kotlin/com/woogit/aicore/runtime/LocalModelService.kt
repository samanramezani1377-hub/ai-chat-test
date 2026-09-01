package com.woogit.aicore.runtime

import com.woogit.aicore.domain.ModelDescriptor
import com.woogit.aicore.domain.ModelError
import com.woogit.aicore.domain.ModelFormat
import com.woogit.aicore.domain.ModelImporter
import com.woogit.aicore.domain.ModelLifecycleManager
import com.woogit.aicore.domain.ModelRepository
import com.woogit.aicore.domain.ModelResult
import com.woogit.aicore.domain.ModelState
import com.woogit.aicore.domain.RuntimeCompatibility
import com.woogit.aicore.domain.ValidationStatus
import java.nio.file.Files
import java.util.concurrent.atomic.AtomicReference

/** Coordinates import, persistence and activation without exposing UI concerns. */
class LocalModelService(
    private val importer: ModelImporter,
    private val repository: ModelRepository,
    private val runtime: RuntimeAdapter
) : ModelLifecycleManager {
    private val activeId = AtomicReference<String?>(null)

    suspend fun importModel(source: java.nio.file.Path): ModelResult<ModelDescriptor> {
        val imported = importer.import(source)
        val model = when (imported) {
            is ModelResult.Success -> imported.value
            is ModelResult.Failure -> return imported
        }
        return when (val registered = repository.register(model)) {
            is ModelResult.Success -> registered.let { ModelResult.Success(model) }
            is ModelResult.Failure -> {
                Files.deleteIfExists(model.path)
                registered
            }
        }
    }

    suspend fun listModels(): ModelResult<List<ModelDescriptor>> = repository.list()

    suspend fun getModel(id: String): ModelResult<ModelDescriptor?> = repository.get(id)

    override suspend fun activate(id: String): ModelResult<ModelDescriptor> {
        val model = when (val result = repository.get(id)) {
            is ModelResult.Success -> result.value
            is ModelResult.Failure -> return result
        } ?: return ModelResult.Failure(ModelError.InvalidModel("Model is not registered"))

        if (model.format != ModelFormat.GGUF || model.validation != ValidationStatus.VALID) {
            return ModelResult.Failure(ModelError.InvalidModel("Model is not valid for activation"))
        }

        val previous = activeId.get()
        return try {
            runtime.load(model)
            repository.setActive(id)
            activeId.set(id)
            if (previous != null && previous != id) {
                // The concrete runtime owns unloading semantics; activation is serialized here.
                runtime.unload()
            }
            ModelResult.Success(model.copy(
                state = ModelState.ACTIVE,
                runtimeCompatibility = RuntimeCompatibility(true, null)
            ))
        } catch (t: OutOfMemoryError) {
            ModelResult.Failure(ModelError.OutOfMemory("Not enough memory to load model", t))
        } catch (t: Throwable) {
            ModelResult.Failure(ModelError.LoadFailed("Unable to load model", t))
        }
    }

    override suspend fun deactivate(): ModelResult<Unit> {
        return try {
            runtime.unload()
            repository.setActive(null)
            activeId.set(null)
            ModelResult.Success(Unit)
        } catch (t: Throwable) {
            ModelResult.Failure(ModelError.LoadFailed("Unable to deactivate model", t))
        }
    }

    override suspend fun unload(): ModelResult<Unit> = deactivate()
}
