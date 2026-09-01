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

/** Coordinates import, persistence and activation without exposing UI concerns. */
class LocalModelService(
    private val importer: ModelImporter,
    private val repository: ModelRepository,
    private val runtime: RuntimeAdapter
) : ModelLifecycleManager {
    private val activationLock = Any()
    private var activeId: String? = null

    suspend fun importModel(source: java.nio.file.Path): ModelResult<ModelDescriptor> {
        val imported = importer.import(source)
        val model = when (imported) {
            is ModelResult.Success -> imported.value
            is ModelResult.Failure -> return imported
        }
        return when (val registered = repository.register(model)) {
            is ModelResult.Success -> ModelResult.Success(model)
            is ModelResult.Failure -> {
                java.nio.file.Files.deleteIfExists(model.path)
                registered
            }
        }
    }

    suspend fun listModels(): ModelResult<List<ModelDescriptor>> = repository.list()

    suspend fun getModel(id: String): ModelResult<ModelDescriptor?> = repository.get(id)

    override suspend fun activate(id: String): ModelResult<ModelDescriptor> = synchronized(activationLock) {
        val model = when (val result = repository.get(id)) {
            is ModelResult.Success -> result.value
            is ModelResult.Failure -> return@synchronized result
        } ?: return@synchronized ModelResult.Failure(ModelError.InvalidModel("Model is not registered"))

        if (model.format != ModelFormat.GGUF || model.validation != ValidationStatus.VALID) {
            return@synchronized ModelResult.Failure(ModelError.InvalidModel("Model is not valid for activation"))
        }

        try {
            // Runtime.load is the single transition point. A concrete runtime must replace
            // its previous loaded model atomically or fail without reporting success.
            runtime.load(model)
            when (val persisted = repository.setActive(id)) {
                is ModelResult.Success -> Unit
                is ModelResult.Failure -> return@synchronized persisted.toModelFailure()
            }
            activeId = id
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

    override suspend fun deactivate(): ModelResult<Unit> = synchronized(activationLock) {
        try {
            runtime.unload()
            when (val persisted = repository.setActive(null)) {
                is ModelResult.Success -> Unit
                is ModelResult.Failure -> return@synchronized persisted
            }
            activeId = null
            ModelResult.Success(Unit)
        } catch (t: Throwable) {
            ModelResult.Failure(ModelError.LoadFailed("Unable to deactivate model", t))
        }
    }

    override suspend fun unload(): ModelResult<Unit> = deactivate()

    private fun <T> ModelResult.Failure.toModelFailure(): ModelResult<T> = ModelResult.Failure(error)
}
