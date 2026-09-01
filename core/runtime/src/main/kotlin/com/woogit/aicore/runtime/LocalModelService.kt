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
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/** Coordinates import, persistence and activation without exposing UI concerns. */
class LocalModelService(
    private val importer: ModelImporter,
    private val repository: ModelRepository,
    private val runtime: RuntimeAdapter
) : ModelLifecycleManager {
    private val lifecycleMutex = Mutex()
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

    override suspend fun activate(id: String): ModelResult<ModelDescriptor> = lifecycleMutex.withLock {
        val model = when (val result = repository.get(id)) {
            is ModelResult.Success -> result.value
            is ModelResult.Failure -> return@withLock result
        } ?: return@withLock ModelResult.Failure(ModelError.InvalidModel("Model is not registered"))

        if (model.format != ModelFormat.GGUF || model.validation != ValidationStatus.VALID) {
            return@withLock ModelResult.Failure(ModelError.InvalidModel("Model is not valid for activation"))
        }

        val persistedActive = when (val result = repository.getActive()) {
            is ModelResult.Success -> result.value?.id
            is ModelResult.Failure -> return@withLock result
        }

        try {
            if (persistedActive != null && persistedActive != id) {
                runtime.unload()
                when (val cleared = repository.setActive(null)) {
                    is ModelResult.Success -> activeId = null
                    is ModelResult.Failure -> return@withLock cleared
                }
            }

            runtime.load(model)
            when (val persisted = repository.setActive(id)) {
                is ModelResult.Success -> Unit
                is ModelResult.Failure -> {
                    runtime.unload()
                    return@withLock persisted
                }
            }
            activeId = id
            ModelResult.Success(
                model.copy(
                    state = ModelState.ACTIVE,
                    runtimeCompatibility = RuntimeCompatibility(true, null)
                )
            )
        } catch (t: OutOfMemoryError) {
            activeId = null
            ModelResult.Failure(ModelError.OutOfMemory("Not enough memory to load model", t))
        } catch (t: Throwable) {
            activeId = null
            ModelResult.Failure(ModelError.LoadFailed("Unable to load model", t))
        }
    }

    override suspend fun deactivate(): ModelResult<Unit> = lifecycleMutex.withLock {
        try {
            runtime.unload()
            when (val persisted = repository.setActive(null)) {
                is ModelResult.Success -> {
                    activeId = null
                    persisted
                }
                is ModelResult.Failure -> persisted
            }
        } catch (t: Throwable) {
            ModelResult.Failure(ModelError.LoadFailed("Unable to deactivate model", t))
        }
    }

    override suspend fun unload(): ModelResult<Unit> = deactivate()

    fun activeModelId(): String? = activeId
}
