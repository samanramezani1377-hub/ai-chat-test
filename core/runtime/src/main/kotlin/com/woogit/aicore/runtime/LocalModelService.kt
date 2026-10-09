package com.woogit.aicore.runtime

import com.woogit.aicore.domain.ChatMessage
import com.woogit.aicore.domain.GenerationRequest
import com.woogit.aicore.domain.GenerationResult
import com.woogit.aicore.domain.InferenceSettings
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
import kotlinx.coroutines.Job
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/** Coordinates import, persistence, activation and real local generation. */
class LocalModelService(
    private val importer: ModelImporter,
    private val repository: ModelRepository,
    private val runtime: RuntimeAdapter
) : ModelLifecycleManager {
    /** Runtime state transitions are always protected by this mutex. */
    private val runtimeMutex = Mutex()
    /** All lifecycle operations are serialized by this mutex. */
    private val generationMutex = Mutex()
    private var activeId: String? = null
    @Volatile private var generationJob: Job? = null

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
    suspend fun activeModel(): ModelResult<ModelDescriptor?> = repository.getActive()

    suspend fun restoreActive(): ModelResult<ModelDescriptor?> {
        val active = when (val result = repository.getActive()) {
            is ModelResult.Success -> result.value
            is ModelResult.Failure -> return result
        } ?: return ModelResult.Success(null)
        return when (val activated = activate(active.id)) {
            is ModelResult.Success -> ModelResult.Success(activated.value)
            is ModelResult.Failure -> activated
        }
    }

    override suspend fun activate(id: String): ModelResult<ModelDescriptor> =
        generationMutex.withLock {
            stopGenerationUnsafe()
            runtimeMutex.withLock {
                val model = when (val result = repository.get(id)) {
                    is ModelResult.Success -> result.value
                    is ModelResult.Failure -> return@withLock result
                } ?: return@withLock ModelResult.Failure(ModelError.InvalidModel("Model is not registered"))

                if (model.format != ModelFormat.GGUF || model.validation != ValidationStatus.VALID) {
                    return@withLock ModelResult.Failure(ModelError.InvalidModel("Model is not valid for activation"))
                }

                try {
                    if (activeId == id) {
                        return@withLock ModelResult.Success(
                            model.copy(
                                state = ModelState.ACTIVE,
                                runtimeCompatibility = RuntimeCompatibility(true, null),
                            )
                        )
                    }
                    if (activeId != null) {
                        runtime.unload()
                    }
                    runtime.load(model)
                    when (val persisted = repository.setActive(id)) {
                        is ModelResult.Success -> {
                            activeId = id
                            ModelResult.Success(model.copy(state = ModelState.ACTIVE, runtimeCompatibility = RuntimeCompatibility(true, null)))
                        }
                        is ModelResult.Failure -> {
                            runtime.unload()
                            activeId = null
                            persisted
                        }
                    }
                } catch (t: OutOfMemoryError) {
                    // A failed target load must never leave a half-loaded native runtime
                    // or a persisted active-model pointer behind.
                    runCatching { runtime.unload() }
                    runCatching { repository.setActive(null) }
                    activeId = null
                    ModelResult.Failure(ModelError.OutOfMemory("Not enough memory to load model", t))
                } catch (t: Throwable) {
                    // Always tear down both the previous/partial native state and the
                    // persisted active state before reporting activation failure.
                    runCatching { runtime.unload() }
                    runCatching { repository.setActive(null) }
                    activeId = null
                    ModelResult.Failure(ModelError.LoadFailed("Unable to load model", t))
                }
            }
        }

    suspend fun generate(
        messages: List<ChatMessage>,
        settings: InferenceSettings,
        onToken: suspend (String) -> Unit = {}
    ): ModelResult<GenerationResult> {
        require(messages.isNotEmpty()) { "messages must not be empty" }

        return generationMutex.withLock {
            val persisted = runtimeMutex.withLock {
                when (val result = repository.getActive()) {
                    is ModelResult.Success -> {
                        val model = result.value
                        // Generation must never implicitly activate/reload a model.
                        // The runtime is loaded explicitly by activate()/restoreActive().
                        // Re-loading here can allocate a second native model/context while
                        // the previous generation has just completed, which is unsafe on
                        // mobile OpenCL and was the source of second-message crashes.
                        model
                    }
                    is ModelResult.Failure -> return@withLock result
                }
            } ?: return@withLock ModelResult.Failure(ModelError.RuntimeUnavailable("No model is active"))

            try {
                generationJob = currentCoroutineContext()[Job]
                currentCoroutineContext().ensureActive()
                val result = runtime.generate(GenerationRequest(messages, settings), onToken)
                currentCoroutineContext().ensureActive()
                ModelResult.Success(result)
            } catch (t: OutOfMemoryError) {
                ModelResult.Failure(ModelError.OutOfMemory("Not enough memory for generation", t))
            } catch (t: Throwable) {
                if (t is kotlinx.coroutines.CancellationException) throw t
                ModelResult.Failure(ModelError.Inference("Local generation failed", t))
            } finally {
                generationJob = null
            }
        }
    }

    suspend fun stopGeneration() {
        // Generation holds generationMutex for its entire lifetime. Waiting for that
        // mutex here prevents the stop signal from ever reaching a running decode.
        // The runtime stop API is specifically designed to interrupt inference safely.
        if (generationJob != null) {
            runCatching { runtime.stopGeneration() }
        }
    }

    private suspend fun stopGenerationUnsafe() {
        generationJob?.cancel()
        runtimeMutex.withLock {
            runCatching { runtime.stopGeneration() }
        }
    }

    override suspend fun deactivate(): ModelResult<Unit> =
        generationMutex.withLock {
            stopGenerationUnsafe()
            runtimeMutex.withLock {
                try {
                    runtime.unload()
                    when (val persisted = repository.setActive(null)) {
                        is ModelResult.Success -> { activeId = null; persisted }
                        is ModelResult.Failure -> persisted
                    }
                } catch (t: Throwable) {
                    ModelResult.Failure(ModelError.LoadFailed("Unable to deactivate model", t))
                }
            }
        }

    override suspend fun unload(): ModelResult<Unit> = deactivate()

    suspend fun deleteModel(id: String): ModelResult<Unit> =
        generationMutex.withLock {
            stopGenerationUnsafe()
            runtimeMutex.withLock {
                val active = repository.getActive()
                if (active is ModelResult.Success && active.value?.id == id) {
                    runtime.unload()
                    repository.setActive(null)
                    activeId = null
                }
                val model = when (val result = repository.get(id)) {
                    is ModelResult.Success -> result.value
                    is ModelResult.Failure -> return@withLock result
                } ?: return@withLock ModelResult.Failure(ModelError.InvalidModel("Model is not registered"))
                java.nio.file.Files.deleteIfExists(model.path)
                repository.unregister(id)
            }
        }

    fun activeModelId(): String? = activeId
}
