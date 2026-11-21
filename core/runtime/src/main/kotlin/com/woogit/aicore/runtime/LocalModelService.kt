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
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Coordinates model persistence and runtime access.
 *
 * Generation is serialized independently from lifecycle operations. A lifecycle
 * operation first requests native cancellation, then waits for the active
 * generation to leave its critical section before touching the runtime.
 */
class LocalModelService(
    private val importer: ModelImporter,
    private val repository: ModelRepository,
    private val runtime: RuntimeAdapter,
) : ModelLifecycleManager {
    private val lifecycleMutex = Mutex()
    private val generationMutex = Mutex()
    private val runtimeMutex = Mutex()
    private val stopInProgress = AtomicBoolean(false)

    @Volatile
    private var activeId: String? = null

    @Volatile
    private var generationJob: Job? = null

    suspend fun importModel(source: Path): ModelResult<ModelDescriptor> {
        val model = when (val imported = importer.import(source)) {
            is ModelResult.Success -> imported.value
            is ModelResult.Failure -> return imported
        }

        return when (val registered = repository.register(model)) {
            is ModelResult.Success -> ModelResult.Success(model)
            is ModelResult.Failure -> {
                Files.deleteIfExists(model.path)
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
        withLifecycleExclusive {
            runtimeMutex.withLock runtimeLock@{
                val model = when (val result = repository.get(id)) {
                    is ModelResult.Success -> result.value
                    is ModelResult.Failure -> return@runtimeLock result
                } ?: return@runtimeLock ModelResult.Failure(("Model is not registered"),
                )

                if (model.format != ModelFormat.GGUF || model.validation != ValidationStatus.VALID) {
                    return@withLock ModelResult.Failure(
                        ModelError.InvalidModel("Model is not valid for activation"),
                    )
                }

                try {
                    if (activeId == id) {
                        return@withLock ModelResult.Success(
                            model.copy(
                                state = ModelState.ACTIVE,
                                runtimeCompatibility = RuntimeCompatibility(true, null),
                            ),
                        )
                    }

                    if (activeId != null) runtime.unload()
                    runtime.load(model)

                    when (val persisted = repository.setActive(id)) {
                        is ModelResult.Success -> {
                            activeId = id
                            ModelResult.Success(
                                model.copy(
                                    state = ModelState.ACTIVE,
                                    runtimeCompatibility = RuntimeCompatibility(true, null),
                                ),
                            )
                        }
                        is ModelResult.Failure -> {
                            runCatching { runtime.unload() }
                            activeId = null
                            persisted
                        }
                    }
                } catch (cancelled: CancellationException) {
                    throw cancelled
                } catch (error: OutOfMemoryError) {
                    runCatching { runtime.unload() }
                    runCatching { repository.setActive(null) }
                    activeId = null
                    ModelResult.Failure(
                        ModelError.OutOfMemory("Not enough memory to load model", error),
                    )
                } catch (error: Throwable) {
                    runCatching { runtime.unload() }
                    runCatching { repository.setActive(null) }
                    activeId = null
                    ModelResult.Failure(
                        ModelError.LoadFailed("Unable to load model", error),
                    )
                }
            }
        }

    suspend fun generate(
        messages: List<ChatMessage>,
        settings: InferenceSettings,
        onToken: suspend (String) -> Unit = {},
    ): ModelResult<GenerationResult> {
        require(messages.isNotEmpty()) { "messages must not be empty" }

        return generationMutex.withLock generationLock@{
            if (stopInProgress.get()) {
                return@generationLock ModelResult.Failure(
                    ModelError.RuntimeUnavailable("Generation is stopping for a lifecycle operation"),
                )
            }

            val currentJob = currentCoroutineContext()[Job]
            generationJob = currentJob

            try {
                currentCoroutineContext().ensureActive()

                val activeModel = runtimeMutex.withLock {
                    when (val result = repository.getActive()) {
                        is ModelResult.Success -> result.value
                        is ModelResult.Failure -> return@generationLock result
                    }
                } ?: return@generationLock ModelResult.Failure(
                    ModelError.RuntimeUnavailable("No model is active"),
                )

                // A stop request may arrive while the active-model record is read.
                currentCoroutineContext().ensureActive()
                if (stopInProgress.get()) {
                    return@generationLock ModelResult.Failure(
                        ModelError.RuntimeUnavailable("Generation was stopped before native execution"),
                    )
                }

                // Generation never performs implicit activation or reload. The runtime
                // must already have been activated explicitly by its lifecycle owner.

                ModelResult.Success(runtime.generate(GenerationRequest(messages, settings), onToken))
            } catch (error: OutOfMemoryError) {
                ModelResult.Failure(
                    ModelError.OutOfMemory("Not enough memory for generation", error),
                )
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (error: Throwable) {
                ModelResult.Failure(ModelError.Inference("Local generation failed", error))
            } finally {
                if (generationJob === currentJob) generationJob = null
            }
        }
    }

    /**
     * Stops native decoding without waiting for the generation mutex first.
     * Waiting for that mutex before requesting stop would deadlock cancellation:
     * generation owns it until the native flow exits.
     */
    suspend fun stopGeneration() {
        withLifecycleExclusive { Unit }
    }

    override suspend fun deactivate(): ModelResult<Unit> =
        withLifecycleExclusive {
            runtimeMutex.withLock {
                try {
                    runtime.unload()
                    when (val persisted = repository.setActive(null)) {
                        is ModelResult.Success -> {
                            activeId = null
                            persisted
                        }
                        is ModelResult.Failure -> persisted
                    }
                } catch (cancelled: CancellationException) {
                    throw cancelled
                } catch (error: Throwable) {
                    ModelResult.Failure(
                        ModelError.LoadFailed("Unable to deactivate model", error),
                    )
                }
            }
        }

    override suspend fun unload(): ModelResult<Unit> = deactivate()

    suspend fun deleteModel(id: String): ModelResult<Unit> =
        withLifecycleExclusive {
            runtimeMutex.withLock runtimeLock@{
                val active = repository.getActive()
                if (active is ModelResult.Success && active.value?.id == id) {
                    runtime.unload()
                    when (val cleared = repository.setActive(null)) {
                        is ModelResult.Success -> activeId = null
                        is ModelResult.Failure -> return@runtimeLock cleared
                    }
                }

                val model = when (val result = repository.get(id)) {
                    is ModelResult.Success -> result.value
                    is ModelResult.Failure -> return@runtimeLock result
                } ?: return@runtimeLock ModelResult.Failure(("Model is not registered"),
                )

                Files.deleteIfExists(model.path)
                repository.unregister(id)
            }
        }

    fun activeModelId(): String? = activeId

    /**
     * Serializes lifecycle changes, signals cancellation before waiting for the
     * generation lock, and keeps new generations out until the transition finishes.
     */
    private suspend fun <T> withLifecycleExclusive(block: suspend () -> T): T =
        lifecycleMutex.withLock {
            stopInProgress.set(true)
            try {
                generationJob?.cancel()
                runCatching { runtime.stopGeneration() }
                generationMutex.withLock { block() }
            } finally {
                stopInProgress.set(false)
            }
        }
}
