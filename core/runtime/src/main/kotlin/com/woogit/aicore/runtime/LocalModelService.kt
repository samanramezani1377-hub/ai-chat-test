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
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.withContext
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

    // A failed or interrupted native transition leaves the resident state uncertain.
    // Never load another model until a subsequent unload has completed successfully.
    @Volatile
    private var runtimeStateUnknown = false

    @Volatile
    private var runtimeReloadRequested = false

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

    /** Ask the next activation of the same model to re-check runtime configuration. */
    fun requestRuntimeReload() {
        runtimeReloadRequested = true
    }

    override suspend fun activate(id: String): ModelResult<ModelDescriptor> =
        withLifecycleExclusive {
            runtimeMutex.withLock runtimeLock@{
                val model = when (val result = repository.get(id)) {
                    is ModelResult.Success -> result.value
                    is ModelResult.Failure -> return@runtimeLock result
                } ?: return@runtimeLock ModelResult.Failure(
                    ModelError.InvalidModel("Model is not registered"),
                )

                if (model.format != ModelFormat.GGUF || model.validation != ValidationStatus.VALID) {
                    return@runtimeLock ModelResult.Failure(
                        ModelError.InvalidModel("Model is not valid for activation"),
                    )
                }

                try {
                    if (activeId == id && !runtimeReloadRequested) {
                        return@runtimeLock ModelResult.Success(
                            model.copy(
                                state = ModelState.ACTIVE,
                                runtimeCompatibility = RuntimeCompatibility(true, null),
                            ),
                        )
                    }

                    if (runtimeStateUnknown) {
                        runtime.unload()
                        runtimeStateUnknown = false
                    }
                    if (activeId != null && activeId != id) {
                        // Clear advertised state before unload so cancellation cannot leave a stale active ID.
                        activeId = null
                        runtimeStateUnknown = true
                        runtime.unload()
                        runtimeStateUnknown = false
                    }
                    runtimeStateUnknown = true
                    runtime.load(model)
                    runtimeStateUnknown = false

                    when (val persisted = repository.setActive(id)) {
                        is ModelResult.Success -> {
                            activeId = id
                            runtimeReloadRequested = false
                            ModelResult.Success(
                                model.copy(
                                    state = ModelState.ACTIVE,
                                    runtimeCompatibility = RuntimeCompatibility(true, null),
                                ),
                            )
                        }
                        is ModelResult.Failure -> {
                            resetRuntimeState()
                            persisted
                        }
                    }
                } catch (cancelled: CancellationException) {
                    resetRuntimeState()
                    throw cancelled
                } catch (error: OutOfMemoryError) {
                    resetRuntimeState()
                    ModelResult.Failure(
                        ModelError.OutOfMemory("Not enough memory to load model", error),
                    )
                } catch (error: Throwable) {
                    resetRuntimeState()
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

                val activeModel = when (val result = runtimeMutex.withLock { repository.getActive() }) {
                    is ModelResult.Failure -> return@generationLock result
                    is ModelResult.Success -> result.value
                } ?: return@generationLock ModelResult.Failure(
                    ModelError.RuntimeUnavailable("No model is active"),
                )

                // A persisted selection is not proof that native weights are resident.
                // Require the runtime lifecycle state and repository selection to agree.
                if (activeId != activeModel.id) {
                    return@generationLock ModelResult.Failure(
                        ModelError.RuntimeUnavailable("The active model is not loaded in the runtime"),
                    )
                }

                // A stop request may arrive while the active-model record is read.
                currentCoroutineContext().ensureActive()
                if (stopInProgress.get()) {
                    return@generationLock ModelResult.Failure(
                        ModelError.RuntimeUnavailable("Generation was stopped before native execution"),
                    )
                }

                // Generation never performs implicit activation or reload. The runtime
                // must already have been activated explicitly by its lifecycle owner.

                val generated = runtime.generate(GenerationRequest(messages, settings), onToken)
                currentCoroutineContext().ensureActive()
                ModelResult.Success(generated)
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
                    // Fail closed before native unload: cancellation or unload failure
                    // must never leave this service advertising weights as resident.
                    activeId = null
                    runtimeStateUnknown = true
                    runtime.unload()
                    runtimeStateUnknown = false
                    when (val persisted = repository.setActive(null)) {
                        is ModelResult.Success -> persisted
                        is ModelResult.Failure -> {
                            resetRuntimeState()
                            persisted
                        }
                    }
                } catch (cancelled: CancellationException) {
                    resetRuntimeState()
                    throw cancelled
                } catch (error: Throwable) {
                    resetRuntimeState()
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
                val active = when (val result = repository.getActive()) {
                    is ModelResult.Success -> result.value
                    is ModelResult.Failure -> return@runtimeLock result
                }
                if (active?.id == id) {
                    try {
                        activeId = null
                        runtimeStateUnknown = true
                        runtime.unload()
                        runtimeStateUnknown = false
                        when (val cleared = repository.setActive(null)) {
                            is ModelResult.Success -> Unit
                            is ModelResult.Failure -> {
                                resetRuntimeState()
                                return@runtimeLock cleared
                            }
                        }
                    } catch (cancelled: CancellationException) {
                        resetRuntimeState()
                        throw cancelled
                    } catch (error: Throwable) {
                        resetRuntimeState()
                        return@runtimeLock ModelResult.Failure(
                            ModelError.LoadFailed("Unable to unload active model before deletion", error),
                        )
                    }
                }

                val model = when (val result = repository.get(id)) {
                    is ModelResult.Success -> result.value
                    is ModelResult.Failure -> return@runtimeLock result
                } ?: return@runtimeLock ModelResult.Failure(
                    ModelError.InvalidModel("Model is not registered"),
                )

                Files.deleteIfExists(model.path)
                repository.unregister(id)
            }
        }

    /**
     * Roll back an interrupted or failed lifecycle transition. Cleanup must finish even
     * after caller cancellation; an unsuccessful unload keeps the state marked unknown
     * so the next activation retries cleanup before loading anything.
     */
    private suspend fun resetRuntimeState() {
        activeId = null
        runtimeReloadRequested = false
        runtimeStateUnknown = true
        withContext(NonCancellable) {
            try {
                runtime.unload()
                runtimeStateUnknown = false
            } catch (_: Throwable) {
                // Keep runtimeStateUnknown=true and fail closed.
            }
            try {
                repository.setActive(null)
            } catch (_: Throwable) {
                // activeId remains null, so generation cannot use stale persisted state.
            }
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
                try {
                    runtime.stopGeneration()
                } catch (cancelled: CancellationException) {
                    throw cancelled
                } catch (_: Throwable) {
                    // Native stop is best-effort; the generation job is still cancelled and
                    // the mutex below remains the definitive barrier before lifecycle mutation.
                }
                generationMutex.withLock { block() }
            } finally {
                stopInProgress.set(false)
            }
        }
}
