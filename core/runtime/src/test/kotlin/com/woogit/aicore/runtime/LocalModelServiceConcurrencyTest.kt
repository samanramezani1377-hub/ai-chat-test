package com.woogit.aicore.runtime

import com.woogit.aicore.domain.ChatMessage
import com.woogit.aicore.domain.GenerationRequest
import com.woogit.aicore.domain.GenerationResult
import com.woogit.aicore.domain.InferenceSettings
import com.woogit.aicore.domain.ModelDescriptor
import com.woogit.aicore.domain.ModelError
import com.woogit.aicore.domain.ModelFormat
import com.woogit.aicore.domain.ModelImporter
import com.woogit.aicore.domain.ModelMetadata
import com.woogit.aicore.domain.ModelRepository
import com.woogit.aicore.domain.ModelResult
import com.woogit.aicore.domain.ModelState
import com.woogit.aicore.domain.Quantization
import com.woogit.aicore.domain.RuntimeCompatibility
import com.woogit.aicore.domain.RuntimeInfo
import com.woogit.aicore.domain.ValidationStatus
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.awaitCancellation
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import java.nio.file.Path
import java.util.concurrent.atomic.AtomicInteger
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class LocalModelServiceConcurrencyTest {
    @Test
    fun repeatedGenerationDoesNotReloadTheActiveModel() = runBlocking {
        val runtime = RecordingRuntime()
        val repository = RecordingRepository(model())
        val service = LocalModelService(NoOpImporter(), repository, runtime)

        assertTrue(service.activate("model-a") is ModelResult.Success<*>)
        assertTrue(service.generate(listOf(userMessage("one")), InferenceSettings(maxNewTokens = 1)) is ModelResult.Success<*>)
        assertTrue(service.generate(listOf(userMessage("two")), InferenceSettings(maxNewTokens = 1)) is ModelResult.Success<*>)

        assertEquals(1, runtime.loadCount.get())
        assertEquals(0, runtime.unloadCount.get())
    }

    @Test
    fun persistedActiveModelMustAlsoBeLoadedBeforeGeneration() = runBlocking {
        val runtime = RecordingRuntime()
        val repository = RecordingRepository(model()).apply { activeId = "model-a" }
        val service = LocalModelService(NoOpImporter(), repository, runtime)

        val result = service.generate(
            listOf(userMessage("must not run")),
            InferenceSettings(maxNewTokens = 1),
        )

        assertTrue(result is ModelResult.Failure)
        assertEquals(0, runtime.generateCount.get(), "native generation must not run for an unloaded model")
    }

    @Test
    fun cancelledActivationUnloadsPartialRuntimeBeforeNextActivation() = runBlocking {
        val runtime = RecordingRuntime().apply { blockLoad = true }
        val repository = RecordingRepository(model())
        val service = LocalModelService(NoOpImporter(), repository, runtime)

        val activation = async { service.activate("model-a") }
        runtime.loadStarted.await()
        activation.cancelAndJoin()

        assertEquals(1, runtime.unloadCount.get(), "cancelled activation must unload partially loaded native state")
        assertEquals(null, service.activeModelId())
        assertEquals(null, repository.activeId)

        runtime.blockLoad = false
        assertTrue(service.activate("model-a") is ModelResult.Success<*>)
        assertEquals(2, runtime.loadCount.get())
        assertEquals("model-a", service.activeModelId())
    }

    @Test
    fun runtimeConfigurationChangeBypassesSameModelActivationShortcut() = runBlocking {
        val runtime = RecordingRuntime()
        val repository = RecordingRepository(model())
        val service = LocalModelService(NoOpImporter(), repository, runtime)

        assertTrue(service.activate("model-a") is ModelResult.Success<*>)
        assertEquals(1, runtime.loadCount.get())

        service.requestRuntimeReload()
        assertTrue(service.activate("model-a") is ModelResult.Success<*>)

        assertEquals(2, runtime.loadCount.get(), "the adapter must be allowed to re-check changed GPU preferences")
        assertEquals("model-a", service.activeModelId())
        assertEquals("model-a", repository.activeId)
    }

    @Test
    fun concurrentLifecycleOperationsAreSerialized() = runBlocking {
        val runtime = RecordingRuntime()
        val repository = RecordingRepository(model())
        val service = LocalModelService(NoOpImporter(), repository, runtime)

        val results = listOf(
            async { service.activate("model-a") },
            async { service.deactivate() },
            async { service.activate("model-a") },
            async { service.deactivate() },
        ).awaitAll()

        assertEquals(4, results.size)
        assertTrue(results.all { it is ModelResult.Success<*> })
        assertEquals(0, runtime.concurrentOperations.get())
        assertEquals(1, runtime.maxConcurrentOperations.get())
        assertEquals(null, service.activeModelId())
        assertEquals(null, repository.activeId)
    }

    @Test
    fun stopGenerationCancelsActiveGenerationWithoutWaitingForItsMutex() = runBlocking {
        val runtime = RecordingRuntime().apply { blockGeneration = true }
        val repository = RecordingRepository(model())
        val service = LocalModelService(NoOpImporter(), repository, runtime)

        assertTrue(service.activate("model-a") is ModelResult.Success<*>)
        runtime.stopCount.set(0)

        val generation = async {
            service.generate(listOf(userMessage("wait")), InferenceSettings(maxNewTokens = 8))
        }
        runtime.generationStarted.await()

        service.stopGeneration()

        assertTrue(generation.isCancelled, "stopGeneration must cancel the active generation job")
        assertEquals(1, runtime.stopCount.get(), "native stop must be requested before waiting for generation to finish")
    }

    private fun userMessage(text: String) =
        ChatMessage(ChatMessage.Role.USER, text)

    private fun model() = ModelDescriptor(
        id = "model-a",
        displayName = "Test model",
        path = Path.of("/tmp/model-a.gguf"),
        format = ModelFormat.GGUF,
        quantization = Quantization.Q6_K,
        sizeBytes = 1024,
        metadata = ModelMetadata(architecture = "qwen2"),
        validation = ValidationStatus.VALID,
        runtimeCompatibility = RuntimeCompatibility(true),
        state = ModelState.READY,
    )

    private class NoOpImporter : ModelImporter {
        override suspend fun import(source: Path): ModelResult<ModelDescriptor> =
            ModelResult.Failure(ModelError.InvalidModel("Not used by this test"))
    }

    private class RecordingRepository(private val model: ModelDescriptor) : ModelRepository {
        var activeId: String? = null

        override suspend fun register(model: ModelDescriptor): ModelResult<Unit> = ModelResult.Success(Unit)
        override suspend fun get(id: String): ModelResult<ModelDescriptor?> =
            ModelResult.Success(if (id == model.id) model else null)
        override suspend fun list(): ModelResult<List<ModelDescriptor>> = ModelResult.Success(listOf(model))
        override suspend fun getActive(): ModelResult<ModelDescriptor?> =
            ModelResult.Success(activeId?.let { model.takeIf { it.id == activeId } })
        override suspend fun setActive(id: String?): ModelResult<Unit> {
            activeId = id
            return ModelResult.Success(Unit)
        }
        override suspend fun unregister(id: String): ModelResult<Unit> = ModelResult.Success(Unit)
    }

    private class RecordingRuntime : RuntimeAdapter {
        val concurrentOperations = AtomicInteger(0)
        val maxConcurrentOperations = AtomicInteger(0)
        val loadCount = AtomicInteger(0)
        val unloadCount = AtomicInteger(0)
        val stopCount = AtomicInteger(0)
        val generationStarted = CompletableDeferred<Unit>()
        val loadStarted = CompletableDeferred<Unit>()
        val generateCount = AtomicInteger(0)

        @Volatile
        var blockGeneration = false

        @Volatile
        var blockLoad = false

        override suspend fun load(model: ModelDescriptor) {
            loadCount.incrementAndGet()
            if (blockLoad) {
                loadStarted.complete(Unit)
                awaitCancellation()
            }
            operation()
        }

        override suspend fun unload() {
            unloadCount.incrementAndGet()
            operation()
        }

        override suspend fun generate(
            request: GenerationRequest,
            onToken: suspend (String) -> Unit,
        ): GenerationResult {
            generateCount.incrementAndGet()
            if (blockGeneration) {
                generationStarted.complete(Unit)
                awaitCancellation()
            }
            onToken("ok")
            return GenerationResult(text = "ok")
        }

        override suspend fun stopGeneration() {
            stopCount.incrementAndGet()
            operation()
        }

        override fun runtimeInfo(): RuntimeInfo = RuntimeInfo("test", "1", "test")

        private suspend fun operation() {
            val current = concurrentOperations.incrementAndGet()
            maxConcurrentOperations.updateAndGet { maxOf(it, current) }
            try {
                delay(20)
            } finally {
                concurrentOperations.decrementAndGet()
            }
        }
    }
}
