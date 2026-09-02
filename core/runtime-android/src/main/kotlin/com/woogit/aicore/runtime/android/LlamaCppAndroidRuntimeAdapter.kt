package com.woogit.aicore.runtime.android

import com.woogit.aicore.domain.ChatMessage
import com.woogit.aicore.domain.GenerationRequest
import com.woogit.aicore.domain.GenerationResult
import com.woogit.aicore.domain.ModelDescriptor
import com.woogit.aicore.domain.ModelError
import com.woogit.aicore.domain.ModelResult
import com.woogit.aicore.domain.RuntimeInfo
import com.woogit.aicore.runtime.RuntimeAdapter
import com.woogit.aicore.runtime.RuntimeDiagnosticsStore
import com.woogit.aicore.runtime.RuntimeMetrics
import com.woogit.aicore.runtime.RuntimeTraceEvent
import com.tensai.llamakt.ChatMessage as EngineChatMessage
import com.tensai.llamakt.LlamaEngine
import com.tensai.llamakt.SamplingParams
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.launch
import kotlinx.coroutines.channels.awaitClose
import java.util.concurrent.atomic.AtomicBoolean

/** Android llama.cpp runtime with conservative CPU+GPU offload and CPU fallback. */
class LlamaCppAndroidRuntimeAdapter(
    private val defaultContextLength: Int = 4096,
) : RuntimeAdapter, RuntimeMetrics {
    private var engine: LlamaEngine? = null
    private val stopRequested = AtomicBoolean(false)
    @Volatile private var selectedGpuLayers: Int = 0
    @Volatile private var selectedCpuThreads: Int = 2
    @Volatile private var selectedBackend: String = "CPU/NEON"
    @Volatile private var loadedContextLength: Int? = null
    @Volatile private var latestGeneration: GenerationResult? = null
    @Volatile private var latestLoadTimeMs: Long? = null

    override fun lastGeneration(): GenerationResult? = latestGeneration
    override fun lastLoadTimeMs(): Long? = latestLoadTimeMs

    override suspend fun load(model: ModelDescriptor) {
        when (val result = loadResult(model)) {
            is ModelResult.Success -> Unit
            is ModelResult.Failure -> throw RuntimeFailure(result.error)
        }
    }

    suspend fun loadResult(model: ModelDescriptor): ModelResult<Unit> {
        currentCoroutineContext().ensureActive()
        unload()
        val file = model.path.toFile()
        if (!file.isFile || !file.canRead()) {
            return ModelResult.Failure(ModelError.FileAccess("Model file cannot be read: ${file.absolutePath}"))
        }

        val startedAt = System.nanoTime()
        return try {
            val metadata = LlamaEngine.readMetadata(file.absolutePath)
            val contextLength = model.metadata.contextLength?.toInt()?.takeIf { it > 0 }
                ?: metadata?.contextLength?.toInt()?.takeIf { it > 0 }
                ?: defaultContextLength

            val probe = LlamaEngine()
            val hasGpu = probe.availableBackends().any { backend ->
                backend.type.equals("gpu", ignoreCase = true) ||
                    backend.type.equals("igpu", ignoreCase = true) ||
                    backend.description.contains("vulkan", ignoreCase = true) ||
                    backend.description.contains("opencl", ignoreCase = true)
            }
            val plan = HybridResourcePolicy.choose(
                blockCount = metadata?.blockCount ?: 0L,
                hasGpu = hasGpu,
                availableProcessors = Runtime.getRuntime().availableProcessors(),
                bigCoreCount = LlamaEngine.detectBigCoreCount(),
            )

            fun createAndLoad(gpuLayers: Int): LlamaEngine {
                val candidate = LlamaEngine()
                candidate.load(
                    path = file.absolutePath,
                    nGpuLayers = gpuLayers,
                    nCtx = contextLength,
                    nThreads = plan.cpuThreads,
                )
                return candidate
            }

            val loaded = try {
                createAndLoad(plan.gpuLayers)
            } catch (gpuFailure: Throwable) {
                if (plan.gpuLayers == 0) throw gpuFailure
                RuntimeDiagnosticsStore.recordTrace(
                    RuntimeTraceEvent.Type.GENERATION_FAILED,
                    "GPU load failed; falling back to CPU: ${gpuFailure.message}",
                )
                createAndLoad(0)
            }

            engine = loaded
            val active = loaded.activeBackend()
            selectedGpuLayers = if (plan.gpuLayers > 0 && !active.equals("CPU", ignoreCase = true)) plan.gpuLayers else 0
            selectedCpuThreads = plan.cpuThreads
            selectedBackend = if (selectedGpuLayers > 0) "Hybrid(CPU+$active)" else "CPU/NEON"
            loadedContextLength = contextLength
            latestGeneration = null
            latestLoadTimeMs = (System.nanoTime() - startedAt) / 1_000_000
            RuntimeDiagnosticsStore.recordLoaded(model, latestLoadTimeMs, runtimeInfo())
            ModelResult.Success(Unit)
        } catch (t: Throwable) {
            engine?.free()
            engine = null
            selectedGpuLayers = 0
            selectedBackend = "CPU/NEON"
            loadedContextLength = null
            latestLoadTimeMs = (System.nanoTime() - startedAt) / 1_000_000
            ModelResult.Failure(RuntimeErrorMapper.loadFailure(t, file.absolutePath))
        }
    }

    override suspend fun unload() {
        stopRequested.set(true)
        engine?.free()
        engine = null
        selectedGpuLayers = 0
        selectedBackend = "CPU/NEON"
        loadedContextLength = null
    }

    override suspend fun generate(request: GenerationRequest, onToken: suspend (String) -> Unit): GenerationResult =
        when (val result = generateResult(request, onToken)) {
            is ModelResult.Success -> result.value
            is ModelResult.Failure -> throw RuntimeFailure(result.error)
        }

    suspend fun generateResult(request: GenerationRequest, onToken: suspend (String) -> Unit): ModelResult<GenerationResult> {
        currentCoroutineContext().ensureActive()
        val runtime = engine ?: return ModelResult.Failure(ModelError.RuntimeUnavailable("No local model is loaded"))
        val settings = request.settings
        val requestedContext = settings.contextLength?.takeIf { it > 0 }
        val effectiveContext = when {
            requestedContext == null -> loadedContextLength ?: defaultContextLength
            loadedContextLength != null -> requestedContext.coerceAtMost(loadedContextLength!!)
            else -> requestedContext
        }
        val promptMessages = request.messages.map { message ->
            EngineChatMessage(role = message.role.name.lowercase(), content = message.content)
        }
        val prompt = try {
            runtime.formatChat(promptMessages, enableThinking = true)
        } catch (t: Throwable) {
            return ModelResult.Failure(ModelError.Inference(t.message ?: "Invalid conversation"))
        }

        val params = SamplingParams(
            nPredict = settings.maxNewTokens.coerceAtLeast(1),
            temperature = settings.temperature.toFloat().coerceAtLeast(0f),
            topK = (settings.topK ?: 40).coerceAtLeast(0),
            topP = (settings.topP ?: 0.9).toFloat().coerceIn(0f, 1f),
            minP = (settings.minP ?: 0.05).toFloat().coerceIn(0f, 1f),
            stopSequences = settings.stopSequences,
        )

        stopRequested.set(false)
        RuntimeDiagnosticsStore.recordTrace(RuntimeTraceEvent.Type.GENERATION_STARTED, "context=$effectiveContext backend=$selectedBackend gpuLayers=$selectedGpuLayers threads=$selectedCpuThreads")
        val startedAt = System.nanoTime()
        var firstTokenAt: Long? = null
        val output = StringBuilder()

        return try {
            val stream = callbackFlow {
                val worker = launch(Dispatchers.Default) {
                    try {
                        runtime.completion(prompt, params) { token ->
                            if (firstTokenAt == null) firstTokenAt = System.nanoTime()
                            trySend(token)
                        }
                        close()
                    } catch (t: Throwable) {
                        close(t)
                    }
                }
                awaitClose {
                    if (!worker.isCompleted) runtime.interrupt()
                    worker.cancel()
                }
            }
            stream.collect { chunk ->
                currentCoroutineContext().ensureActive()
                if (!chunk.isEmpty() && !stopRequested.get()) {
                    output.append(chunk)
                    onToken(chunk)
                }
            }

            val completed = GenerationResult(
                text = output.toString(),
                outputTokens = null,
                firstTokenTimeMs = firstTokenAt?.let { (it - startedAt) / 1_000_000 },
                generationTimeMs = (System.nanoTime() - startedAt) / 1_000_000,
                stopped = stopRequested.get(),
            )
            latestGeneration = completed
            RuntimeDiagnosticsStore.recordGeneration(settings, completed, runtimeInfo())
            ModelResult.Success(completed)
        } catch (t: CancellationException) {
            if (stopRequested.get()) {
                val stopped = GenerationResult(
                    text = output.toString(),
                    outputTokens = null,
                    firstTokenTimeMs = firstTokenAt?.let { (it - startedAt) / 1_000_000 },
                    generationTimeMs = (System.nanoTime() - startedAt) / 1_000_000,
                    stopped = true,
                )
                latestGeneration = stopped
                RuntimeDiagnosticsStore.recordGeneration(settings, stopped, runtimeInfo())
                ModelResult.Success(stopped)
            } else throw t
        } catch (t: Throwable) {
            if (stopRequested.get()) {
                val stopped = GenerationResult(
                    text = output.toString(),
                    outputTokens = null,
                    firstTokenTimeMs = firstTokenAt?.let { (it - startedAt) / 1_000_000 },
                    generationTimeMs = (System.nanoTime() - startedAt) / 1_000_000,
                    stopped = true,
                )
                latestGeneration = stopped
                RuntimeDiagnosticsStore.recordGeneration(settings, stopped, runtimeInfo())
                ModelResult.Success(stopped)
            } else {
                RuntimeDiagnosticsStore.recordTrace(RuntimeTraceEvent.Type.GENERATION_FAILED, t.message)
                ModelResult.Failure(RuntimeErrorMapper.inferenceFailure(t))
            }
        } finally {
            stopRequested.set(false)
        }
    }

    override suspend fun stopGeneration() {
        stopRequested.set(true)
        RuntimeDiagnosticsStore.recordTrace(RuntimeTraceEvent.Type.GENERATION_STOPPED)
        engine?.interrupt()
    }

    override fun runtimeInfo(): RuntimeInfo = RuntimeInfo(
        name = "llama.cpp-android-hybrid",
        version = "llama.kt@54ac0e85",
        backend = selectedBackend,
        threads = selectedCpuThreads,
        gpuLayers = selectedGpuLayers,
        contextLength = loadedContextLength ?: defaultContextLength,
    )

    private class RuntimeFailure(val error: ModelError) : IllegalStateException(error.message)
}
