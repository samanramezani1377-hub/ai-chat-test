package com.woogit.aicore.runtime.android

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
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.collect
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.roundToInt

/** Direct llama.cpp Android runtime. OpenCL GPU is preferred; CPU is the automatic fallback. */
class LlamaCppAndroidRuntimeAdapter(
    private val defaultContextLength: Int = 4096,
    gpuLayers: Int = GPU_LAYERS_CPU_ONLY,
) : RuntimeAdapter, RuntimeMetrics {
    companion object {
        const val GPU_LAYERS_CPU_ONLY = 0
        const val GPU_LAYERS_70 = 70
        const val GPU_LAYERS_MAX = 99
    }

    private val stopRequested = AtomicBoolean(false)
    @Volatile private var gpuLayersMode = gpuLayers
    @Volatile private var selectedGpuLayers = 0
    @Volatile private var selectedCpuThreads = 2
    @Volatile private var selectedBackend = "CPU/NEON"
    @Volatile private var loadedContextLength: Int? = null
    @Volatile private var latestGeneration: GenerationResult? = null
    @Volatile private var latestLoadTimeMs: Long? = null

    init { require(gpuLayers in setOf(GPU_LAYERS_CPU_ONLY, GPU_LAYERS_70, GPU_LAYERS_MAX)) { "Unsupported diagnostic GPU layer mode: $gpuLayers" } }

    fun setGpuLayers(value: Int) {
        require(value in setOf(GPU_LAYERS_CPU_ONLY, GPU_LAYERS_70, GPU_LAYERS_MAX)) { "Unsupported diagnostic GPU layer mode: $value" }
        gpuLayersMode = value
        RuntimeDiagnosticsStore.recordNativeEvent("GPU_LAYER_MODE_SELECTED percent=$value")
        RuntimeDiagnosticsStore.recordTrace(RuntimeTraceEvent.Type.GENERATION_STARTED, "GPU_LAYER_MODE_SELECTED percent=$value (applies on next activation)")
    }

    fun gpuLayers(): Int = gpuLayersMode
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
        if (!file.isFile || !file.canRead()) return ModelResult.Failure(ModelError.FileAccess("Model file cannot be read: ${file.absolutePath}"))
        val startedAt = System.nanoTime()
        val requestedGpuPercent = gpuLayersMode
        val totalBlocks = model.metadata.blockCount?.toInt()?.takeIf { it > 0 }
        val requestedGpuLayers = when {
            requestedGpuPercent <= 0 -> 0
            requestedGpuPercent >= 99 -> 99
            totalBlocks != null -> ((totalBlocks * requestedGpuPercent) / 100.0).roundToInt().coerceIn(1, totalBlocks)
            else -> 99
        }
        return try {
            val requested = model.metadata.contextLength?.toInt()?.takeIf { it > 0 } ?: defaultContextLength
            RuntimeDiagnosticsStore.recordNativeEvent("NATIVE_LOAD_STARTED file=${file.name} sizeBytes=${file.length()} context=$requested gpuPercent=$requestedGpuPercent gpuLayers=$actualGpuLayers totalBlocks=${totalBlocks ?: "unknown"}")
            RuntimeDiagnosticsStore.recordTrace(RuntimeTraceEvent.Type.GENERATION_STARTED, "NATIVE_LOAD_STARTED file=${file.name} sizeBytes=${file.length()} requestedContext=$requested gpuPercent=$requestedGpuPercent gpuLayers=$requestedGpuLayers totalBlocks=${totalBlocks ?: "unknown"}")
            val result = NativeLlamaCpp.load(file.absolutePath, requested, requestedGpuLayers)
            RuntimeDiagnosticsStore.recordNativeEvent("NATIVE_LOAD_RETURNED code=$result gpuPercent=$requestedGpuPercent gpuLayers=$actualGpuLayers")
            if (result != 0) return ModelResult.Failure(ModelError.Inference("llama.cpp failed to load the model (code=$result)"))
            val info = NativeLlamaCpp.runtimeInfo()
            selectedBackend = info.substringBefore(';').ifBlank { "CPU/NEON" }
            selectedGpuLayers = if (selectedBackend.contains("OpenCL", ignoreCase = true)) actualGpuLayers else 0
            selectedCpuThreads = 2
            loadedContextLength = NativeLlamaCpp.contextLength().takeIf { it > 0 } ?: requested
            latestGeneration = null
            latestLoadTimeMs = (System.nanoTime() - startedAt) / 1_000_000
            RuntimeDiagnosticsStore.recordNativeEvent("NATIVE_RUNTIME_READY backend=$selectedBackend gpuPercent=$requestedGpuPercent gpuLayers=$selectedGpuLayers context=$loadedContextLength")
            RuntimeDiagnosticsStore.recordLoaded(model, latestLoadTimeMs, runtimeInfo())
            ModelResult.Success(Unit)
        } catch (t: Throwable) {
            RuntimeDiagnosticsStore.recordNativeEvent("NATIVE_LOAD_EXCEPTION ${t::class.java.name}: ${t.message}")
            NativeLlamaCpp.unload()
            selectedGpuLayers = 0
            selectedBackend = "CPU/NEON"
            loadedContextLength = null
            latestLoadTimeMs = (System.nanoTime() - startedAt) / 1_000_000
            RuntimeDiagnosticsStore.recordTrace(RuntimeTraceEvent.Type.GENERATION_FAILED, "MODEL_LOAD_DIAGNOSTIC_FAILED ${t.message}")
            ModelResult.Failure(RuntimeErrorMapper.loadFailure(t, file.absolutePath))
        }
    }

    override suspend fun unload() {
        stopRequested.set(true)
        NativeLlamaCpp.unload()
        stopRequested.set(false)
        selectedGpuLayers = 0
        selectedBackend = "CPU/NEON"
        loadedContextLength = null
    }

    override suspend fun generate(request: GenerationRequest, onToken: suspend (String) -> Unit): GenerationResult = when (val result = generateResult(request, onToken)) {
        is ModelResult.Success -> result.value
        is ModelResult.Failure -> throw RuntimeFailure(result.error)
    }

    suspend fun generateResult(request: GenerationRequest, onToken: suspend (String) -> Unit): ModelResult<GenerationResult> {
        currentCoroutineContext().ensureActive()
        if (loadedContextLength == null) return ModelResult.Failure(ModelError.RuntimeUnavailable("No local model is loaded"))
        val settings = request.settings
        val prompt = try { Qwen3PromptFormatter.format(request.messages) } catch (t: Throwable) { return ModelResult.Failure(ModelError.Inference(t.message ?: "Invalid conversation")) }
        stopRequested.set(false)
        RuntimeDiagnosticsStore.recordNativeEvent("NATIVE_GENERATE_STARTED context=$loadedContextLength backend=$selectedBackend gpuLayers=$selectedGpuLayers threads=$selectedCpuThreads")
        RuntimeDiagnosticsStore.recordTrace(RuntimeTraceEvent.Type.GENERATION_STARTED, "context=${loadedContextLength} backend=$selectedBackend gpuLayers=$selectedGpuLayers threads=$selectedCpuThreads")
        val startedAt = System.nanoTime()
        var firstTokenAt: Long? = null
        val output = StringBuilder()
        val pending = StringBuilder()
        val maxStopLength = settings.stopSequences.maxOfOrNull { it.length } ?: 0
        return try {
            NativeLlamaCpp.generate(prompt = prompt, maxTokens = settings.maxNewTokens.coerceAtLeast(1), temperature = settings.temperature.toFloat().coerceAtLeast(0f), topK = (settings.topK ?: 40).coerceAtLeast(0), topP = (settings.topP ?: 0.9).toFloat().coerceIn(0f, 1f), minP = (settings.minP ?: 0.05).toFloat().coerceIn(0f, 1f)).collect { chunk ->
                currentCoroutineContext().ensureActive()
                if (chunk.isEmpty() || stopRequested.get()) return@collect
                if (firstTokenAt == null) firstTokenAt = System.nanoTime()
                pending.append(chunk)
                val stop = settings.stopSequences.firstOrNull { pending.indexOf(it) >= 0 }
                if (stop != null) {
                    val index = pending.indexOf(stop)
                    val visible = pending.substring(0, index)
                    if (visible.isNotEmpty()) { onToken(visible); output.append(visible) }
                    pending.setLength(0)
                    stopRequested.set(true)
                    NativeLlamaCpp.stop()
                } else {
                    val safeCount = if (maxStopLength > 0) pending.length - maxStopLength + 1 else pending.length
                    if (safeCount > 0) { val safe = pending.substring(0, safeCount); pending.delete(0, safeCount); onToken(safe); output.append(safe) }
                }
            }
            if (!stopRequested.get() && pending.isNotEmpty()) { onToken(pending.toString()); output.append(pending); pending.setLength(0) }
            val completed = GenerationResult(text = output.toString(), firstTokenTimeMs = firstTokenAt?.let { (it - startedAt) / 1_000_000 }, generationTimeMs = (System.nanoTime() - startedAt) / 1_000_000, stopped = stopRequested.get())
            latestGeneration = completed
            RuntimeDiagnosticsStore.recordNativeEvent("NATIVE_GENERATE_RETURNED tokens=${completed.outputTokens ?: "n/a"}")
            RuntimeDiagnosticsStore.recordGeneration(settings, completed, runtimeInfo())
            ModelResult.Success(completed)
        } catch (t: CancellationException) {
            if (stopRequested.get()) {
                val stopped = GenerationResult(text = output.toString(), firstTokenTimeMs = firstTokenAt?.let { (it - startedAt) / 1_000_000 }, generationTimeMs = (System.nanoTime() - startedAt) / 1_000_000, stopped = true)
                latestGeneration = stopped
                RuntimeDiagnosticsStore.recordGeneration(settings, stopped, runtimeInfo())
                ModelResult.Success(stopped)
            } else throw t
        } catch (t: Throwable) {
            RuntimeDiagnosticsStore.recordNativeEvent("NATIVE_GENERATE_EXCEPTION ${t::class.java.name}: ${t.message}")
            if (stopRequested.get()) {
                val stopped = GenerationResult(text = output.toString(), firstTokenTimeMs = firstTokenAt?.let { (it - startedAt) / 1_000_000 }, generationTimeMs = (System.nanoTime() - startedAt) / 1_000_000, stopped = true)
                latestGeneration = stopped
                RuntimeDiagnosticsStore.recordGeneration(settings, stopped, runtimeInfo())
                ModelResult.Success(stopped)
            } else {
                RuntimeDiagnosticsStore.recordTrace(RuntimeTraceEvent.Type.GENERATION_FAILED, t.message)
                ModelResult.Failure(RuntimeErrorMapper.inferenceFailure(t))
            }
        } finally { stopRequested.set(false) }
    }

    override suspend fun stopGeneration() {
        stopRequested.set(true)
        RuntimeDiagnosticsStore.recordNativeEvent("NATIVE_STOP_REQUESTED")
        RuntimeDiagnosticsStore.recordTrace(RuntimeTraceEvent.Type.GENERATION_STOPPED)
        NativeLlamaCpp.stop()
    }

    override fun runtimeInfo(): RuntimeInfo = RuntimeInfo(name = "llama.cpp-android-direct", version = "c5fc7e34885ba31217e330809437afa993d27745", backend = selectedBackend, threads = selectedCpuThreads, gpuLayers = selectedGpuLayers, contextLength = loadedContextLength ?: defaultContextLength)

    private class RuntimeFailure(val error: ModelError) : IllegalStateException(error.message)
}
