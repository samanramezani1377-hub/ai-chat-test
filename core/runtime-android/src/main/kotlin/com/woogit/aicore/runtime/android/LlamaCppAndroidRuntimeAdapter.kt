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
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.roundToInt

/** Direct llama.cpp Android runtime. OpenCL is the only supported inference backend. */
class LlamaCppAndroidRuntimeAdapter(
    private val defaultContextLength: Int = 4096,
    gpuLayers: Int = GPU_LAYERS_MAX,
) : RuntimeAdapter, RuntimeMetrics {
    companion object {
        const val GPU_LAYERS_MAX = 99
    }

    private val stopRequested = AtomicBoolean(false)
    /** Serializes model load/unload and generation. */
    private val nativeOperationMutex = Mutex()
    @Volatile private var gpuLayersMode = gpuLayers
    @Volatile private var selectedGpuLayers = 0
    @Volatile private var selectedCpuThreads = 2
    @Volatile private var selectedBackend = "OpenCL"
    @Volatile private var loadedContextLength: Int? = null
    @Volatile private var loadedArchitecture: String = "unknown"
    @Volatile private var loadedModelPath: String? = null
    @Volatile private var latestGeneration: GenerationResult? = null
    @Volatile private var latestLoadTimeMs: Long? = null

    init { require(gpuLayers == GPU_LAYERS_MAX) { "Unsupported diagnostic GPU layer mode: $gpuLayers" } }

    fun setGpuLayers(value: Int) {
        require(value == GPU_LAYERS_MAX) { "Unsupported diagnostic GPU layer mode: $value" }
        gpuLayersMode = value
        RuntimeDiagnosticsStore.recordNativeEvent("OPENCL_GPU_ONLY_SELECTED")
        RuntimeDiagnosticsStore.recordTrace(RuntimeTraceEvent.Type.GENERATION_STARTED, "OPENCL_GPU_ONLY_SELECTED (applies on next activation)")
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

    suspend fun loadResult(model: ModelDescriptor): ModelResult<Unit> = nativeOperationMutex.withLock {
        currentCoroutineContext().ensureActive()
        val requestedPath = model.path.toFile().absolutePath
        if (loadedContextLength != null && loadedModelPath == requestedPath) {
            RuntimeDiagnosticsStore.recordNativeEvent("NATIVE_LOAD_SKIPPED_ALREADY_LOADED file=${model.path.toFile().name} context=$loadedContextLength gpuLayers=$selectedGpuLayers")
            return ModelResult.Success(Unit)
        }
        stopRequested.set(true)
        NativeLlamaCpp.unload()
        stopRequested.set(false)
        selectedGpuLayers = 0
        selectedBackend = "OpenCL"
        loadedContextLength = null
        loadedArchitecture = "unknown"
        loadedModelPath = null
        val file = model.path.toFile()
        if (!file.isFile || !file.canRead()) return ModelResult.Failure(ModelError.FileAccess("Model file cannot be read: ${file.absolutePath}"))
        val startedAt = System.nanoTime()
        val requestedGpuPercent = 100
        val totalBlocks = model.metadata.blockCount?.toInt()?.takeIf { it > 0 }
        val requestedGpuLayers = when {
            totalBlocks != null -> totalBlocks
            else -> 99
        }
        return try {
            // Do not blindly allocate the model's advertised maximum context on mobile.
            // Qwen3-1.7B uses about 112 KiB of FP16 KV cache per token, so 32K
            // context alone is ~3.7 GiB before model weights and graph buffers.
            // Keep the Android runtime at a conservative 4K working context.
            val modelContext = model.metadata.contextLength?.toInt()?.takeIf { it > 0 } ?: defaultContextLength
            val requested = minOf(modelContext, defaultContextLength)
            RuntimeDiagnosticsStore.recordNativeEvent("NATIVE_LOAD_STARTED file=${file.name} sizeBytes=${file.length()} context=$requested gpuPercent=$requestedGpuPercent gpuLayers=$requestedGpuLayers totalBlocks=${totalBlocks ?: "unknown"}")
            RuntimeDiagnosticsStore.recordTrace(RuntimeTraceEvent.Type.GENERATION_STARTED, "NATIVE_LOAD_STARTED file=${file.name} sizeBytes=${file.length()} requestedContext=$requested gpuPercent=$requestedGpuPercent gpuLayers=$requestedGpuLayers totalBlocks=${totalBlocks ?: "unknown"}")
            val result = NativeLlamaCpp.load(file.absolutePath, requested, requestedGpuLayers)
            RuntimeDiagnosticsStore.recordNativeEvent("NATIVE_LOAD_RETURNED code=$result gpuPercent=$requestedGpuPercent gpuLayers=$requestedGpuLayers")
            if (result != 0) return ModelResult.Failure(ModelError.Inference("llama.cpp failed to load the model (code=$result)"))
            val info = NativeLlamaCpp.runtimeInfo()
            selectedBackend = info.substringBefore(';').ifBlank { "OpenCL" }
            selectedGpuLayers = if (selectedBackend.contains("OpenCL", ignoreCase = true)) requestedGpuLayers else 0
            selectedCpuThreads = 2
            loadedContextLength = NativeLlamaCpp.contextLength().takeIf { it > 0 } ?: requested
            loadedArchitecture = model.metadata.architecture?.lowercase() ?: "unknown"
            loadedModelPath = file.absolutePath
            latestGeneration = null
            latestLoadTimeMs = (System.nanoTime() - startedAt) / 1_000_000
            RuntimeDiagnosticsStore.recordNativeEvent("NATIVE_RUNTIME_READY backend=$selectedBackend gpuPercent=$requestedGpuPercent gpuLayers=$selectedGpuLayers context=$loadedContextLength")
            RuntimeDiagnosticsStore.recordLoaded(model, latestLoadTimeMs, runtimeInfo())
            ModelResult.Success(Unit)
        } catch (t: Throwable) {
            RuntimeDiagnosticsStore.recordNativeEvent("NATIVE_LOAD_EXCEPTION ${t::class.java.name}: ${t.message}")
            NativeLlamaCpp.unload()
            selectedGpuLayers = 0
            selectedBackend = "OpenCL"
            loadedContextLength = null
            loadedArchitecture = "unknown"
            loadedModelPath = null
            latestLoadTimeMs = (System.nanoTime() - startedAt) / 1_000_000
            RuntimeDiagnosticsStore.recordTrace(RuntimeTraceEvent.Type.GENERATION_FAILED, "MODEL_LOAD_DIAGNOSTIC_FAILED ${t.message}")
            ModelResult.Failure(RuntimeErrorMapper.loadFailure(t, file.absolutePath))
        }
    }

    override suspend fun unload() = nativeOperationMutex.withLock {
        unloadUnsafe()
    }

    private fun unloadUnsafe() {
        stopRequested.set(true)
        NativeLlamaCpp.unload()
        stopRequested.set(false)
        selectedGpuLayers = 0
        selectedBackend = "OpenCL"
        loadedContextLength = null
        loadedArchitecture = "unknown"
        loadedModelPath = null
    }

    override suspend fun generate(request: GenerationRequest, onToken: suspend (String) -> Unit): GenerationResult = when (val result = generateResult(request, onToken)) {
        is ModelResult.Success -> result.value
        is ModelResult.Failure -> throw RuntimeFailure(result.error)
    }

    suspend fun generateResult(request: GenerationRequest, onToken: suspend (String) -> Unit): ModelResult<GenerationResult> {
        currentCoroutineContext().ensureActive()
        if (loadedContextLength == null) return ModelResult.Failure(ModelError.RuntimeUnavailable("No local model is loaded"))
        val settings = request.settings
        val safeMessages = trimMessagesToContext(request.messages, settings.maxNewTokens.coerceAtLeast(1))
        val prompt = try {
            when (loadedArchitecture) {
                "lfm2" -> Lfm2PromptFormatter.format(safeMessages)
                else -> Qwen3PromptFormatter.format(safeMessages)
            }
        } catch (t: Throwable) {
            return ModelResult.Failure(ModelError.Inference(t.message ?: "Invalid conversation"))
        }
        stopRequested.set(false)
        RuntimeDiagnosticsStore.recordNativeEvent("NATIVE_GENERATE_STARTED context=$loadedContextLength backend=$selectedBackend gpuLayers=$selectedGpuLayers threads=$selectedCpuThreads")
        RuntimeDiagnosticsStore.recordTrace(RuntimeTraceEvent.Type.GENERATION_STARTED, "context=${loadedContextLength} backend=$selectedBackend gpuLayers=$selectedGpuLayers threads=$selectedCpuThreads")
        val startedAt = System.nanoTime()
        var firstTokenAt: Long? = null
        val output = StringBuilder()
        val pending = StringBuilder()
        val maxStopLength = settings.stopSequences.maxOfOrNull { it.length } ?: 0
        return nativeOperationMutex.withLock {
            try {
                NativeLlamaCpp.generate(
                prompt = prompt,
                maxTokens = settings.maxNewTokens.coerceAtLeast(1),
                temperature = if (loadedArchitecture == "lfm2") 0.3f else settings.temperature.toFloat().coerceAtLeast(0f),
                topK = (settings.topK ?: 40).coerceAtLeast(0),
                topP = (settings.topP ?: 0.9).toFloat().coerceIn(0f, 1f),
                minP = if (loadedArchitecture == "lfm2") 0.15f else (settings.minP ?: 0.05).toFloat().coerceIn(0f, 1f)
            ).collect { chunk ->
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
    /**
     * Keep as much recent conversation as possible while reserving the full generation
     * budget. The actual GGUF vocabulary counts tokens, so mixed Persian/English text
     * is handled without a character-based guess.
     */
    private fun trimMessagesToContext(
        messages: List<com.woogit.aicore.domain.ChatMessage>,
        maxNewTokens: Int,
    ): List<com.woogit.aicore.domain.ChatMessage> {
        val context = loadedContextLength ?: return messages
        val reserve = maxNewTokens + 32
        val budget = context - reserve
        if (messages.isEmpty() || budget <= 0) return messages.takeLast(1)

        val selected = ArrayDeque<com.woogit.aicore.domain.ChatMessage>()
        for (index in messages.lastIndex downTo 0) {
            selected.addFirst(messages[index])
            val candidate = selected.toList()
            val formatted = when (loadedArchitecture) {
                "lfm2" -> Lfm2PromptFormatter.format(candidate)
                else -> Qwen3PromptFormatter.format(candidate)
            }
            val tokenCount = NativeLlamaCpp.countTokens(formatted)
            if (tokenCount < 0 || tokenCount > budget) {
                selected.removeFirst()
                break
            }
        }
        val result = selected.toList()
        RuntimeDiagnosticsStore.recordNativeEvent(
            "NATIVE_PROMPT_BUDGET context=" + context +
                " reserve=" + reserve +
                " selectedMessages=" + result.size +
                " originalMessages=" + messages.size
        )
        return if (result.isNotEmpty()) result else messages.takeLast(1)
    }

    override suspend fun stopGeneration() {
        stopRequested.set(true)
        RuntimeDiagnosticsStore.recordNativeEvent("NATIVE_STOP_REQUESTED")
        RuntimeDiagnosticsStore.recordTrace(RuntimeTraceEvent.Type.GENERATION_STOPPED)
        NativeLlamaCpp.stop()
    }

    override fun runtimeInfo(): RuntimeInfo = RuntimeInfo(name = "llama.cpp-android-direct", version = "2ca15f5", backend = selectedBackend, threads = selectedCpuThreads, gpuLayers = selectedGpuLayers, contextLength = loadedContextLength ?: defaultContextLength)

    private class RuntimeFailure(val error: ModelError) : IllegalStateException(error.message)
}
