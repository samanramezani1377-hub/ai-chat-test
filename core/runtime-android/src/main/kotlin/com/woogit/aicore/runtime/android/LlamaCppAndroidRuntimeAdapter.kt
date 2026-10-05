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
    private val defaultContextLength: Int = 8192,
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
            // Use a larger working context so a long user prompt does not immediately
            // consume the entire generation window. We still never exceed the model's
            // trained context; prompt trimming below reserves output capacity.
            val modelContext = model.metadata.contextLength?.toInt()?.takeIf { it > 0 } ?: defaultContextLength
            val requested = minOf(modelContext, defaultContextLength)
            RuntimeDiagnosticsStore.recordNativeEvent("NATIVE_LOAD_STARTED file=${file.name} sizeBytes=${file.length()} context=$requested gpuPercent=$requestedGpuPercent gpuLayers=$requestedGpuLayers totalBlocks=${totalBlocks ?: "unknown"}")
            RuntimeDiagnosticsStore.recordTrace(RuntimeTraceEvent.Type.GENERATION_STARTED, "NATIVE_LOAD_STARTED file=${file.name} sizeBytes=${file.length()} requestedContext=$requested gpuPercent=$requestedGpuPercent gpuLayers=$requestedGpuLayers totalBlocks=${totalBlocks ?: "unknown"}")
            val draftCandidates = listOf(
    File(file.parentFile, "Qwen3-0.6B-Q4_K_M.gguf"),
    File(file.parentFile, "Qwen3-0.6B-Q5_K_M.gguf"),
    File(file.parentFile, "Qwen3-0.6B-Q6_K.gguf")
)
val draftFile = draftCandidates.firstOrNull { it.isFile && it.canRead() }
if (draftFile != null) {
    RuntimeDiagnosticsStore.recordNativeEvent("SPECULATIVE_DRAFT_SELECTED file=" + draftFile.name + " sizeBytes=" + draftFile.length())
} else {
    RuntimeDiagnosticsStore.recordNativeEvent("SPECULATIVE_DRAFT_NOT_FOUND target=" + file.name)
}
val result = NativeLlamaCpp.load(file.absolutePath, requested, requestedGpuLayers, draftFile?.absolutePath)
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

    suspend fun generateResult(request: GenerationRequest, onToken: suspend (String) -> Unit): ModelResult<GenerationResult> =
        nativeOperationMutex.withLock {
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
        val promptTokens = NativeLlamaCpp.countTokens(prompt).takeIf { it >= 0 }
        RuntimeDiagnosticsStore.recordNativeEvent("NATIVE_PROMPT_READY context=$loadedContextLength promptTokens=${promptTokens ?: "n/a"} maxNewTokens=${settings.maxNewTokens}")
        val startedAt = System.nanoTime()
        var firstTokenAt: Long? = null
        val output = StringBuilder()
        val pending = StringBuilder()
        val maxStopLength = settings.stopSequences.maxOfOrNull { it.length } ?: 0
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
            val completed = GenerationResult(text = output.toString(), inputTokens = promptTokens?.toLong(), outputTokens = NativeLlamaCpp.countTokens(output.toString()).takeIf { it >= 0 }?.toLong(), firstTokenTimeMs = firstTokenAt?.let { (it - startedAt) / 1_000_000 }, generationTimeMs = (System.nanoTime() - startedAt) / 1_000_000, stopped = stopRequested.get())
            latestGeneration = completed
            RuntimeDiagnosticsStore.recordNativeEvent("NATIVE_GENERATE_RETURNED tokens=${completed.outputTokens ?: "n/a"}")
            RuntimeDiagnosticsStore.recordGeneration(settings, completed, runtimeInfo())
            ModelResult.Success(completed)
        } catch (t: CancellationException) {
            if (stopRequested.get()) {
                val stopped = GenerationResult(text = output.toString(), inputTokens = promptTokens?.toLong(), outputTokens = NativeLlamaCpp.countTokens(output.toString()).takeIf { it >= 0 }?.toLong(), firstTokenTimeMs = firstTokenAt?.let { (it - startedAt) / 1_000_000 }, generationTimeMs = (System.nanoTime() - startedAt) / 1_000_000, stopped = true)
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
    private suspend fun trimMessagesToContext(
        messages: List<com.woogit.aicore.domain.ChatMessage>,
        maxNewTokens: Int,
    ): List<com.woogit.aicore.domain.ChatMessage> {
        val context = loadedContextLength ?: return messages
        val reserve = maxNewTokens + 32
        val budget = context - reserve
        if (messages.isEmpty() || budget <= 0) return messages.takeLast(1)

        val selected = ArrayDeque<com.woogit.aicore.domain.ChatMessage>()
        var truncatedLatest = false

        for (index in messages.lastIndex downTo 0) {
            val message = messages[index]
            val candidate = buildList {
                add(message)
                addAll(selected)
            }
            val tokenCount = NativeLlamaCpp.countTokens(formatMessages(candidate))
            if (tokenCount in 0..budget) {
                selected.addFirst(message)
                continue
            }

            // If the newest message itself is larger than the whole prompt budget,
            // never fall back to the unbounded message (the previous implementation
            // did exactly that and nativeGenerate correctly rejected it as overflow).
            if (selected.isEmpty() && index == messages.lastIndex) {
                val fitted = fitMessageToBudget(message, budget)
                selected.addFirst(fitted)
                truncatedLatest = fitted.content != message.content
            }
            break
        }

        val result = selected.toList()
        val finalTokenCount = if (result.isEmpty()) 0 else NativeLlamaCpp.countTokens(formatMessages(result))
        RuntimeDiagnosticsStore.recordNativeEvent(
            "NATIVE_PROMPT_BUDGET context=" + context +
                " reserve=" + reserve +
                " budget=" + budget +
                " selectedMessages=" + result.size +
                " originalMessages=" + messages.size +
                " promptTokens=" + finalTokenCount +
                " truncatedLatest=" + truncatedLatest
        )
        return result
    }

    private fun formatMessages(messages: List<com.woogit.aicore.domain.ChatMessage>): String =
        when (loadedArchitecture) {
            "lfm2" -> Lfm2PromptFormatter.format(messages)
            else -> Qwen3PromptFormatter.format(messages)
        }

    /**
     * Fit one oversized message to the token budget without character-count guesses.
     * The search keeps both the beginning and end of the message so instructions at
     * the start and the actual payload/result at the end are retained.
     */
    private fun fitMessageToBudget(
        message: com.woogit.aicore.domain.ChatMessage,
        budget: Int,
    ): com.woogit.aicore.domain.ChatMessage {
        if (budget <= 0) return message.copy(content = "")

        fun candidate(length: Int): com.woogit.aicore.domain.ChatMessage {
            if (length >= message.content.length) return message
            if (length <= 0) return message.copy(content = "")
            val marker = "\n[… محتوای میانی برای جا شدن در پنجرهٔ متن حذف شد …]\n"
            val payloadLength = (length - marker.length).coerceAtLeast(0)
            val head = (payloadLength * 0.6f).toInt().coerceAtMost(message.content.length)
            val tail = payloadLength - head
            val content = if (tail <= 0) {
                message.content.take(head) + marker
            } else {
                message.content.take(head) + marker + message.content.takeLast(tail)
            }
            return message.copy(content = content)
        }

        var low = 0
        var high = message.content.length
        var best = message.copy(content = "")
        while (low <= high) {
            val mid = low + (high - low) / 2
            val candidateMessage = candidate(mid)
            val tokens = NativeLlamaCpp.countTokens(formatMessages(listOf(candidateMessage)))
            if (tokens in 0..budget) {
                best = candidateMessage
                low = mid + 1
            } else {
                high = mid - 1
            }
        }
        return best
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
