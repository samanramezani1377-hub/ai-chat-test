package com.woogit.aicore.runtime.android

import com.woogit.aicore.domain.ChatMessage
import com.woogit.aicore.domain.GenerationRequest
import com.woogit.aicore.domain.GenerationResult
import com.woogit.aicore.domain.ModelDescriptor
import com.woogit.aicore.domain.ModelError
import com.woogit.aicore.domain.ModelResult
import com.woogit.aicore.domain.RuntimeInfo
import com.woogit.aicore.runtime.RuntimeAdapter
import com.woogit.aicore.runtime.RuntimeMetrics
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import org.codeshipping.llamakotlin.LlamaConfig
import org.codeshipping.llamakotlin.LlamaModel
import java.util.concurrent.atomic.AtomicBoolean

/** Real Android RuntimeAdapter backed by a native llama.cpp streaming runtime. */
class LlamaCppAndroidRuntimeAdapter(
    private val defaultContextLength: Int = 4096,
    private val defaultThreads: Int = 4,
    private val gpuLayers: Int = 0,
) : RuntimeAdapter, RuntimeMetrics {
    private var loadedModel: LlamaModel? = null
    private val stopRequested = AtomicBoolean(false)
    @Volatile private var latestGeneration: GenerationResult? = null
    @Volatile private var loadedContextLength: Int? = null

    override fun lastGeneration(): GenerationResult? = latestGeneration

    override suspend fun load(model: ModelDescriptor) {
        loadResult(model).let { result ->
            if (result is ModelResult.Failure) throw RuntimeFailure(result.error)
        }
    }

    suspend fun loadResult(model: ModelDescriptor): ModelResult<Unit> {
        currentCoroutineContext().ensureActive()
        unload()
        val file = model.path.toFile()
        if (!file.isFile || !file.canRead()) return ModelResult.Failure(ModelError.FileAccess("Model file cannot be read: ${file.absolutePath}"))
        return try {
            val contextLength = model.metadata.contextLength?.toInt()?.takeIf { it > 0 } ?: defaultContextLength
            loadedModel = LlamaModel.load(
                modelPath = file.absolutePath,
                config = LlamaConfig(
                    contextSize = contextLength,
                    threads = defaultThreads.coerceAtLeast(1),
                    gpuLayers = gpuLayers.coerceAtLeast(0),
                ),
            )
            loadedContextLength = contextLength
            latestGeneration = null
            ModelResult.Success(Unit)
        } catch (t: Throwable) {
            loadedModel = null
            loadedContextLength = null
            ModelResult.Failure(RuntimeErrorMapper.loadFailure(t, file.absolutePath))
        }
    }

    override suspend fun unload() {
        stopRequested.set(true)
        loadedModel?.close()
        loadedModel = null
        loadedContextLength = null
    }

    override suspend fun generate(request: GenerationRequest, onToken: suspend (String) -> Unit): GenerationResult =
        when (val result = generateResult(request, onToken)) {
            is ModelResult.Success -> result.value
            is ModelResult.Failure -> throw RuntimeFailure(result.error)
        }

    suspend fun generateResult(request: GenerationRequest, onToken: suspend (String) -> Unit): ModelResult<GenerationResult> {
        currentCoroutineContext().ensureActive()
        val model = loadedModel ?: return ModelResult.Failure(ModelError.RuntimeUnavailable("No local model is loaded"))
        val prompt = try { buildPrompt(request.messages) } catch (t: Throwable) {
            return ModelResult.Failure(ModelError.Inference(t.message ?: "Invalid conversation"))
        }
        val settings = request.settings
        val requestedContext = settings.contextLength?.takeIf { it > 0 }
        val effectiveContext = when {
            requestedContext == null -> loadedContextLength ?: defaultContextLength
            loadedContextLength != null -> requestedContext.coerceAtMost(loadedContextLength!!)
            else -> requestedContext
        }
        val config = LlamaConfig(
            contextSize = effectiveContext,
            threads = defaultThreads.coerceAtLeast(1),
            temperature = settings.temperature.toFloat().coerceAtLeast(0f),
            topP = (settings.topP ?: 0.9).toFloat().coerceIn(0f, 1f),
            topK = (settings.topK ?: 40).coerceAtLeast(0),
            repeatPenalty = (settings.repeatPenalty ?: 1.1).toFloat().coerceAtLeast(0f),
            maxTokens = settings.maxNewTokens.coerceAtLeast(1),
            stopSequences = settings.stopSequences,
            seed = settings.seed?.coerceIn(Int.MIN_VALUE.toLong(), Int.MAX_VALUE.toLong())?.toInt() ?: -1,
            gpuLayers = gpuLayers.coerceAtLeast(0),
        )
        stopRequested.set(false)
        val startedAt = System.nanoTime()
        var firstTokenAt: Long? = null
        val output = StringBuilder()
        var emittedLength = 0
        val maxStopLength = settings.stopSequences.maxOfOrNull { it.length } ?: 0

        return try {
            model.generateStream(prompt, config).collect { chunk ->
                currentCoroutineContext().ensureActive()
                if (chunk.isEmpty() || stopRequested.get()) return@collect
                if (firstTokenAt == null) firstTokenAt = System.nanoTime()
                output.append(chunk)
                val stopIndex = firstStopIndex(output, settings.stopSequences)
                if (stopIndex >= 0) {
                    emitRange(output, emittedLength, stopIndex, onToken)
                    emittedLength = stopIndex
                    stopRequested.set(true)
                    model.cancelGeneration()
                    return@collect
                }
                val holdBack = (maxStopLength - 1).coerceAtLeast(0)
                val safeEnd = (output.length - holdBack).coerceAtLeast(emittedLength)
                if (safeEnd > emittedLength) {
                    emitRange(output, emittedLength, safeEnd, onToken)
                    emittedLength = safeEnd
                }
            }
            if (!stopRequested.get() && emittedLength < output.length) {
                emitRange(output, emittedLength, output.length, onToken)
            }
            val completed = GenerationResult(
                text = trimAtStop(output.toString(), settings.stopSequences),
                outputTokens = null,
                firstTokenTimeMs = firstTokenAt?.let { (it - startedAt) / 1_000_000 },
                generationTimeMs = (System.nanoTime() - startedAt) / 1_000_000,
                stopped = stopRequested.get(),
            )
            latestGeneration = completed
            ModelResult.Success(completed)
        } catch (t: CancellationException) {
            if (stopRequested.get()) {
                val stopped = stoppedResult(output, firstTokenAt, startedAt, settings.stopSequences)
                latestGeneration = stopped.value
                stopped
            } else throw t
        } catch (t: Throwable) {
            if (stopRequested.get()) {
                val stopped = stoppedResult(output, firstTokenAt, startedAt, settings.stopSequences)
                latestGeneration = stopped.value
                stopped
            } else ModelResult.Failure(RuntimeErrorMapper.inferenceFailure(t))
        } finally {
            stopRequested.set(false)
        }
    }

    override suspend fun stopGeneration() {
        stopRequested.set(true)
        loadedModel?.cancelGeneration()
    }

    override fun runtimeInfo(): RuntimeInfo = RuntimeInfo(
        name = "llama.cpp-android",
        version = runCatching { LlamaModel.getVersion() }.getOrDefault("unknown"),
        backend = if (gpuLayers > 0) "GPU" else "CPU/NEON",
    )

    private fun buildPrompt(messages: List<ChatMessage>): String = Qwen3PromptFormatter.format(messages)

    private suspend fun emitRange(text: StringBuilder, start: Int, end: Int, onToken: suspend (String) -> Unit) {
        if (end > start) onToken(text.substring(start, end))
    }

    private fun firstStopIndex(text: StringBuilder, stops: List<String>): Int {
        var first = -1
        for (stop in stops) {
            if (stop.isEmpty()) continue
            val index = text.indexOf(stop)
            if (index >= 0 && (first < 0 || index < first)) first = index
        }
        return first
    }

    private fun trimAtStop(text: String, stops: List<String>): String {
        var first = -1
        for (stop in stops) {
            if (stop.isEmpty()) continue
            val index = text.indexOf(stop)
            if (index >= 0 && (first < 0 || index < first)) first = index
        }
        return if (first >= 0) text.substring(0, first) else text
    }

    private fun stoppedResult(output: StringBuilder, firstTokenAt: Long?, startedAt: Long, stops: List<String>): ModelResult.Success<GenerationResult> =
        ModelResult.Success(GenerationResult(
            text = trimAtStop(output.toString(), stops),
            outputTokens = null,
            firstTokenTimeMs = firstTokenAt?.let { (it - startedAt) / 1_000_000 },
            generationTimeMs = (System.nanoTime() - startedAt) / 1_000_000,
            stopped = true,
        ))

    private class RuntimeFailure(val error: ModelError) : IllegalStateException(error.message)
}
