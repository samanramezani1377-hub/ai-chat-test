package com.woogit.aicore.runtime.android

import ai.onnxruntime.genai.Generator
import ai.onnxruntime.genai.GeneratorParams
import ai.onnxruntime.genai.Model
import ai.onnxruntime.genai.Tokenizer
import ai.onnxruntime.genai.TokenizerStream
import com.woogit.aicore.domain.ChatMessage
import com.woogit.aicore.domain.GenerationRequest
import com.woogit.aicore.domain.GenerationResult
import com.woogit.aicore.domain.ModelDescriptor
import com.woogit.aicore.domain.ModelError
import com.woogit.aicore.domain.ModelFormat
import com.woogit.aicore.domain.ModelResult
import com.woogit.aicore.domain.RuntimeInfo
import com.woogit.aicore.runtime.RuntimeAdapter
import com.woogit.aicore.runtime.RuntimeDiagnosticsStore
import com.woogit.aicore.runtime.RuntimeMetrics
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import java.nio.file.Files
import java.util.concurrent.atomic.AtomicBoolean

/** Primary local inference engine. It loads ONNX Runtime GenAI model packages, never GGUF files. */
class OnnxRuntimeGenAiAndroidRuntimeAdapter : RuntimeAdapter, RuntimeMetrics {
    companion object {
        private const val GENAI_VERSION = "0.11.4"
        private const val TAG = "ONNX_RUNTIME_GENAI"
    }

    private val lock = Any()
    @Volatile private var loadedModel: Model? = null
    @Volatile private var tokenizer: Tokenizer? = null
    @Volatile private var loadedDescriptor: ModelDescriptor? = null
    @Volatile private var loadedModelType: String = "qwen3"
    @Volatile private var loadedContextLength: Int = 4096
    private val stopRequested = AtomicBoolean(false)
    @Volatile private var latestGeneration: GenerationResult? = null
    @Volatile private var latestLoadTimeMs: Long? = null
    @Volatile private var generating = false

    override suspend fun load(model: ModelDescriptor) = withContext(Dispatchers.IO) {
        require(model.format == ModelFormat.ONNX_GENAI) {
            "مدل انتخاب‌شده از نوع ONNX Runtime GenAI نیست؛ فایل GGUF را نمی‌توان در این موتور بارگذاری کرد."
        }
        require(model.validation == com.woogit.aicore.domain.ValidationStatus.VALID) {
            "بستهٔ مدل معتبر نیست؛ دوباره آن را وارد کن."
        }
        require(Files.isDirectory(model.path) && Files.isRegularFile(model.path.resolve("genai_config.json"))) {
            "پیکربندی genai_config.json در پوشهٔ مدل پیدا نشد."
        }

        val started = System.nanoTime()
        var candidate: Model? = null
        var candidateTokenizer: Tokenizer? = null
        try {
            synchronized(lock) {
                closeQuietly(tokenizer)
                closeQuietly(loadedModel)
                tokenizer = null
                loadedModel = null
                loadedDescriptor = null
            }
            RuntimeDiagnosticsStore.recordNativeEvent("$TAG LOAD_BEGIN model=${model.displayName} path=${model.path}")
            candidate = Model(model.path.toAbsolutePath().toString())
            candidateTokenizer = candidate.createTokenizer()
            val config = Files.readString(model.path.resolve("genai_config.json"))
            val type = Regex("""["']type["']\s*:\s*["']([^"']+)["']""")
                .find(config)?.groupValues?.get(1)?.lowercase() ?: model.metadata.architecture.orEmpty().lowercase()
            synchronized(lock) {
                loadedModel = candidate
                tokenizer = candidateTokenizer
                loadedDescriptor = model
                loadedModelType = type.ifBlank { "qwen3" }
                loadedContextLength = (model.metadata.contextLength ?: 4096L).coerceIn(256L, 131072L).toInt()
                latestLoadTimeMs = (System.nanoTime() - started) / 1_000_000L
            }
            RuntimeDiagnosticsStore.recordNativeEvent("$TAG LOAD_READY model=${model.displayName} type=$loadedModelType context=$loadedContextLength loadMs=$latestLoadTimeMs")
            RuntimeDiagnosticsStore.recordLoaded(model, latestLoadTimeMs, runtimeInfo())
        } catch (t: Throwable) {
            closeQuietly(candidateTokenizer)
            closeQuietly(candidate)
            synchronized(lock) {
                tokenizer = null
                loadedModel = null
                loadedDescriptor = null
            }
            RuntimeDiagnosticsStore.recordNativeEvent("$TAG LOAD_FAILED type=${t::class.java.name} message=${t.message}")
            throw t
        }
    }

    /** Compatibility hook for the existing model manager; ONNX GenAI has no llama.cpp draft model. */
    fun setDraftPath(path: String?) = Unit

    /** Compatibility hook: report success only when this exact target is already loaded. */
    suspend fun loadResult(model: ModelDescriptor): ModelResult<Unit> = try {
        if (loadedDescriptor?.id != model.id) load(model)
        ModelResult.Success(Unit)
    } catch (t: Throwable) {
        ModelResult.Failure(ModelError.LoadFailed(t.message ?: "Unable to load ONNX Runtime GenAI model", t))
    }

    override suspend fun unload() = withContext(Dispatchers.IO) {
        stopRequested.set(true)
        synchronized(lock) {
            closeQuietly(tokenizer)
            closeQuietly(loadedModel)
            tokenizer = null
            loadedModel = null
            loadedDescriptor = null
            generating = false
            latestGeneration = null
        }
        RuntimeDiagnosticsStore.recordNativeEvent("$TAG UNLOAD_COMPLETED")
    }

    override suspend fun generate(
        request: GenerationRequest,
        onToken: suspend (String) -> Unit,
    ): GenerationResult = withContext(Dispatchers.Default) {
        val activeModel = loadedModel ?: error("هیچ مدل ONNX Runtime GenAI فعال نیست؛ ابتدا مدل را فعال کن.")
        val activeTokenizer = tokenizer ?: error("Tokenizer مدل بارگذاری نشده است.")
        require(request.messages.isNotEmpty()) { "messages must not be empty" }
        stopRequested.set(false)
        generating = true

        var encoded: Any? = null
        var params: GeneratorParams? = null
        var generator: Generator? = null
        var stream: TokenizerStream? = null
        val output = StringBuilder()
        val pending = StringBuilder()
        var outputTokens = 0L
        val startedAt = System.nanoTime()
        var firstTokenAt: Long? = null
        val prompt = formatPrompt(request.messages, loadedModelType, request.settings.enableThinking)
        var promptTokenCount = 0
        try {
            currentCoroutineContext().ensureActive()
            val sequences = activeTokenizer.encode(prompt)
            encoded = sequences
            promptTokenCount = sequences.getSequence(0).size
            val contextBudget = (loadedContextLength - promptTokenCount - 8).coerceAtLeast(1)
            val maxNewTokens = request.settings.maxNewTokens.coerceIn(1, contextBudget)
            params = GeneratorParams(activeModel)
            params.setInput(sequences)
            params.setSearchOption("max_length", (promptTokenCount + maxNewTokens).toDouble())
            params.setSearchOption("do_sample", request.settings.temperature > 0.0001)
            if (request.settings.temperature > 0.0001) {
                params.setSearchOption("temperature", request.settings.temperature.coerceIn(0.01, 2.0))
            }
            request.settings.topK?.takeIf { it > 0 }?.let { params.setSearchOption("top_k", it.toDouble()) }
            request.settings.topP?.let { params.setSearchOption("top_p", it.coerceIn(0.01, 1.0)) }
            request.settings.repeatPenalty?.let { params.setSearchOption("repetition_penalty", it.coerceIn(0.5, 2.0)) }
            generator = Generator(activeModel, params)
            stream = activeTokenizer.createStream()
            RuntimeDiagnosticsStore.recordTrace(com.woogit.aicore.runtime.RuntimeTraceEvent.Type.GENERATION_STARTED, "engine=onnxruntime-genai promptTokens=$promptTokenCount maxNewTokens=$maxNewTokens")
            RuntimeDiagnosticsStore.recordNativeEvent("$TAG GENERATE_BEGIN promptTokens=$promptTokenCount maxNewTokens=$maxNewTokens type=$loadedModelType")

            val maxStopLength = request.settings.stopSequences.maxOfOrNull { it.length } ?: 0
            while (!generator.isDone() && outputTokens < maxNewTokens && !stopRequested.get()) {
                currentCoroutineContext().ensureActive()
                generator.generateNextToken()
                outputTokens++
                val chunk = stream.decode(generator.getLastTokenInSequence(0))
                if (chunk.isNotEmpty()) {
                    if (firstTokenAt == null) firstTokenAt = System.nanoTime()
                    pending.append(chunk)
                    val stop = request.settings.stopSequences.firstOrNull { pending.indexOf(it) >= 0 }
                    if (stop != null) {
                        val visible = pending.substring(0, pending.indexOf(stop))
                        if (visible.isNotEmpty()) {
                            onToken(visible)
                            output.append(visible)
                        }
                        pending.setLength(0)
                        stopRequested.set(true)
                    } else {
                        val safeCount = if (maxStopLength > 0) pending.length - maxStopLength + 1 else pending.length
                        if (safeCount > 0) {
                            val visible = pending.substring(0, safeCount)
                            pending.delete(0, safeCount)
                            onToken(visible)
                            output.append(visible)
                        }
                    }
                }
            }
            if (!stopRequested.get() && pending.isNotEmpty()) {
                onToken(pending.toString())
                output.append(pending)
            }
            val result = GenerationResult(
                text = output.toString(),
                inputTokens = promptTokenCount.toLong(),
                outputTokens = outputTokens,
                firstTokenTimeMs = firstTokenAt?.let { (it - startedAt) / 1_000_000L },
                generationTimeMs = (System.nanoTime() - startedAt) / 1_000_000L,
                stopped = stopRequested.get(),
            )
            latestGeneration = result
            RuntimeDiagnosticsStore.recordNativeEvent("$TAG GENERATE_END outputTokens=$outputTokens stopped=${result.stopped}")
            RuntimeDiagnosticsStore.recordGeneration(request.settings, result, runtimeInfo())
            result
        } catch (t: CancellationException) {
            throw t
        } catch (t: Throwable) {
            RuntimeDiagnosticsStore.recordNativeEvent("$TAG GENERATE_FAILED type=${t::class.java.name} message=${t.message}")
            RuntimeDiagnosticsStore.recordTrace(com.woogit.aicore.runtime.RuntimeTraceEvent.Type.GENERATION_FAILED, t.message)
            throw t
        } finally {
            generating = false
            closeQuietly(stream)
            closeQuietly(generator)
            closeQuietly(params)
            closeQuietly(encoded)
        }
    }

    override suspend fun stopGeneration() {
        stopRequested.set(true)
        RuntimeDiagnosticsStore.recordNativeEvent("$TAG STOP_REQUESTED generating=$generating")
    }

    override fun runtimeInfo(): RuntimeInfo = RuntimeInfo(
        name = "onnxruntime-genai-android",
        version = GENAI_VERSION,
        backend = "Configured by genai_config.json (default provider when unspecified)",
        threads = null,
        gpuLayers = null,
        contextLength = loadedContextLength,
    )

    override fun lastGeneration(): GenerationResult? = latestGeneration
    override fun lastLoadTimeMs(): Long? = latestLoadTimeMs

    private fun formatPrompt(
        messages: List<ChatMessage>,
        modelType: String,
        enableThinking: Boolean,
    ): String {
        val type = modelType.lowercase()
        return when {
            type.startsWith("phi3") || type == "phi" -> buildString {
                messages.forEach { message ->
                    val role = when (message.role) {
                        ChatMessage.Role.SYSTEM -> "system"
                        ChatMessage.Role.USER -> "user"
                        ChatMessage.Role.ASSISTANT -> "assistant"
                        ChatMessage.Role.TOOL -> "user"
                    }
                    append("<|$role|>\n")
                    append(message.content)
                    append("\n")
                }
                append("<|assistant|>\n")
            }
            type.startsWith("llama") || type.startsWith("mistral") -> buildString {
                messages.forEach { message ->
                    val role = when (message.role) {
                        ChatMessage.Role.SYSTEM -> "system"
                        ChatMessage.Role.USER -> "user"
                        ChatMessage.Role.ASSISTANT -> "assistant"
                        ChatMessage.Role.TOOL -> "user"
                    }
                    append("<|start_header_id|>$role<|end_header_id|>\n\n")
                    append(message.content)
                    append("<|eot_id|>\n")
                }
                append("<|start_header_id|>assistant<|end_header_id|>\n\n")
            }
            else -> buildString {
                messages.forEach { message ->
                    val role = when (message.role) {
                        ChatMessage.Role.SYSTEM -> "system"
                        ChatMessage.Role.USER -> "user"
                        ChatMessage.Role.ASSISTANT -> "assistant"
                        ChatMessage.Role.TOOL -> "user"
                    }
                    append("<|im_start|>$role\n")
                    append(message.content)
                    append("<|im_end|>\n")
                }
                append("<|im_start|>assistant\n")
                if (!enableThinking && (type.startsWith("qwen2") || type.startsWith("qwen3"))) {
                    append("<think>\n</think>\n")
                }
            }
        }
    }

    private fun closeQuietly(value: Any?) {
        if (value == null) return
        runCatching { value.javaClass.getMethod("close").invoke(value) }
    }
}
