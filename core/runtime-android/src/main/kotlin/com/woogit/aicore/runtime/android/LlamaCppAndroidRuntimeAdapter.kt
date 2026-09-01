package com.woogit.aicore.runtime.android

import com.woogit.aicore.domain.ChatMessage
import com.woogit.aicore.domain.GenerationRequest
import com.woogit.aicore.domain.GenerationResult
import com.woogit.aicore.domain.ModelDescriptor
import com.woogit.aicore.domain.RuntimeInfo
import com.woogit.aicore.runtime.RuntimeAdapter
import dev.ffmpegkit.llama.Llama
import dev.ffmpegkit.llama.LlamaConfig
import dev.ffmpegkit.llama.LlamaModel
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive

/**
 * Real Android RuntimeAdapter backed by the llama.cpp AAR.
 *
 * The selected free AAR exposes full completion rather than token Flow streaming, so
 * this adapter emits the completed response through the existing callback as one chunk.
 * It is intentionally not presented as incremental streaming.
 */
class LlamaCppAndroidRuntimeAdapter(
    private val defaultContextLength: Int = 4096,
    private val defaultThreads: Int = 4,
    private val gpuLayers: Int = 0,
) : RuntimeAdapter {
    private var loadedModel: LlamaModel? = null

    override suspend fun load(model: ModelDescriptor) {
        currentCoroutineContext().ensureActive()
        unload()

        val file = model.path.toFile()
        require(file.isFile && file.canRead()) {
            "Model file cannot be read: ${file.absolutePath}"
        }

        try {
            loadedModel = Llama.loadModel(
                modelPath = file.absolutePath,
                config = LlamaConfig(
                    contextSize = model.metadata.contextLength?.toInt()
                        ?.takeIf { it > 0 }
                        ?: defaultContextLength,
                    threads = defaultThreads.coerceAtLeast(1),
                    gpuLayers = gpuLayers.coerceAtLeast(0),
                ),
            )
        } catch (t: OutOfMemoryError) {
            throw t
        } catch (t: Throwable) {
            loadedModel = null
            throw IllegalStateException("Unable to load GGUF model", t)
        }
    }

    override suspend fun unload() {
        loadedModel?.let(Llama::releaseModel)
        loadedModel = null
    }

    override suspend fun generate(
        request: GenerationRequest,
        onToken: suspend (String) -> Unit,
    ): GenerationResult {
        currentCoroutineContext().ensureActive()
        val model = loadedModel
            ?: throw IllegalStateException("No local model is loaded")

        val systemPrompt = request.messages
            .firstOrNull { it.role == ChatMessage.Role.SYSTEM }
            ?.content
            .orEmpty()

        val prompt = request.messages
            .filter { it.role != ChatMessage.Role.SYSTEM }
            .joinToString("\n") { message ->
                when (message.role) {
                    ChatMessage.Role.USER -> "User: ${message.content}"
                    ChatMessage.Role.ASSISTANT -> "Assistant: ${message.content}"
                    ChatMessage.Role.TOOL -> "Tool: ${message.content}"
                    ChatMessage.Role.SYSTEM -> message.content
                }
            }

        require(prompt.isNotBlank()) {
            "Generation request contains no user/tool content"
        }

        try {
            val startedAt = System.nanoTime()
            val result = Llama.complete(
                model = model,
                prompt = prompt,
                systemPrompt = systemPrompt,
                maxTokens = request.settings.maxNewTokens.coerceAtLeast(1),
            )
            currentCoroutineContext().ensureActive()

            val text = request.settings.stopSequences.fold(result.text) { value, stop ->
                value.substringBefore(stop)
            }
            if (text.isNotEmpty()) onToken(text)

            return GenerationResult(
                text = text,
                outputTokens = result.tokensGenerated.toLong(),
                firstTokenTimeMs = result.promptEvalTimeMs,
                generationTimeMs = result.generateTimeMs.takeIf { it > 0 }
                    ?: ((System.nanoTime() - startedAt) / 1_000_000),
                stopped = false,
            )
        } catch (t: OutOfMemoryError) {
            throw t
        } catch (t: Throwable) {
            throw IllegalStateException("Local model inference failed", t)
        }
    }

    override suspend fun stopGeneration() {
        // The selected free AAR does not expose a native mid-generation interrupt API.
        // We still honor coroutine cancellation at the adapter boundary. A true native
        // stop will require a streaming/interrupt-capable binding in a later runtime.
        currentCoroutineContext().ensureActive()
    }

    override fun runtimeInfo(): RuntimeInfo = RuntimeInfo(
        name = "llama.cpp-android",
        version = "b9878",
        backend = "CPU/NEON",
    )
}
