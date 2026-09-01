package com.woogit.aicore.runtime.android

import com.woogit.aicore.domain.ChatMessage
import com.woogit.aicore.domain.GenerationRequest
import com.woogit.aicore.domain.GenerationResult
import com.woogit.aicore.domain.ModelDescriptor
import com.woogit.aicore.domain.ModelError
import com.woogit.aicore.domain.RuntimeInfo
import com.woogit.aicore.runtime.RuntimeAdapter
import dev.ffmpegkit.llama.Llama
import dev.ffmpegkit.llama.LlamaConfig
import dev.ffmpegkit.llama.LlamaModel
import java.io.File
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive

/**
 * Real Android RuntimeAdapter backed by the llama.cpp AAR.
 *
 * The AAR currently exposes full-completion rather than token Flow streaming in its
 * free artifact, so this adapter emits the completed response as one token callback.
 * This is intentionally not presented as incremental streaming. The RuntimeAdapter
 * boundary remains ready for a streaming backend later.
 */
class LlamaCppAndroidRuntimeAdapter(
    private val defaultContextLength: Int = 4096,
    private val defaultThreads: Int = 4,
    private val gpuLayers: Int = 0,
) : RuntimeAdapter {
    private var loadedModel: LlamaModel? = null
    private var loadedDescriptor: ModelDescriptor? = null

    override suspend fun load(model: ModelDescriptor) {
        currentCoroutineContext().ensureActive()
        unload()

        val file = model.path.toFile()
        if (!file.isFile || !file.canRead()) {
            throw ModelError.FileAccess("Model file cannot be read: ${file.absolutePath}")
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
            loadedDescriptor = model
        } catch (t: OutOfMemoryError) {
            throw t
        } catch (t: Throwable) {
            loadedModel = null
            loadedDescriptor = null
            throw ModelError.LoadFailed("Unable to load GGUF model", t)
        }
    }

    override suspend fun unload() {
        loadedModel?.let(Llama::releaseModel)
        loadedModel = null
        loadedDescriptor = null
    }

    override suspend fun generate(
        request: GenerationRequest,
        onToken: suspend (String) -> Unit,
    ): GenerationResult {
        currentCoroutineContext().ensureActive()
        val model = loadedModel ?: throw ModelError.RuntimeUnavailable("No local model is loaded")

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

        if (prompt.isBlank()) {
            throw ModelError.Inference("Generation request contains no user/tool content")
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

            request.settings.stopSequences
                .fold(result.text) { text, stop -> text.substringBefore(stop) }
                .also { text ->
                    if (text.isNotEmpty()) onToken(text)
                }

            return GenerationResult(
                text = request.settings.stopSequences.fold(result.text) { text, stop ->
                    text.substringBefore(stop)
                },
                outputTokens = result.tokensGenerated.toLong(),
                firstTokenTimeMs = result.promptEvalTimeMs,
                generationTimeMs = result.generateTimeMs.takeIf { it > 0 }
                    ?: ((System.nanoTime() - startedAt) / 1_000_000),
                stopped = false,
            )
        } catch (t: ModelError) {
            throw t
        } catch (t: OutOfMemoryError) {
            throw t
        } catch (t: Throwable) {
            throw ModelError.Inference("Local model inference failed", t)
        }
    }

    override suspend fun stopGeneration() {
        // The selected Maven Central free AAR does not expose a native interrupt API.
        // Coroutine cancellation is honored by the adapter around the native call, while
        // a true mid-call native stop will be enabled when a streaming/interrupt-capable
        // binding is selected.
        currentCoroutineContext().ensureActive()
    }

    override fun runtimeInfo(): RuntimeInfo = RuntimeInfo(
        name = "llama.cpp-android",
        version = "b9878",
        backend = "CPU/NEON",
    )
}
