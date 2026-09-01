package com.woogit.aicore.runtime.android

import com.woogit.aicore.domain.GenerationRequest
import com.woogit.aicore.domain.GenerationResult
import com.woogit.aicore.domain.ModelDescriptor
import com.woogit.aicore.domain.ModelError
import com.woogit.aicore.domain.ModelResult
import com.woogit.aicore.domain.RuntimeInfo
import com.woogit.aicore.runtime.RuntimeAdapter
import dev.ffmpegkit.llama.Llama
import dev.ffmpegkit.llama.LlamaConfig
import dev.ffmpegkit.llama.LlamaModel
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive

/** Real Android RuntimeAdapter backed by the llama.cpp AAR. */
class LlamaCppAndroidRuntimeAdapter(
    private val defaultContextLength: Int = 4096,
    private val defaultThreads: Int = 4,
    private val gpuLayers: Int = 0,
) : RuntimeAdapter {
    private var loadedModel: LlamaModel? = null

    override suspend fun load(model: ModelDescriptor) {
        loadResult(model).let { result ->
            if (result is ModelResult.Failure) {
                throw RuntimeFailure(result.error)
            }
        }
    }

    suspend fun loadResult(model: ModelDescriptor): ModelResult<Unit> {
        currentCoroutineContext().ensureActive()
        unload()

        val file = model.path.toFile()
        if (!file.isFile || !file.canRead()) {
            return ModelResult.Failure(
                ModelError.FileAccess("Model file cannot be read: ${file.absolutePath}")
            )
        }

        return try {
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
            ModelResult.Success(Unit)
        } catch (t: Throwable) {
            loadedModel = null
            ModelResult.Failure(RuntimeErrorMapper.loadFailure(t, file.absolutePath))
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
        return when (val result = generateResult(request, onToken)) {
            is ModelResult.Success -> result.value
            is ModelResult.Failure -> throw RuntimeFailure(result.error)
        }
    }

    suspend fun generateResult(
        request: GenerationRequest,
        onToken: suspend (String) -> Unit,
    ): ModelResult<GenerationResult> {
        currentCoroutineContext().ensureActive()
        val model = loadedModel
            ?: return ModelResult.Failure(ModelError.RuntimeUnavailable("No local model is loaded"))

        val prompt = try {
            Qwen3PromptFormatter.format(request.messages)
        } catch (t: IllegalArgumentException) {
            return ModelResult.Failure(ModelError.Inference(t.message ?: "Invalid conversation"))
        }

        return try {
            val startedAt = System.nanoTime()
            // The selected free AAR exposes completion rather than token streaming.
            // Keep the domain callback contract, but emit only the completed response.
            // Qwen3's ChatML prompt is formatted explicitly so multi-turn history and
            // role boundaries are preserved instead of using generic User/Assistant text.
            val result = Llama.complete(
                model = model,
                prompt = prompt,
                systemPrompt = "",
                maxTokens = request.settings.maxNewTokens.coerceAtLeast(1),
            )
            currentCoroutineContext().ensureActive()

            val text = request.settings.stopSequences.fold(result.text) { value, stop ->
                value.substringBefore(stop)
            }
            if (text.isNotEmpty()) onToken(text)

            ModelResult.Success(
                GenerationResult(
                    text = text,
                    outputTokens = result.tokensGenerated.toLong(),
                    firstTokenTimeMs = result.promptEvalTimeMs,
                    generationTimeMs = result.generateTimeMs.takeIf { it > 0 }
                        ?: ((System.nanoTime() - startedAt) / 1_000_000),
                    stopped = false,
                )
            )
        } catch (t: Throwable) {
            ModelResult.Failure(RuntimeErrorMapper.inferenceFailure(t))
        }
    }

    override suspend fun stopGeneration() {
        // The free AAR does not expose a native interrupt API. Coroutine cancellation
        // remains the only honest stop mechanism until a streaming/interrupt-capable
        // runtime is selected.
        currentCoroutineContext().ensureActive()
    }

    override fun runtimeInfo(): RuntimeInfo = RuntimeInfo(
        name = "llama.cpp-android",
        version = "b9878",
        backend = "CPU/NEON",
    )

    private class RuntimeFailure(val error: ModelError) : IllegalStateException(error.message)
}
