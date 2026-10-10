package com.samanramezani.aichattest

import com.woogit.aicore.domain.GenerationRequest
import com.woogit.aicore.domain.GenerationResult
import com.woogit.aicore.domain.ModelDescriptor
import com.woogit.aicore.domain.ModelFormat
import com.woogit.aicore.domain.RuntimeInfo
import com.woogit.aicore.runtime.RuntimeAdapter
import com.woogit.aicore.runtime.RuntimeMetrics
import com.woogit.aicore.runtime.android.LlamaCppAndroidRuntimeAdapter
import com.woogit.aicore.runtime.mnn.MnnAndroidRuntimeAdapter

/** Routes GGUF to llama.cpp and exported MNN packs to MNN without mixing model formats. */
class CompositeModelRuntimeAdapter(
    private val llama: LlamaCppAndroidRuntimeAdapter,
    private val mnn: MnnAndroidRuntimeAdapter,
) : RuntimeAdapter, RuntimeMetrics {
    @Volatile private var activeFormat: ModelFormat? = null
    private fun active(): RuntimeAdapter = if (activeFormat == ModelFormat.MNN) mnn else llama
    override suspend fun load(model: ModelDescriptor) {
        when (model.format) {
            ModelFormat.GGUF -> { mnn.unload(); llama.load(model); activeFormat = ModelFormat.GGUF }
            ModelFormat.MNN -> { llama.unload(); mnn.load(model); activeFormat = ModelFormat.MNN }
            ModelFormat.UNKNOWN -> error("Unsupported model format: UNKNOWN")
        }
    }
    override suspend fun unload() {
        var failure: Throwable? = null
        try { llama.unload() } catch (t: Throwable) { failure = t }
        try { mnn.unload() } catch (t: Throwable) { if (failure == null) failure = t }
        activeFormat = null
        failure?.let { throw it }
    }
    override suspend fun generate(request: GenerationRequest, onToken: suspend (String) -> Unit): GenerationResult = active().generate(request, onToken)
    override suspend fun stopGeneration() = active().stopGeneration()
    override fun runtimeInfo(): RuntimeInfo = active().runtimeInfo()
    override fun lastGeneration(): GenerationResult? = (active() as? RuntimeMetrics)?.lastGeneration()
    override fun lastLoadTimeMs(): Long? = (active() as? RuntimeMetrics)?.lastLoadTimeMs()
}
