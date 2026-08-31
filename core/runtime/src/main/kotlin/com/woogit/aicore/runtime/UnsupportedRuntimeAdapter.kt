package com.woogit.aicore.runtime

import com.woogit.aicore.domain.*

/** Explicit foundation adapter. A real runtime must implement ModelRuntime without leaking its API into Core. */
class UnsupportedRuntimeAdapter : ModelRuntime {
    override suspend fun load(model: ModelDescriptor) = error("No model runtime adapter configured")

    override suspend fun unload() = Unit

    override suspend fun generate(request: GenerationRequest, onToken: suspend (String) -> Unit): GenerationResult =
        error("No model runtime adapter configured")

    override suspend fun stopGeneration() = Unit

    override fun runtimeInfo(): RuntimeInfo = RuntimeInfo(
        name = "unconfigured",
        version = "0",
        backend = null
    )
}
