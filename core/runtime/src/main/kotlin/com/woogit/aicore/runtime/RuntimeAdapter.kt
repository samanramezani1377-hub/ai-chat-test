package com.woogit.aicore.runtime

import com.woogit.aicore.domain.GenerationRequest
import com.woogit.aicore.domain.GenerationResult
import com.woogit.aicore.domain.ModelDescriptor
import com.woogit.aicore.domain.ModelRuntime
import com.woogit.aicore.domain.RuntimeInfo

/** Boundary between AI Core and a concrete model runtime. */
interface RuntimeAdapter : ModelRuntime

/**
 * Base adapter that deliberately fails until a real runtime is supplied.
 * It prevents accidental fake-success behavior in development.
 */
class UnconfiguredRuntimeAdapter : RuntimeAdapter {
    override suspend fun load(model: ModelDescriptor): Nothing =
        error("No model runtime adapter is configured")

    override suspend fun unload(): Nothing =
        error("No model runtime adapter is configured")

    override suspend fun generate(
        request: GenerationRequest,
        onToken: suspend (String) -> Unit
    ): Nothing = error("No model runtime adapter is configured")

    override suspend fun stopGeneration(): Nothing =
        error("No model runtime adapter is configured")

    override fun runtimeInfo(): RuntimeInfo =
        RuntimeInfo(name = "unconfigured", version = "0", backend = null)
}
