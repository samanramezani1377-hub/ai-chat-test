package com.woogit.aicore.actions

import com.woogit.aicore.domain.Action
import com.woogit.aicore.domain.RiskLevel

/** Actions whose data comes from an application/platform provider rather than the core module. */
class ProviderAction(
    override val id: String,
    override val risk: RiskLevel = RiskLevel.LOW,
    private val provider: suspend () -> Any,
) : Action<Any, Any> {
    override suspend fun execute(input: Any): Any = provider()
}

fun DefaultActionRegistry.registerProviderActions(
    modelInfo: (suspend () -> Any)? = null,
    performanceStats: (suspend () -> Any)? = null,
    deviceInfo: (suspend () -> Any)? = null,
) {
    modelInfo?.let { register("runtime", ProviderAction("get_model_info", provider = it)) }
    performanceStats?.let { register("runtime", ProviderAction("get_performance_stats", provider = it)) }
    deviceInfo?.let { register("device", ProviderAction("get_device_info", provider = it)) }
}
