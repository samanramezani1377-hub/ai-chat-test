package com.woogit.aicore.actions

import com.woogit.aicore.domain.Action
import com.woogit.aicore.domain.RiskLevel

/** Actions whose data comes from an application/platform provider rather than the core module. */
class ProviderAction(
    override val id: String,
    override val risk: RiskLevel = RiskLevel.LOW,
    override val description: String = id,
    private val provider: suspend () -> Any,
) : Action<Any, Any> {
    override suspend fun execute(input: Any): Any = provider()
}

fun DefaultActionRegistry.registerProviderActions(
    modelInfo: (suspend () -> Any)? = null,
    performanceStats: (suspend () -> Any)? = null,
    deviceInfo: (suspend () -> Any)? = null,
) {
    modelInfo?.let { register("runtime", ProviderAction("get_model_info", description = "Return available local model and runtime information.", provider = it)) }
    performanceStats?.let { register("runtime", ProviderAction("get_performance_stats", description = "Return the latest local inference performance metrics when available.", provider = it)) }
    deviceInfo?.let { register("device", ProviderAction("get_device_info", description = "Return basic device manufacturer, model, and Android SDK information.", provider = it)) }
}
