package com.woogit.aicore.actions

import com.woogit.aicore.domain.Action
import com.woogit.aicore.domain.CapabilityProvider
import com.woogit.aicore.domain.VerificationResult
import com.woogit.aicore.domain.Verifier

sealed interface ActionExecutionState {
    data object Prepared : ActionExecutionState
    data object AwaitingApproval : ActionExecutionState
    data object Executing : ActionExecutionState
    data class Completed(val verification: VerificationResult) : ActionExecutionState
    data class Failed(val message: String) : ActionExecutionState
    data object Rejected : ActionExecutionState
}

data class PreparedAction(
    val actionId: String,
    val input: Any,
    val risk: com.woogit.aicore.domain.RiskLevel,
    val state: ActionExecutionState = ActionExecutionState.Prepared
)

interface ActionApprovalPolicy {
    fun requiresApproval(risk: com.woogit.aicore.domain.RiskLevel): Boolean
}

class DefaultActionApprovalPolicy : ActionApprovalPolicy {
    override fun requiresApproval(risk: com.woogit.aicore.domain.RiskLevel): Boolean =
        risk == com.woogit.aicore.domain.RiskLevel.SENSITIVE
}

class ActionLifecycle(
    private val registry: ActionRegistryFacade,
    private val capabilityProvider: CapabilityProvider,
    private val approvalPolicy: ActionApprovalPolicy = DefaultActionApprovalPolicy()
) {
    suspend fun prepare(actionId: String, input: Any): PreparedAction {
        val action = registry.find(actionId)
            ?: error("Action not found: $actionId")
        return PreparedAction(action.id, input, action.risk)
    }

    fun requiresApproval(prepared: PreparedAction): Boolean =
        approvalPolicy.requiresApproval(prepared.risk)

    fun validate(prepared: PreparedAction, capability: String?): PreparedAction {
        if (capability != null && !capabilityProvider.supports(capability)) {
            error("Required capability is unavailable: $capability")
        }
        return prepared.copy(state = ActionExecutionState.AwaitingApproval)
    }

    suspend fun execute(
        prepared: PreparedAction,
        verifier: Verifier<Any>
    ): ActionExecutionState {
        val action = registry.find(prepared.actionId)
            ?: return ActionExecutionState.Failed("Action not found: ${prepared.actionId}")
        return try {
            val result = action.execute(prepared.input)
            val verification = verifier.verify(result)
            ActionExecutionState.Completed(verification)
        } catch (t: Throwable) {
            ActionExecutionState.Failed(t.message ?: t::class.simpleName.orEmpty())
        }
    }
}

interface ActionRegistryFacade {
    fun find(actionId: String): Action<Any, Any>?
}
