package com.woogit.aicore.actions

import com.woogit.aicore.domain.ActionRegistry
import com.woogit.aicore.domain.CapabilityProvider
import com.woogit.aicore.domain.Verifier
import java.util.UUID

sealed interface ActionExecutionState {
    data object Prepared : ActionExecutionState
    data object AwaitingApproval : ActionExecutionState
    data object Approved : ActionExecutionState
    data object Executing : ActionExecutionState
    data class Completed(val verification: com.woogit.aicore.domain.VerificationResult) : ActionExecutionState
    data class Failed(val message: String) : ActionExecutionState
    data object Rejected : ActionExecutionState
}

data class PreparedAction(
    val executionId: String = UUID.randomUUID().toString(),
    val actionId: String,
    val input: Any,
    val risk: com.woogit.aicore.domain.RiskLevel,
    val state: ActionExecutionState = ActionExecutionState.Prepared
)

interface ActionApprovalPolicy { fun requiresApproval(risk: com.woogit.aicore.domain.RiskLevel): Boolean }

class DefaultActionApprovalPolicy : ActionApprovalPolicy {
    override fun requiresApproval(risk: com.woogit.aicore.domain.RiskLevel): Boolean = risk == com.woogit.aicore.domain.RiskLevel.SENSITIVE
}

interface ActionCheckpointStore {
    suspend fun save(action: PreparedAction)
    suspend fun get(executionId: String): PreparedAction?
}

class InMemoryActionCheckpointStore : ActionCheckpointStore {
    private val values = mutableMapOf<String, PreparedAction>()
    @Synchronized override suspend fun save(action: PreparedAction) { values[action.executionId] = action }
    @Synchronized override suspend fun get(executionId: String): PreparedAction? = values[executionId]
}

class ActionLifecycle(
    private val registry: ActionRegistry,
    private val capabilityProvider: CapabilityProvider,
    private val checkpointStore: ActionCheckpointStore,
    private val approvalPolicy: ActionApprovalPolicy = DefaultActionApprovalPolicy()
) {
    suspend fun prepare(actionId: String, input: Any): PreparedAction {
        val action = registry.find(actionId) ?: error("Action not found: $actionId")
        return PreparedAction(actionId = action.id, input = input, risk = action.risk).also(checkpointStore::save)
    }

    fun requiresApproval(prepared: PreparedAction): Boolean = approvalPolicy.requiresApproval(prepared.risk)

    suspend fun validate(prepared: PreparedAction, capability: String?): PreparedAction {
        if (capability != null && !capabilityProvider.supports(capability)) error("Required capability is unavailable: $capability")
        val state = if (requiresApproval(prepared)) ActionExecutionState.AwaitingApproval else ActionExecutionState.Approved
        return prepared.copy(state = state).also(checkpointStore::save)
    }

    suspend fun approve(executionId: String): PreparedAction {
        val prepared = checkpointStore.get(executionId) ?: error("Prepared action not found: $executionId")
        require(prepared.state == ActionExecutionState.AwaitingApproval) { "Action is not awaiting final approval" }
        return prepared.copy(state = ActionExecutionState.Approved).also(checkpointStore::save)
    }

    suspend fun reject(prepared: PreparedAction): PreparedAction = prepared.copy(state = ActionExecutionState.Rejected).also(checkpointStore::save)

    suspend fun executeApproved(executionId: String, verifier: Verifier<Any>): ActionExecutionState {
        val prepared = checkpointStore.get(executionId) ?: return ActionExecutionState.Failed("Prepared action not found: $executionId")
        if (prepared.state != ActionExecutionState.Approved) return ActionExecutionState.Failed("Action is not approved for execution")
        val action = registry.find(prepared.actionId) ?: return ActionExecutionState.Failed("Action not found: ${prepared.actionId}")
        return try {
            checkpointStore.save(prepared.copy(state = ActionExecutionState.Executing))
            val result = action.execute(prepared.input)
            val verification = verifier.verify(result)
            if (!verification.success) {
                val failed = ActionExecutionState.Failed(verification.evidence ?: "Action verification failed")
                checkpointStore.save(prepared.copy(state = failed))
                return failed
            }
            val completed = ActionExecutionState.Completed(verification)
            checkpointStore.save(prepared.copy(state = completed))
            completed
        } catch (t: Throwable) {
            val failed = ActionExecutionState.Failed(t.message ?: t::class.simpleName.orEmpty())
            checkpointStore.save(prepared.copy(state = failed))
            failed
        }
    }
}
