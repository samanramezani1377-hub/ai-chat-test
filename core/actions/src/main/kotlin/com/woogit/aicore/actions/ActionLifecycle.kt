package com.woogit.aicore.actions

import com.woogit.aicore.domain.Action
import com.woogit.aicore.domain.ActionRegistry
import com.woogit.aicore.domain.CapabilityProvider
import com.woogit.aicore.domain.VerificationResult
import com.woogit.aicore.domain.Verifier
import java.util.UUID

sealed interface ActionExecutionState {
    data object Prepared : ActionExecutionState
    data object AwaitingApproval : ActionExecutionState
    data object Executing : ActionExecutionState
    data class Completed(val verification: VerificationResult) : ActionExecutionState
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

interface ActionApprovalPolicy {
    fun requiresApproval(risk: com.woogit.aicore.domain.RiskLevel): Boolean
}

class DefaultActionApprovalPolicy : ActionApprovalPolicy {
    override fun requiresApproval(risk: com.woogit.aicore.domain.RiskLevel): Boolean =
        risk == com.woogit.aicore.domain.RiskLevel.SENSITIVE
}

interface ActionCheckpointStore {
    suspend fun save(action: PreparedAction)
    suspend fun get(executionId: String): PreparedAction?
}

class InMemoryActionCheckpointStore : ActionCheckpointStore {
    private val values = mutableMapOf<String, PreparedAction>()

    @Synchronized
    override suspend fun save(action: PreparedAction) { values[action.executionId] = action }

    @Synchronized
    override suspend fun get(executionId: String): PreparedAction? = values[executionId]
}

class ActionLifecycle(
    private val registry: ActionRegistry,
    private val capabilityProvider: CapabilityProvider,
    private val checkpointStore: ActionCheckpointStore,
    private val approvalPolicy: ActionApprovalPolicy = DefaultActionApprovalPolicy()
) {
    suspend fun prepare(actionId: String, input: Any): PreparedAction {
        val action = registry.find(actionId) ?: error("Action not found: $actionId")
        return PreparedAction(actionId = action.id, input = input, risk = action.risk).also {
            checkpointStore.save(it)
        }
    }

    fun requiresApproval(prepared: PreparedAction): Boolean =
        approvalPolicy.requiresApproval(prepared.risk)

    suspend fun validate(prepared: PreparedAction, capability: String?): PreparedAction {
        if (capability != null && !capabilityProvider.supports(capability)) {
            error("Required capability is unavailable: $capability")
        }
        val validated = prepared.copy(state = ActionExecutionState.AwaitingApproval)
        checkpointStore.save(validated)
        return validated
    }

    suspend fun reject(prepared: PreparedAction): PreparedAction {
        val rejected = prepared.copy(state = ActionExecutionState.Rejected)
        checkpointStore.save(rejected)
        return rejected
    }

    suspend fun executeApproved(
        executionId: String,
        verifier: Verifier<Any>
    ): ActionExecutionState {
        val prepared = checkpointStore.get(executionId)
            ?: return ActionExecutionState.Failed("Prepared action not found: $executionId")
        if (prepared.state != ActionExecutionState.AwaitingApproval) {
            return ActionExecutionState.Failed("Action is not awaiting final approval")
        }

        val action = registry.find(prepared.actionId)
            ?: return ActionExecutionState.Failed("Action not found: ${prepared.actionId}")
        return try {
            val executing = prepared.copy(state = ActionExecutionState.Executing)
            checkpointStore.save(executing)
            val result = action.execute(prepared.input)
            val verification = verifier.verify(result)
            val completed = ActionExecutionState.Completed(verification)
            checkpointStore.save(executing.copy(state = completed))
            completed
        } catch (t: Throwable) {
            val failed = ActionExecutionState.Failed(t.message ?: t::class.simpleName.orEmpty())
            checkpointStore.save(prepared.copy(state = failed))
            failed
        }
    }
}
