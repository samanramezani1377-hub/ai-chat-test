package com.woogit.aicore.actions

import com.woogit.aicore.domain.ActionRegistry
import com.woogit.aicore.domain.CapabilityProvider
import com.woogit.aicore.domain.VerificationResult
import com.woogit.aicore.domain.Verifier
import java.util.UUID

sealed interface ActionExecutionState {
    data object Prepared : ActionExecutionState
    data object AwaitingApproval : ActionExecutionState
    data object Approved : ActionExecutionState
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
    val state: ActionExecutionState = ActionExecutionState.Prepared,
    val retryCount: Int = 0
)

interface ActionApprovalPolicy { fun requiresApproval(risk: com.woogit.aicore.domain.RiskLevel): Boolean }
class DefaultActionApprovalPolicy : ActionApprovalPolicy {
    override fun requiresApproval(risk: com.woogit.aicore.domain.RiskLevel) = risk == com.woogit.aicore.domain.RiskLevel.SENSITIVE
}

interface ActionRetryPolicy { fun maxRetries(action: PreparedAction): Int }
class DefaultActionRetryPolicy(private val defaultMaxRetries: Int = 2) : ActionRetryPolicy {
    init { require(defaultMaxRetries >= 0) { "defaultMaxRetries must be non-negative" } }
    override fun maxRetries(action: PreparedAction): Int = defaultMaxRetries
}

interface ActionCheckpointStore {
    suspend fun save(action: PreparedAction)
    suspend fun get(executionId: String): PreparedAction?
    suspend fun list(): List<PreparedAction> = emptyList()
}
class InMemoryActionCheckpointStore : ActionCheckpointStore {
    private val values = mutableMapOf<String, PreparedAction>()
    override suspend fun save(action: PreparedAction) { synchronized(values) { values[action.executionId] = action } }
    override suspend fun get(executionId: String): PreparedAction? = synchronized(values) { values[executionId] }
    override suspend fun list(): List<PreparedAction> = synchronized(values) { values.values.toList() }
}

/** Independent verification hook. Implementations must verify the external side effect, not just the return value. */
fun interface ActionResultVerifier {
    suspend fun verify(action: PreparedAction, result: Any): VerificationResult
}

class ActionLifecycle(
    private val registry: ActionRegistry,
    private val capabilityProvider: CapabilityProvider,
    private val checkpointStore: ActionCheckpointStore,
    private val approvalPolicy: ActionApprovalPolicy = DefaultActionApprovalPolicy(),
    private val retryPolicy: ActionRetryPolicy = DefaultActionRetryPolicy(),
    private val traceSink: ActionTraceSink? = null,
    private val errorLogSink: ActionErrorLogSink? = null,
    private val actionResultVerifier: ActionResultVerifier? = null,
) {
    private suspend fun trace(id: String, type: ActionTraceType, message: String? = null) = traceSink?.record(ActionTraceEvent(id, type, message))
    private suspend fun logError(executionId: String?, actionId: String?, raw: String, userMessage: String) = errorLogSink?.record(ActionErrorLog(executionId, actionId, userMessage, raw))

    suspend fun prepare(actionId: String, input: Any): PreparedAction {
        val action = registry.find(actionId) ?: error("Action not found: $actionId")
        return PreparedAction(actionId = action.id, input = input, risk = action.risk).also {
            checkpointStore.save(it)
            trace(it.executionId, ActionTraceType.PREPARED, it.actionId)
        }
    }

    fun requiresApproval(prepared: PreparedAction) = approvalPolicy.requiresApproval(prepared.risk)

    suspend fun validate(prepared: PreparedAction, capability: String?): PreparedAction {
        if (capability != null && !capabilityProvider.supports(capability)) {
            val raw = "Required capability is unavailable: $capability"
            logError(prepared.executionId, prepared.actionId, raw, "قابلیت موردنیاز برای اجرای عملیات در دسترس نیست.")
            error(raw)
        }
        val state = if (requiresApproval(prepared)) ActionExecutionState.AwaitingApproval else ActionExecutionState.Approved
        return prepared.copy(state = state).also {
            checkpointStore.save(it)
            trace(it.executionId, ActionTraceType.VALIDATED, capability)
            if (state == ActionExecutionState.AwaitingApproval) trace(it.executionId, ActionTraceType.APPROVAL_REQUESTED)
        }
    }

    suspend fun checkpoint(executionId: String): PreparedAction = checkpointStore.get(executionId) ?: error("Prepared action not found: $executionId")
    suspend fun pendingApprovals(): List<PreparedAction> = checkpointStore.list().filter { it.state == ActionExecutionState.AwaitingApproval }

    suspend fun approve(executionId: String): PreparedAction {
        val prepared = checkpoint(executionId)
        require(prepared.state == ActionExecutionState.AwaitingApproval) { "Action is not awaiting final approval" }
        return prepared.copy(state = ActionExecutionState.Approved).also { checkpointStore.save(it); trace(executionId, ActionTraceType.APPROVED) }
    }

    suspend fun reject(prepared: PreparedAction): PreparedAction {
        require(prepared.state == ActionExecutionState.AwaitingApproval) { "Action is not awaiting approval" }
        return prepared.copy(state = ActionExecutionState.Rejected).also { checkpointStore.save(it); trace(it.executionId, ActionTraceType.REJECTED) }
    }

    suspend fun executeApproved(executionId: String, verifier: Verifier<Any>): ActionExecutionState {
        val prepared = checkpointStore.get(executionId) ?: return ActionExecutionState.Failed("Prepared action not found: $executionId")
        if (prepared.state != ActionExecutionState.Approved) return ActionExecutionState.Failed("Action is not approved for execution")
        val action = registry.find(prepared.actionId) ?: return ActionExecutionState.Failed("Action not found: ${prepared.actionId}")
        return try {
            checkpointStore.save(prepared.copy(state = ActionExecutionState.Executing))
            trace(executionId, ActionTraceType.EXECUTION_STARTED, prepared.actionId)
            val result = action.execute(prepared.input)
            val verification = actionResultVerifier?.verify(prepared, result) ?: verifier.verify(result)
            if (!verification.success) {
                val raw = verification.evidence ?: "Action verification failed"
                checkpointStore.save(prepared.copy(state = ActionExecutionState.Failed(raw)))
                trace(executionId, ActionTraceType.VERIFICATION_FAILED, raw)
                logError(executionId, prepared.actionId, raw, "اجرای عملیات انجام شد، اما نتیجه قابل تأیید نبود.")
                return ActionExecutionState.Failed(raw)
            }
            val completed = ActionExecutionState.Completed(verification)
            checkpointStore.save(prepared.copy(state = completed))
            trace(executionId, ActionTraceType.EXECUTION_COMPLETED)
            completed
        } catch (t: Throwable) {
            val raw = t.message ?: t::class.simpleName.orEmpty()
            checkpointStore.save(prepared.copy(state = ActionExecutionState.Failed(raw)))
            trace(executionId, ActionTraceType.EXECUTION_FAILED, raw)
            logError(executionId, prepared.actionId, raw, "اجرای عملیات با خطا مواجه شد.")
            ActionExecutionState.Failed(raw)
        }
    }

    suspend fun retryFailed(executionId: String, verifier: Verifier<Any>): ActionExecutionState {
        val failed = checkpoint(executionId)
        if (failed.state !is ActionExecutionState.Failed) return ActionExecutionState.Failed("Only failed executions can be retried")
        val maxRetries = retryPolicy.maxRetries(failed).coerceAtLeast(0)
        if (failed.retryCount >= maxRetries) return ActionExecutionState.Failed("Retry limit reached: $maxRetries")
        trace(executionId, ActionTraceType.RETRY_REQUESTED, "attempt=${failed.retryCount + 1}/$maxRetries")
        val reset = failed.copy(state = if (requiresApproval(failed)) ActionExecutionState.AwaitingApproval else ActionExecutionState.Approved, retryCount = failed.retryCount + 1)
        checkpointStore.save(reset)
        return if (reset.state == ActionExecutionState.Approved) executeApproved(executionId, verifier) else reset.state
    }
}
