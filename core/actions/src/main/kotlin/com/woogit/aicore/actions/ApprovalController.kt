package com.woogit.aicore.actions

import kotlinx.coroutines.CancellationException

/** Final user decision over an already prepared execution. The UI cannot alter its input. */
sealed interface ApprovalDecision {
    data object Approved : ApprovalDecision
    data object Rejected : ApprovalDecision
}

data class ApprovalResult(
    val executionId: String,
    val decision: ApprovalDecision,
    val state: ActionExecutionState
)

class ApprovalController(private val lifecycle: ActionLifecycle) {
    suspend fun approve(executionId: String): ApprovalResult = try {
        val prepared = lifecycle.approve(executionId)
        ApprovalResult(executionId, ApprovalDecision.Approved, prepared.state)
    } catch (t: Throwable) {
        if (t is CancellationException) throw t
        ApprovalResult(executionId, ApprovalDecision.Rejected, ActionExecutionState.Failed(t.message ?: "Approval failed"))
    }

    suspend fun reject(executionId: String): ApprovalResult = try {
        val prepared = lifecycle.checkpoint(executionId)
        require(prepared.state == ActionExecutionState.AwaitingApproval) { "Action is not awaiting final approval" }
        val rejected = lifecycle.reject(prepared)
        ApprovalResult(executionId, ApprovalDecision.Rejected, rejected.state)
    } catch (t: Throwable) {
        if (t is CancellationException) throw t
        ApprovalResult(executionId, ApprovalDecision.Rejected, ActionExecutionState.Failed(t.message ?: "Rejection failed"))
    }
}
