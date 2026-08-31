package com.woogit.aicore.actions

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

class ApprovalController(
    private val checkpoints: ActionCheckpointStore
) {
    suspend fun approve(executionId: String): ApprovalResult = decide(executionId, ApprovalDecision.Approved)

    suspend fun reject(executionId: String): ApprovalResult = decide(executionId, ApprovalDecision.Rejected)

    private suspend fun decide(executionId: String, decision: ApprovalDecision): ApprovalResult {
        val prepared = checkpoints.get(executionId)
            ?: return ApprovalResult(executionId, decision, ActionExecutionState.Failed("Prepared action not found: $executionId"))

        if (prepared.state != ActionExecutionState.AwaitingApproval) {
            return ApprovalResult(executionId, decision, ActionExecutionState.Failed("Action is not awaiting final approval"))
        }

        val state = when (decision) {
            ApprovalDecision.Approved -> ActionExecutionState.AwaitingApproval
            ApprovalDecision.Rejected -> ActionExecutionState.Rejected
        }
        val updated = prepared.copy(state = state)
        checkpoints.save(updated)
        return ApprovalResult(executionId, decision, state)
    }
}
