package com.woogit.aicore.actions

import com.woogit.aicore.domain.Verifier

/**
 * Single application-facing entry point for final approval and execution.
 * The service deliberately accepts only an execution id after planning.
 */
class ActionExecutionService(
    private val lifecycle: ActionLifecycle,
    private val approvalController: ApprovalController
) {
    suspend fun approveAndExecute(
        executionId: String,
        verifier: Verifier<Any>
    ): ActionExecutionState {
        val approval = approvalController.approve(executionId)
        if (approval.state != ActionExecutionState.Approved) {
            return approval.state
        }
        return lifecycle.executeApproved(executionId, verifier)
    }
}
