package com.woogit.aicore.actions

/**
 * Recovery boundary for failed executions. Recovery never silently retries a sensitive action.
 * A caller must explicitly choose whether the failed execution may be retried.
 */
sealed interface RecoveryDecision {
    data object Retry : RecoveryDecision
    data object Dismiss : RecoveryDecision
}

class ActionRecoveryService(private val checkpoints: ActionCheckpointStore) {
    suspend fun inspect(executionId: String): PreparedAction? = checkpoints.get(executionId)

    suspend fun dismiss(executionId: String): ActionExecutionState {
        val current = checkpoints.get(executionId)
            ?: return ActionExecutionState.Failed("Prepared action not found: $executionId")
        if (current.state !is ActionExecutionState.Failed) {
            return ActionExecutionState.Failed("Only failed executions can be dismissed")
        }
        val rejected = current.copy(state = ActionExecutionState.Rejected)
        checkpoints.save(rejected)
        return rejected.state
    }
}
