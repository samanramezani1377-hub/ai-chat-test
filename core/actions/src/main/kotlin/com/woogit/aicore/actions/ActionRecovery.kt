package com.woogit.aicore.actions

/** Outcome after inspecting persisted state following an interrupted execution. */
sealed interface RecoveryDecision {
    data object AlreadyCompleted : RecoveryDecision
    data object SafeToRetry : RecoveryDecision
    data object RequiresVerification : RecoveryDecision
    data object RequiresUserIntervention : RecoveryDecision
}

interface ActionStateVerifier {
    suspend fun inspect(executionId: String): RecoveryDecision
}

class ActionRecovery(
    private val checkpoints: ActionCheckpointStore,
    private val stateVerifier: ActionStateVerifier
) {
    suspend fun recover(executionId: String): RecoveryDecision {
        val checkpoint = checkpoints.get(executionId)
            ?: return RecoveryDecision.RequiresUserIntervention

        return when (checkpoint.state) {
            is ActionExecutionState.Completed -> RecoveryDecision.AlreadyCompleted
            ActionExecutionState.Prepared,
            ActionExecutionState.AwaitingApproval -> RecoveryDecision.RequiresUserIntervention
            ActionExecutionState.Approved -> RecoveryDecision.SafeToRetry
            ActionExecutionState.Executing,
            ActionExecutionState.Unknown -> stateVerifier.inspect(executionId)
            is ActionExecutionState.Failed -> RecoveryDecision.RequiresVerification
            ActionExecutionState.Rejected -> RecoveryDecision.RequiresUserIntervention
        }
    }
}
