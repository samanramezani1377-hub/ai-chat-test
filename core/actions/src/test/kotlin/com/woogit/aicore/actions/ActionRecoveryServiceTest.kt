package com.woogit.aicore.actions

import kotlin.test.Test
import kotlin.test.assertIs

class ActionRecoveryServiceTest {
    @Test
    fun failedExecutionRequiresVerificationBeforeRecovery() = kotlinx.coroutines.test.runTest {
        val checkpoints = InMemoryActionCheckpointStore()
        val execution = PreparedAction(actionId = "write", input = "value", risk = com.woogit.aicore.domain.RiskLevel.SENSITIVE, state = ActionExecutionState.Failed("mismatch"))
        checkpoints.save(execution)
        val recovery = ActionRecovery(checkpoints, object : ActionStateVerifier {
            override suspend fun inspect(executionId: String): RecoveryDecision = RecoveryDecision.RequiresVerification
        })
        assertIs<RecoveryDecision.RequiresVerification>(recovery.recover(execution.executionId))
    }

    @Test
    fun missingCheckpointRequiresUserIntervention() = kotlinx.coroutines.test.runTest {
        val recovery = ActionRecovery(InMemoryActionCheckpointStore(), object : ActionStateVerifier {
            override suspend fun inspect(executionId: String): RecoveryDecision = RecoveryDecision.SafeToRetry
        })
        assertIs<RecoveryDecision.RequiresUserIntervention>(recovery.recover("missing"))
    }
}
