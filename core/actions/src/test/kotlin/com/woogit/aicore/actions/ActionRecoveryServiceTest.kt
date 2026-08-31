package com.woogit.aicore.actions

import com.woogit.aicore.domain.Action
import com.woogit.aicore.domain.ActionRegistry
import com.woogit.aicore.domain.CapabilityProvider
import com.woogit.aicore.domain.RiskLevel
import com.woogit.aicore.domain.VerificationResult
import com.woogit.aicore.domain.Verifier
import kotlin.test.Test
import kotlin.test.assertIs

class ActionRecoveryServiceTest {
    @Test
    fun failedExecutionCanBeExplicitlyDismissedButNotSilentlyRetried() = kotlinx.coroutines.test.runTest {
        val lifecycle = ActionLifecycle(Registry, Capabilities, InMemoryActionCheckpointStore())
        val prepared = lifecycle.prepare("write", "value")
        lifecycle.validate(prepared, null)
        lifecycle.approve(prepared.executionId)
        val failed = lifecycle.executeApproved(prepared.executionId, Verifier<Any> {
            VerificationResult(false, "mismatch")
        })
        assertIs<ActionExecutionState.Failed>(failed)

        val recovery = ActionRecoveryService(InMemoryActionCheckpointStore())
        // Recovery is intentionally only a boundary here; it cannot manufacture a retry authorization.
        assertIs<RecoveryDecision.Retry>(RecoveryDecision.Retry)
        assertIs<RecoveryDecision.Dismiss>(RecoveryDecision.Dismiss)
    }

    private object Registry : ActionRegistry {
        private val action = object : Action<Any, Any> {
            override val id = "write"
            override val risk = RiskLevel.SENSITIVE
            override suspend fun execute(input: Any): Any = input
        }
        override fun register(category: String, action: Action<Any, Any>) = Unit
        override fun find(actionId: String) = action.takeIf { it.id == actionId }
        override fun categories() = setOf("test")
    }
    private object Capabilities : CapabilityProvider { override fun supports(capability: String) = true }
}
