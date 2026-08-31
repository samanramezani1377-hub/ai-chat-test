package com.woogit.aicore.actions

import com.woogit.aicore.domain.Action
import com.woogit.aicore.domain.ActionRegistry
import com.woogit.aicore.domain.CapabilityProvider
import com.woogit.aicore.domain.RiskLevel
import com.woogit.aicore.domain.VerificationResult
import com.woogit.aicore.domain.Verifier
import kotlin.test.Test
import kotlin.test.assertIs
import kotlin.test.assertEquals

class VerificationFailureTest {
    @Test
    fun failedVerificationNeverBecomesCompleted() = kotlinx.coroutines.test.runTest {
        val action = object : Action<Any, Any> {
            override val id = "write"
            override val risk = RiskLevel.SENSITIVE
            override suspend fun execute(input: Any): Any = input
        }
        val registry = object : ActionRegistry {
            override fun register(category: String, action: Action<Any, Any>) = Unit
            override fun find(actionId: String) = action.takeIf { it.id == actionId }
            override fun categories() = setOf("test")
        }
        val lifecycle = ActionLifecycle(registry, object : CapabilityProvider {
            override fun supports(capability: String) = true
        }, InMemoryActionCheckpointStore())
        val prepared = lifecycle.prepare("write", "value")
        lifecycle.validate(prepared, null)
        lifecycle.approve(prepared.executionId)

        val state = lifecycle.executeApproved(prepared.executionId, Verifier<Any> {
            VerificationResult(false, "verification mismatch")
        })

        assertIs<ActionExecutionState.Failed>(state)
        assertEquals("verification mismatch", (state as ActionExecutionState.Failed).message)
    }
}
