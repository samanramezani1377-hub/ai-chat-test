package com.woogit.aicore.actions

import com.woogit.aicore.domain.Action
import com.woogit.aicore.domain.ActionRegistry
import com.woogit.aicore.domain.CapabilityProvider
import com.woogit.aicore.domain.RiskLevel
import com.woogit.aicore.domain.VerificationResult
import com.woogit.aicore.domain.Verifier
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertTrue

class ActionLifecycleTest {
    @Test
    fun sensitiveActionCannotExecuteBeforeApproval() = kotlinx.coroutines.test.runTest {
        val checkpoint = InMemoryActionCheckpointStore()
        val registry = TestRegistry(TestAction("change-price", RiskLevel.SENSITIVE))
        val lifecycle = ActionLifecycle(registry, TestCapabilities, checkpoint)
        val prepared = lifecycle.prepare("change-price", "100")
        val validated = lifecycle.validate(prepared, null)

        assertTrue(lifecycle.requiresApproval(validated))
        assertIs<ActionExecutionState.Failed>(lifecycle.executeApproved(prepared.executionId, TestVerifier))
    }

    @Test
    fun approvalExecutesPreparedActionAndVerifies() = kotlinx.coroutines.test.runTest {
        val checkpoint = InMemoryActionCheckpointStore()
        val action = TestAction("change-price", RiskLevel.SENSITIVE)
        val lifecycle = ActionLifecycle(TestRegistry(action), TestCapabilities, checkpoint)
        val prepared = lifecycle.prepare("change-price", "100")
        lifecycle.validate(prepared, null)
        val approved = lifecycle.approve(prepared.executionId)
        assertEquals(ActionExecutionState.Approved, approved.state)

        val result = lifecycle.executeApproved(prepared.executionId, TestVerifier)

        assertIs<ActionExecutionState.Completed>(result)
        assertEquals("100", action.lastInput)
        assertEquals(VerificationResult(true, "verified"), (result as ActionExecutionState.Completed).verification)
    }

    @Test
    fun rejectedActionCannotExecute() = kotlinx.coroutines.test.runTest {
        val checkpoint = InMemoryActionCheckpointStore()
        val action = TestAction("change-price", RiskLevel.SENSITIVE)
        val lifecycle = ActionLifecycle(TestRegistry(action), TestCapabilities, checkpoint)
        val prepared = lifecycle.prepare("change-price", "100")
        lifecycle.validate(prepared, null)
        val rejected = lifecycle.reject(prepared)
        assertIs<ActionExecutionState.Rejected>(rejected.state)
        assertIs<ActionExecutionState.Failed>(lifecycle.executeApproved(prepared.executionId, TestVerifier))
        assertEquals(null, action.lastInput)
    }

    @Test
    fun completedActionCannotExecuteAgain() = kotlinx.coroutines.test.runTest {
        val checkpoint = InMemoryActionCheckpointStore()
        val action = TestAction("change-price", RiskLevel.SENSITIVE)
        val lifecycle = ActionLifecycle(TestRegistry(action), TestCapabilities, checkpoint)
        val prepared = lifecycle.prepare("change-price", "100")
        lifecycle.validate(prepared, null)
        lifecycle.approve(prepared.executionId)
        assertIs<ActionExecutionState.Completed>(lifecycle.executeApproved(prepared.executionId, TestVerifier))
        assertIs<ActionExecutionState.Failed>(lifecycle.executeApproved(prepared.executionId, TestVerifier))
        assertEquals(1, action.calls)
    }

    private class TestAction(
        override val id: String,
        override val risk: RiskLevel
    ) : Action<Any, Any> {
        var lastInput: Any? = null
        var calls = 0
        override suspend fun execute(input: Any): Any {
            lastInput = input
            calls++
            return input
        }
    }

    private class TestRegistry(private val action: Action<Any, Any>) : ActionRegistry {
        override fun register(category: String, action: Action<Any, Any>) = Unit
        override fun find(actionId: String): Action<Any, Any>? = action.takeIf { it.id == actionId }
        override fun categories(): Set<String> = setOf("test")
    }

    private object TestCapabilities : CapabilityProvider {
        override fun supports(capability: String): Boolean = true
    }

    private object TestVerifier : Verifier<Any> {
        override suspend fun verify(result: Any): VerificationResult = VerificationResult(true, "verified")
    }
}
