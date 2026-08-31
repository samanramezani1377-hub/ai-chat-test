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

class ActionApprovalSecurityTest {
    @Test
    fun approvalUsesPersistedInputAndActionIdentity() = kotlinx.coroutines.test.runTest {
        val store = InMemoryActionCheckpointStore()
        val action = RecordingAction()
        val lifecycle = ActionLifecycle(Registry(action), Capabilities, store)
        val prepared = lifecycle.prepare("sensitive", "original-input")
        lifecycle.validate(prepared, null)
        val controller = ApprovalController(lifecycle)

        val approved = controller.approve(prepared.executionId)
        assertEquals(ApprovalDecision.Approved, approved.decision)
        val result = lifecycle.executeApproved(prepared.executionId, object : Verifier<Any> {
            override fun verify(output: Any): VerificationResult = VerificationResult(true, "ok")
        })

        assertIs<ActionExecutionState.Completed>(result)
        assertEquals("original-input", action.receivedInput)
        assertEquals(1, action.calls)
    }

    private class RecordingAction : Action<Any, Any> {
        override val id = "sensitive"
        override val risk = RiskLevel.SENSITIVE
        var receivedInput: Any? = null
        var calls = 0
        override suspend fun execute(input: Any): Any {
            receivedInput = input
            calls++
            return input
        }
    }

    private class Registry(private val action: Action<Any, Any>) : ActionRegistry {
        override fun register(category: String, action: Action<Any, Any>) = Unit
        override fun find(actionId: String) = action.takeIf { it.id == actionId }
        override fun categories() = setOf("test")
    }

    private object Capabilities : CapabilityProvider {
        override fun supports(capability: String) = true
    }
}
