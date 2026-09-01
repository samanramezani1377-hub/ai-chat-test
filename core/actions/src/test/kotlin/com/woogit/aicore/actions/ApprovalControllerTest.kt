package com.woogit.aicore.actions

import com.woogit.aicore.domain.Action
import com.woogit.aicore.domain.ActionRegistry
import com.woogit.aicore.domain.CapabilityProvider
import com.woogit.aicore.domain.RiskLevel
import com.woogit.aicore.domain.Verifier
import com.woogit.aicore.domain.VerificationResult
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs

class ApprovalControllerTest {
    @Test
    fun rejectMakesPreparedActionNonExecutable() = kotlinx.coroutines.test.runTest {
        val store = InMemoryActionCheckpointStore()
        val lifecycle = ActionLifecycle(TestRegistry, TestCapabilities, store)
        val controller = ApprovalController(lifecycle)
        val prepared = lifecycle.prepare("sensitive", "input")
        lifecycle.validate(prepared, null)

        val rejected = controller.reject(prepared.executionId)

        assertEquals(ApprovalDecision.Rejected, rejected.decision)
        assertIs<ActionExecutionState.Rejected>(rejected.state)
        assertIs<ActionExecutionState.Failed>(lifecycle.executeApproved(prepared.executionId, PassingVerifier))
        assertEquals(0, TestRegistry.executions)
    }

    @Test
    fun approvalExecutesExactlyOnceAndSecondApprovalIsRejected() = kotlinx.coroutines.test.runTest {
        TestRegistry.executions = 0
        val store = InMemoryActionCheckpointStore()
        val lifecycle = ActionLifecycle(TestRegistry, TestCapabilities, store)
        val controller = ApprovalController(lifecycle)
        val prepared = lifecycle.prepare("sensitive", "input")
        lifecycle.validate(prepared, null)

        val approved = controller.approve(prepared.executionId)
        val completed = lifecycle.executeApproved(prepared.executionId, PassingVerifier)

        assertEquals(ApprovalDecision.Approved, approved.decision)
        assertIs<ActionExecutionState.Completed>(completed)
        assertEquals(1, TestRegistry.executions)

        val secondApproval = controller.approve(prepared.executionId)
        assertEquals(ApprovalDecision.Failed, secondApproval.decision)
        assertEquals(1, TestRegistry.executions)
    }

    private object PassingVerifier : Verifier<Any> {
        override suspend fun verify(result: Any) = VerificationResult(true, result.toString())
    }

    private object TestRegistry : ActionRegistry {
        var executions: Int = 0
        private val action = object : Action<Any, Any> {
            override val id = "sensitive"
            override val risk = RiskLevel.SENSITIVE
            override suspend fun execute(input: Any): Any {
                executions++
                return input
            }
        }
        override fun register(category: String, action: Action<Any, Any>) = Unit
        override fun find(actionId: String): Action<Any, Any>? = action.takeIf { it.id == actionId }
        override fun categories() = setOf("test")
    }

    private object TestCapabilities : CapabilityProvider {
        override fun supports(capability: String) = true
    }
}
