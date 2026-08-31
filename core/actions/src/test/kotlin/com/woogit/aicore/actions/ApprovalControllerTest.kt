package com.woogit.aicore.actions

import com.woogit.aicore.domain.Action
import com.woogit.aicore.domain.ActionRegistry
import com.woogit.aicore.domain.CapabilityProvider
import com.woogit.aicore.domain.RiskLevel
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
        assertIs<ActionExecutionState.Failed>(
            lifecycle.executeApproved(prepared.executionId, object : com.woogit.aicore.domain.Verifier<Any> {
                override suspend fun verify(result: Any) = com.woogit.aicore.domain.VerificationResult(true)
            })
        )
    }

    private object TestRegistry : ActionRegistry {
        private val action = object : Action<Any, Any> {
            override val id = "sensitive"
            override val risk = RiskLevel.SENSITIVE
            override suspend fun execute(input: Any): Any = input
        }
        override fun register(category: String, action: Action<Any, Any>) = Unit
        override fun find(actionId: String): Action<Any, Any>? = action.takeIf { it.id == actionId }
        override fun categories() = setOf("test")
    }

    private object TestCapabilities : CapabilityProvider {
        override fun supports(capability: String) = true
    }
}
