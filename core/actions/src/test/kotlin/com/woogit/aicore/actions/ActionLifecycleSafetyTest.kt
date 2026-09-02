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

class ActionLifecycleSafetyTest {
    @Test
    fun deniedPermissionStopsBeforeApprovalOrExecution() = kotlinx.coroutines.test.runTest {
        val action = TestAction("write", RiskLevel.SENSITIVE)
        val lifecycle = ActionLifecycle(
            TestRegistry(action),
            TestCapabilities,
            InMemoryActionCheckpointStore(),
            permissionPolicy = object : ActionPermissionPolicy {
                override fun isAllowed(actionId: String, permission: String) = false
            }
        )

        val prepared = lifecycle.prepare(action.id, "payload")
        val error = runCatching { lifecycle.validate(prepared, null) }.exceptionOrNull()

        assertEquals("Action permission denied: action:write", error?.message)
        assertEquals(0, action.calls)
    }

    @Test
    fun interruptedExecutionIsMarkedUnknownAndNeverSilentlyRetried() = kotlinx.coroutines.test.runTest {
        val checkpoint = InMemoryActionCheckpointStore()
        val action = TestAction("write", RiskLevel.NORMAL)
        val lifecycle = ActionLifecycle(TestRegistry(action), TestCapabilities, checkpoint)
        val interrupted = PreparedAction(
            executionId = "interrupted",
            actionId = action.id,
            input = "payload",
            risk = action.risk,
            state = ActionExecutionState.Executing
        )
        checkpoint.save(interrupted)

        val recovered = lifecycle.markInterruptedExecutionsUnknown()

        assertEquals(1, recovered.size)
        assertIs<ActionExecutionState.Unknown>(recovered.single().state)
        assertIs<ActionExecutionState.Unknown>(lifecycle.checkpoint(interrupted.executionId).state)

        val retry = lifecycle.retryFailed(interrupted.executionId, TestVerifier)
        assertIs<ActionExecutionState.Failed>(retry)
        assertEquals("Only failed executions can be retried", retry.message)
        assertEquals(0, action.calls)
    }

    @Test
    fun startupReconciliationMarksExecutingCheckpointUnknown() = kotlinx.coroutines.test.runTest {
        val checkpoint = InMemoryActionCheckpointStore()
        val action = PreparedAction(
            executionId = "interrupted",
            actionId = "write",
            input = "payload",
            risk = RiskLevel.NORMAL,
            state = ActionExecutionState.Executing
        )
        checkpoint.save(action)
        val lifecycle = ActionLifecycle(TestRegistry(TestAction("write", RiskLevel.NORMAL)), TestCapabilities, checkpoint)

        val recovered = lifecycle.markInterruptedExecutionsUnknown()

        assertEquals(1, recovered.size)
        assertIs<ActionExecutionState.Unknown>(recovered.single().state)
        assertIs<ActionExecutionState.Unknown>(lifecycle.checkpoint("interrupted").state)
    }

    private class TestAction(
        override val id: String,
        override val risk: RiskLevel,
        private val throws: Boolean = false
    ) : Action<Any, Any> {
        var calls = 0
        override suspend fun execute(input: Any): Any {
            calls++
            if (throws) error("simulated interruption")
            return input
        }
    }

    private class TestRegistry(private val action: Action<Any, Any>) : ActionRegistry {
        override fun register(category: String, action: Action<Any, Any>) = Unit
        override fun find(actionId: String): Action<Any, Any>? = action.takeIf { it.id == actionActionId }
        override fun categories(): Set<String> = setOf("test")
    }

    private object TestCapabilities : CapabilityProvider {
        override fun supports(capability: String): Boolean = true
    }

    private object TestVerifier : Verifier<Any> {
        override suspend fun verify(result: Any): VerificationResult = VerificationResult(true, "verified")
    }
}
