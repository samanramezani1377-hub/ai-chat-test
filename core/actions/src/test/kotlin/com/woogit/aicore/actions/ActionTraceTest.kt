package com.woogit.aicore.actions

import com.woogit.aicore.domain.Action
import com.woogit.aicore.domain.ActionRegistry
import com.woogit.aicore.domain.CapabilityProvider
import com.woogit.aicore.domain.RiskLevel
import com.woogit.aicore.domain.VerificationResult
import com.woogit.aicore.domain.Verifier
import kotlin.test.Test
import kotlin.test.assertEquals

class ActionTraceTest {
    @Test
    fun lifecycleTraceContainsOrderedTechnicalEvents() = kotlinx.coroutines.test.runTest {
        val trace = InMemoryActionTraceSink()
        val lifecycle = ActionLifecycle(Registry, Capabilities, InMemoryActionCheckpointStore(), traceSink = trace)
        val prepared = lifecycle.prepare("write", "value")
        lifecycle.validate(prepared, null)
        lifecycle.approve(prepared.executionId)
        lifecycle.executeApproved(prepared.executionId, Verifier<Any> { VerificationResult(true, "verified") })

        assertEquals(
            listOf(
                ActionTraceType.PREPARED,
                ActionTraceType.VALIDATED,
                ActionTraceType.APPROVAL_REQUESTED,
                ActionTraceType.APPROVED,
                ActionTraceType.EXECUTION_STARTED,
                ActionTraceType.EXECUTION_COMPLETED
            ),
            trace.snapshot().map { it.type }
        )
        assertEquals(setOf(prepared.executionId), trace.snapshot().map { it.executionId }.toSet())
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

    private object Capabilities : CapabilityProvider {
        override fun supports(capability: String) = true
    }
}
