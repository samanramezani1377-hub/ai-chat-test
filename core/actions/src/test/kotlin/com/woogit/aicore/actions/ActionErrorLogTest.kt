package com.woogit.aicore.actions

import com.woogit.aicore.domain.Action
import com.woogit.aicore.domain.ActionRegistry
import com.woogit.aicore.domain.CapabilityProvider
import com.woogit.aicore.domain.RiskLevel
import com.woogit.aicore.domain.VerificationResult
import com.woogit.aicore.domain.Verifier
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ActionErrorLogTest {
    @Test
    fun runtimeFailureKeepsPersianMessageAndRawMachineError() = kotlinx.coroutines.test.runTest {
        val errors = InMemoryActionErrorLogSink()
        val lifecycle = ActionLifecycle(Registry, Capabilities, InMemoryActionCheckpointStore(), errorLogSink = errors)
        val prepared = lifecycle.prepare("failing", "value")
        lifecycle.validate(prepared, null)
        lifecycle.approve(prepared.executionId)

        val state = lifecycle.executeApproved(prepared.executionId, object : Verifier<Any> {
            override suspend fun verify(result: Any): VerificationResult = VerificationResult(true, "verified")
        })

        assertTrue(state is ActionExecutionState.Failed)
        val log = errors.snapshot().last()
        assertEquals(prepared.executionId, log.executionId)
        assertEquals("failing", log.actionId)
        assertEquals("اجرای عملیات با خطا مواجه شد.", log.userMessageFa)
        assertEquals("machine failure: simulated", log.rawMachineError)
    }

    private object Registry : ActionRegistry {
        private val action = object : Action<Any, Any> {
            override val id = "failing"
            override val risk = RiskLevel.SENSITIVE
            override suspend fun execute(input: Any): Any = error("machine failure: simulated")
        }
        override fun register(category: String, action: Action<Any, Any>) = Unit
        override fun find(actionId: String) = action.takeIf { it.id == actionId }
        override fun categories() = setOf("test")
    }

    private object Capabilities : CapabilityProvider {
        override fun supports(capability: String) = true
    }
}
