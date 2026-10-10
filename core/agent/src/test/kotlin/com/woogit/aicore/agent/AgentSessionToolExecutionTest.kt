package com.woogit.aicore.agent

import com.woogit.aicore.actions.ActionExecutionState
import com.woogit.aicore.actions.ActionLifecycle
import com.woogit.aicore.actions.CalculateAction
import com.woogit.aicore.actions.DefaultActionRegistry
import com.woogit.aicore.actions.InMemoryActionCheckpointStore
import com.woogit.aicore.conversation.ConversationMessage
import com.woogit.aicore.conversation.ConversationStore
import com.woogit.aicore.conversation.DefaultContextProvider
import com.woogit.aicore.domain.CapabilityProvider
import com.woogit.aicore.domain.ChatMessage
import com.woogit.aicore.domain.GenerationRequest
import com.woogit.aicore.domain.GenerationResult
import com.woogit.aicore.domain.InferenceSettings
import com.woogit.aicore.domain.ModelRuntime
import com.woogit.aicore.domain.RuntimeInfo
import com.woogit.aicore.domain.VerificationResult
import com.woogit.aicore.domain.Verifier
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import kotlinx.coroutines.test.runTest

class AgentSessionToolExecutionTest {
    @Test
    fun modelReceivesActionCatalogAndItsRequestExecutesBeforeFinalAnswer() = runTest {
        val registry = DefaultActionRegistry().apply { register("utility", CalculateAction()) }
        val lifecycle = ActionLifecycle(
            registry = registry,
            capabilityProvider = object : CapabilityProvider {
                override fun supports(capability: String) = capability == "utility"
            },
            checkpointStore = InMemoryActionCheckpointStore(),
        )
        val store = TestStore()
        val provider = DefaultContextProvider(
            conversationStore = store,
            systemContext = { ActionToolPrompt.build(registry) },
            persistentTaskContext = { null },
            workspaceContext = { null },
        )
        val runtime = SequencedRuntime(
            listOf(
                """{"version":1,"actionId":"calc-1","action":"calculate","arguments":{"expression":"(125 * 8) + (275 * 4) - 300"}}""",
                "The verified calculation result is 1800.",
            )
        )
        val verifier = object : Verifier<Any> {
            override suspend fun verify(result: Any) = VerificationResult(result.toString().isNotBlank(), result.toString())
        }
        val coordinator = ActionPlanCoordinator(lifecycle, ProtocolActionIntentPlanner())
        val events = mutableListOf<AgentEvent>()
        val session = AgentSession(
            orchestrator = AgentOrchestrator(provider, runtime),
            conversationStore = store,
            actionPlanCoordinator = coordinator,
            actionExecutor = { plan ->
                when (val state = lifecycle.executeApproved(plan.prepared.executionId, verifier)) {
                    is ActionExecutionState.Completed -> ActionExecutionOutcome(true, state.verification.success, state.verification.evidence.orEmpty(), state.verification.evidence)
                    is ActionExecutionState.Failed -> ActionExecutionOutcome(false, false, state.message, errorCode = "ACTION_FAILED")
                    else -> ActionExecutionOutcome(false, false, "Action was not executed")
                }
            },
            eventSink = { events += it },
            contextProvider = provider,
            maxActionSteps = 2,
            conversationId = "test-conversation",
        )

        val reply = session.send("Calculate this and use the calculation tool.", InferenceSettings(maxNewTokens = 128, maxActionSteps = 2))

        assertEquals(2, runtime.requests.size)
        assertTrue(runtime.requests.first().messages.any {
            it.role == ChatMessage.Role.SYSTEM && it.content.contains("calculate") && it.content.contains("ActionRequest v1")
        })
        assertTrue(runtime.requests.last().messages.any {
            it.role == ChatMessage.Role.TOOL && it.content.contains("\"success\":true") && it.content.contains("1800")
        })
        assertEquals("The verified calculation result is 1800.", reply.generation.text)
        assertEquals("calculate", assertNotNull(reply.actionPlan).intent.actionId)
        assertEquals("1800", reply.actionResult)
        assertTrue(events.any { it is AgentEvent.ActionExecuted })
        assertTrue(store.messages.any { it.role == ConversationMessage.Role.TOOL })
    }

    private class TestStore : ConversationStore {
        val messages = mutableListOf<ConversationMessage>()
        override suspend fun append(message: ConversationMessage) { messages += message }
        override suspend fun recent(limit: Int) = messages.takeLast(limit)
        override suspend fun summary(): String? = null
        override suspend fun replaceSummary(summary: String) = Unit
    }

    private class SequencedRuntime(private val outputs: List<String>) : ModelRuntime {
        val requests = mutableListOf<GenerationRequest>()
        private var next = 0
        override suspend fun load(model: com.woogit.aicore.domain.ModelDescriptor) = Unit
        override suspend fun unload() = Unit
        override suspend fun generate(request: GenerationRequest, onToken: suspend (String) -> Unit): GenerationResult {
            requests += request
            val text = outputs[next++]
            onToken(text)
            return GenerationResult(text)
        }
        override suspend fun stopGeneration() = Unit
        override fun runtimeInfo() = RuntimeInfo("test", "1", "test")
    }
}
