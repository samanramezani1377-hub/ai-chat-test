package com.woogit.aicore.agent

import com.woogit.aicore.conversation.ConversationMessage
import com.woogit.aicore.conversation.ConversationStore
import com.woogit.aicore.conversation.DefaultContextProvider
import com.woogit.aicore.domain.GenerationRequest
import com.woogit.aicore.domain.GenerationResult
import com.woogit.aicore.domain.InferenceSettings
import com.woogit.aicore.domain.ModelRuntime
import com.woogit.aicore.domain.RuntimeInfo
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class AgentSessionTest {
    @Test
    fun sendPersistsUserAndAssistantMessagesAndStreamsTokens() = kotlinx.coroutines.test.runTest {
        val store = TestStore()
        val runtime = TestRuntime()
        val provider = DefaultContextProvider(store, { "system" }, { "task" }, { "workspace" })
        val orchestrator = AgentOrchestrator(provider, runtime)
        val events = mutableListOf<AgentEvent>()
        val session = AgentSession(orchestrator = orchestrator, conversationStore = store, eventSink = { event -> events += event })

        session.send("hello", InferenceSettings())

        assertEquals(listOf("hello", "answer"), store.messages.map { it.content })
        assertTrue(events.any { it is AgentEvent.Token && it.value == "answer" })
        assertTrue(events.last() is AgentEvent.Completed)
    }

    @Test
    fun sendUsesSuppliedInferenceSettingsInsteadOfGlobalSnapshot() = kotlinx.coroutines.test.runTest {
        val store = TestStore()
        val runtime = TestRuntime()
        val provider = DefaultContextProvider(store, { "system" }, { "task" }, { "workspace" })
        val session = AgentSession(AgentOrchestrator(provider, runtime), store)
        val supplied = InferenceSettings(temperature = 0.15, maxNewTokens = 77, recentMessages = 3, maxActionSteps = 1)

        session.send("settings", supplied)

        assertEquals(0.15, runtime.lastRequest!!.settings.temperature)
        assertEquals(77, runtime.lastRequest!!.settings.maxNewTokens)
    }

    private class TestStore : ConversationStore {
        val messages = mutableListOf<ConversationMessage>()
        var summaryValue: String? = null
        override suspend fun append(message: ConversationMessage) { messages += message }
        override suspend fun recent(limit: Int) = messages.takeLast(limit)
        override suspend fun summary() = summaryValue
        override suspend fun replaceSummary(summary: String) { summaryValue = summary }
    }

    private class TestRuntime : ModelRuntime {
        var lastRequest: GenerationRequest? = null
        override suspend fun load(model: com.woogit.aicore.domain.ModelDescriptor) = Unit
        override suspend fun unload() = Unit
        override suspend fun generate(request: GenerationRequest, onToken: suspend (String) -> Unit): GenerationResult {
            lastRequest = request
            onToken("answer")
            return GenerationResult("answer")
        }
        override suspend fun stopGeneration() = Unit
        override fun runtimeInfo() = RuntimeInfo("test", "1", "test")
    }
}
