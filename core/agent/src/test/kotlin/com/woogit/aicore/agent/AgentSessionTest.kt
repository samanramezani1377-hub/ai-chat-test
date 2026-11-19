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
    fun sendPublishesSafeProseChunksInOrderWithoutAppendingTheFinalTextTwice() = kotlinx.coroutines.test.runTest {
        val store = TestStore()
        val runtime = TestRuntime(listOf("Hello", ", ", "world."))
        val provider = DefaultContextProvider(store, { "system" }, { "task" }, { "workspace" })
        val events = mutableListOf<AgentEvent>()
        val session = AgentSession(
            orchestrator = AgentOrchestrator(provider, runtime),
            conversationStore = store,
            eventSink = { event -> events += event },
        )

        val reply = session.send("say hello", InferenceSettings())

        assertEquals("Hello, world.", reply.generation.text)
        assertEquals(
            listOf("Hello", ", ", "world."),
            events.filterIsInstance<AgentEvent.Token>().map { it.value },
        )
        assertEquals("Hello, world.", store.messages.last().content)
    }

    @Test
    fun textStreamingStartsImmediatelyForShortPlainResponses() {
        val streamer = SafeAgentResponseStreamer()

        assertEquals(listOf("Hi"), streamer.accept("Hi"))
        assertEquals(listOf("!"), streamer.accept("!"))
        assertTrue(streamer.finish().isEmpty())
    }

    @Test
    fun textStreamingSkipsReasoningMarkersAcrossChunkBoundaries() {
        val streamer = SafeAgentResponseStreamer()
        val output = buildList {
            listOf("<thi", "nk>private analysis", "</thi", "nk>The answer", " is 42.")
                .forEach { addAll(streamer.accept(it)) }
            addAll(streamer.finish())
        }

        assertEquals("The answer is 42.", output.joinToString(""))
        assertTrue(output.none { it.contains("think") || it.contains("private analysis") })
    }

    @Test
    fun textStreamingWithholdsStructuredActionPayloads() {
        val streamer = SafeAgentResponseStreamer()

        assertTrue(streamer.accept("""{"version":1,"actionId":"x"}""").isEmpty())
        assertTrue(streamer.finish().isEmpty())
    }

    @Test
    fun textStreamingCanResumeAfterReasoningAndStillAvoidDuplicateFinalText() = kotlinx.coroutines.test.runTest {
        val store = TestStore()
        val runtime = TestRuntime(listOf("<think>hidden</think>", "Visible", " answer"))
        val provider = DefaultContextProvider(store, { "system" }, { "task" }, { "workspace" })
        val events = mutableListOf<AgentEvent>()
        val session = AgentSession(
            orchestrator = AgentOrchestrator(provider, runtime),
            conversationStore = store,
            eventSink = { event -> events += event },
            textChatStreaming = true,
        )

        val reply = session.send("question", InferenceSettings())

        assertEquals("Visible answer", reply.generation.text)
        assertEquals(listOf("Visible", " answer"), events.filterIsInstance<AgentEvent.Token>().map { it.value })
        assertEquals("Visible answer", store.messages.last().content)
    }

    @Test
    fun voiceSessionKeepsItsLegacyNonIncrementalReasoningStreamPath() = kotlinx.coroutines.test.runTest {
        val store = TestStore()
        val runtime = TestRuntime(listOf("<think>private analysis</think>", "Voice answer"))
        val provider = DefaultContextProvider(store, { "system" }, { "task" }, { "workspace" })
        val events = mutableListOf<AgentEvent>()
        val session = AgentSession(
            orchestrator = AgentOrchestrator(provider, runtime),
            conversationStore = store,
            eventSink = { event -> events += event },
            textChatStreaming = false,
        )

        val reply = session.send("voice question", InferenceSettings())

        assertEquals("Voice answer", reply.generation.text)
        assertEquals(listOf("Voice answer"), events.filterIsInstance<AgentEvent.Token>().map { it.value })
    }

    @Test
    fun textChatExtractsAssistantBodyFromChatMlEnvelope() = kotlinx.coroutines.test.runTest {
        val store = TestStore()
        val raw = "<|im_start|>user\nسلام<|im_end|>\n<|im_start|>assistant\nسلام!<|im_end|>"
        val runtime = TestRuntime(listOf(raw))
        val provider = DefaultContextProvider(store, { "system" }, { "task" }, { "workspace" })
        val events = mutableListOf<AgentEvent>()
        val session = AgentSession(
            orchestrator = AgentOrchestrator(provider, runtime),
            conversationStore = store,
            eventSink = { event -> events += event },
            textChatStreaming = true,
        )

        val reply = session.send("سلام", InferenceSettings())

        assertEquals("سلام!", reply.generation.text)
        assertEquals(listOf("سلام!"), events.filterIsInstance<AgentEvent.Token>().map { it.value })
        assertEquals("سلام!", store.messages.last().content)
    }

    @Test
    fun textChatNeverEchoesUserEnvelopeWhenModelReturnsOnlyChatMlUserMessage() = kotlinx.coroutines.test.runTest {
        val store = TestStore()
        val raw = "<|im_start|>user\nسلام"
        val runtime = TestRuntime(listOf(raw))
        val provider = DefaultContextProvider(store, { "system" }, { "task" }, { "workspace" })
        val events = mutableListOf<AgentEvent>()
        val session = AgentSession(
            orchestrator = AgentOrchestrator(provider, runtime),
            conversationStore = store,
            eventSink = { event -> events += event },
            textChatStreaming = true,
        )

        val reply = session.send("سلام", InferenceSettings())

        assertTrue(!reply.generation.text.contains("<|im_start|>"))
        assertTrue(!reply.generation.text.contains("user"))
        assertTrue(!reply.generation.text.contains("سلام"))
        assertTrue(events.filterIsInstance<AgentEvent.Token>().none { it.value.contains("<|im_start|>") })
    }

    @Test
    fun textChatRemovesChatMlControlTokensButKeepsAssistantText() = kotlinx.coroutines.test.runTest {
        val store = TestStore()
        val runtime = TestRuntime(listOf("<|im_start|>assistant\nHello<|im_end|>"))
        val provider = DefaultContextProvider(store, { "system" }, { "task" }, { "workspace" })
        val events = mutableListOf<AgentEvent>()
        val session = AgentSession(
            orchestrator = AgentOrchestrator(provider, runtime),
            conversationStore = store,
            eventSink = { event -> events += event },
            textChatStreaming = true,
        )

        val reply = session.send("hello", InferenceSettings())

        assertEquals("Hello", reply.generation.text)
        assertEquals(listOf("Hello"), events.filterIsInstance<AgentEvent.Token>().map { it.value })
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

    private class TestRuntime(private val chunks: List<String> = listOf("answer")) : ModelRuntime {
        var lastRequest: GenerationRequest? = null
        override suspend fun load(model: com.woogit.aicore.domain.ModelDescriptor) = Unit
        override suspend fun unload() = Unit
        override suspend fun generate(request: GenerationRequest, onToken: suspend (String) -> Unit): GenerationResult {
            lastRequest = request
            chunks.forEach { onToken(it) }
            return GenerationResult(chunks.joinToString(""))
        }
        override suspend fun stopGeneration() = Unit
        override fun runtimeInfo() = RuntimeInfo("test", "1", "test")
    }
}
