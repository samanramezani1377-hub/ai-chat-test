package com.woogit.aicore.agent

import com.woogit.aicore.conversation.ConversationStore
import com.woogit.aicore.conversation.ConversationMessage
import com.woogit.aicore.domain.GenerationResult
import com.woogit.aicore.domain.InferenceSettings
import java.util.UUID

class AgentSession(
    private val orchestrator: AgentOrchestrator,
    private val conversationStore: ConversationStore,
    private val eventSink: suspend (AgentEvent) -> Unit = {}
) {
    suspend fun send(
        content: String,
        settings: InferenceSettings,
        requestedRecentMessages: Int = 10
    ): GenerationResult {
        require(content.isNotBlank()) { "content must not be blank" }
        val timestamp = System.currentTimeMillis()
        conversationStore.append(
            ConversationMessage(UUID.randomUUID().toString(), ConversationMessage.Role.USER, content, timestamp)
        )
        eventSink(AgentEvent.Started)

        return try {
            val result = orchestrator.generate(settings, requestedRecentMessages) { token ->
                eventSink(AgentEvent.Token(token))
            }
            conversationStore.append(
                ConversationMessage(
                    UUID.randomUUID().toString(),
                    ConversationMessage.Role.ASSISTANT,
                    result.text,
                    System.currentTimeMillis()
                )
            )
            eventSink(AgentEvent.Completed)
            result
        } catch (t: Throwable) {
            eventSink(AgentEvent.Failed(t.message ?: t::class.simpleName.orEmpty()))
            throw t
        }
    }
}
