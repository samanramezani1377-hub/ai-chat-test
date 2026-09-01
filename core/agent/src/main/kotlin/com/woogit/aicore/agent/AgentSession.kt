package com.woogit.aicore.agent

import com.woogit.aicore.conversation.ConversationMessage
import com.woogit.aicore.conversation.ContextProvider
import com.woogit.aicore.conversation.ConversationStore
import com.woogit.aicore.conversation.DefaultContextProvider
import com.woogit.aicore.domain.GenerationResult
import com.woogit.aicore.domain.InferenceSettings
import java.util.UUID

sealed interface AgentSessionResult {
    data class Reply(val generation: GenerationResult, val actionPlan: ActionPlan? = null) : AgentSessionResult
}

class AgentSession(
    private val orchestrator: AgentOrchestrator,
    private val conversationStore: ConversationStore,
    private val actionPlanCoordinator: ActionPlanCoordinator? = null,
    private val eventSink: suspend (AgentEvent) -> Unit = {},
    private val contextProvider: ContextProvider = DefaultContextProvider(
        conversationStore,
        systemContext = { null },
        persistentTaskContext = { null },
        workspaceContext = { null }
    )
) {
    suspend fun send(
        content: String,
        settings: InferenceSettings,
        requestedRecentMessages: Int = 10
    ): AgentSessionResult.Reply {
        require(content.isNotBlank()) { "content must not be blank" }
        conversationStore.append(
            ConversationMessage(UUID.randomUUID().toString(), ConversationMessage.Role.USER, content, System.currentTimeMillis())
        )
        eventSink(AgentEvent.Started)

        return try {
            val result = orchestrator.generate(settings, requestedRecentMessages) { token ->
                eventSink(AgentEvent.Token(token))
            }
            conversationStore.append(
                ConversationMessage(UUID.randomUUID().toString(), ConversationMessage.Role.ASSISTANT, result.text, System.currentTimeMillis())
            )

            val context = contextProvider.build(requestedRecentMessages)
            val plan = actionPlanCoordinator?.prepare(context)
            plan?.let {
                eventSink(AgentEvent.ActionPrepared(it.prepared.executionId, it.prepared.actionId))
                if (it.prepared.risk == com.woogit.aicore.domain.RiskLevel.SENSITIVE) {
                    eventSink(AgentEvent.ApprovalRequired(it.prepared.executionId))
                }
            }
            eventSink(AgentEvent.Completed)
            AgentSessionResult.Reply(result, plan)
        } catch (t: Throwable) {
            eventSink(AgentEvent.Failed(t.message ?: t::class.simpleName.orEmpty()))
            throw t
        }
    }
}
