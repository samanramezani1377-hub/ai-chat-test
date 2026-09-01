package com.woogit.aicore.agent

import com.woogit.aicore.conversation.ConversationMessage
import com.woogit.aicore.conversation.ContextProvider
import com.woogit.aicore.conversation.ConversationStore
import com.woogit.aicore.conversation.DefaultContextProvider
import com.woogit.aicore.domain.GenerationResult
import com.woogit.aicore.domain.InferenceSettings
import java.util.UUID

sealed interface AgentSessionResult {
    data class Reply(
        val generation: GenerationResult,
        val actionPlan: ActionPlan? = null,
        val actionResult: String? = null,
    ) : AgentSessionResult
}

/**
 * Application-level agent loop.
 * The Action System remains the only authority capable of executing an action.
 */
class AgentSession(
    private val orchestrator: AgentOrchestrator,
    private val conversationStore: ConversationStore,
    private val actionPlanCoordinator: ActionPlanCoordinator? = null,
    private val actionExecutor: (suspend (ActionPlan) -> ActionExecutionOutcome)? = null,
    private val eventSink: suspend (AgentEvent) -> Unit = {},
    private val contextProvider: ContextProvider = DefaultContextProvider(
        conversationStore,
        systemContext = { null },
        persistentTaskContext = { null },
        workspaceContext = { null }
    ),
    private val maxActionSteps: Int = 4,
) {
    init { require(maxActionSteps >= 0) { "maxActionSteps must be non-negative" } }

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
            var result = orchestrator.generate(settings, requestedRecentMessages) { token ->
                eventSink(AgentEvent.Token(token))
            }
            var plan: ActionPlan? = null
            var actionResult: String? = null
            var steps = 0

            while (steps < maxActionSteps) {
                conversationStore.append(
                    ConversationMessage(UUID.randomUUID().toString(), ConversationMessage.Role.ASSISTANT, result.text, System.currentTimeMillis())
                )
                plan = actionPlanCoordinator?.prepare(contextProvider.build(requestedRecentMessages)) ?: break
                eventSink(AgentEvent.ActionPrepared(plan.prepared.executionId, plan.prepared.actionId))

                if (plan.prepared.risk == com.woogit.aicore.domain.RiskLevel.SENSITIVE) {
                    eventSink(AgentEvent.ApprovalRequired(plan.prepared.executionId))
                    break
                }

                val executor = actionExecutor ?: break
                val outcome = executor(plan)
                actionResult = outcome.message
                conversationStore.append(
                    ConversationMessage(
                        UUID.randomUUID().toString(),
                        ConversationMessage.Role.TOOL,
                        outcome.toProtocolResult(plan),
                        System.currentTimeMillis()
                    )
                )
                steps++
                if (!outcome.success) break

                result = orchestrator.generate(settings, requestedRecentMessages) { token ->
                    eventSink(AgentEvent.Token(token))
                }
            }

            if (steps == maxActionSteps && maxActionSteps > 0) {
                eventSink(AgentEvent.Failed("Agent action step limit reached"))
            }

            // When an action was executed, the final generation is the post-tool answer.
            // With no action, the first generation is the final answer.
            conversationStore.append(
                ConversationMessage(UUID.randomUUID().toString(), ConversationMessage.Role.ASSISTANT, result.text, System.currentTimeMillis())
            )
            eventSink(AgentEvent.Completed)
            AgentSessionResult.Reply(result, plan, actionResult)
        } catch (t: Throwable) {
            eventSink(AgentEvent.Failed(t.message ?: t::class.simpleName.orEmpty()))
            throw t
        }
    }
}

data class ActionExecutionOutcome(
    val success: Boolean,
    val verified: Boolean,
    val message: String,
    val data: String? = null,
    val errorCode: String? = null,
) {
    fun toProtocolResult(plan: ActionPlan): String = buildString {
        append("{\"version\":1,\"actionId\":\"")
        append(plan.intent.requestId.replace("\\", "\\\\").replace("\"", "\\\""))
        append("\",\"success\":").append(success)
        append(",\"verified\":").append(verified)
        append(",\"data\":")
        append(data?.let { "\"${it.replace("\\", "\\\\").replace("\"", "\\\"")}\"" } ?: "null")
        append(",\"error\":")
        append(errorCode?.let { "\"${it.replace("\"", "\\\"")}" }?.let { "\"$it\"" } ?: "null")
        append("}")
    }
}
