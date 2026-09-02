package com.woogit.aicore.agent

import com.woogit.aicore.conversation.ConversationMessage
import com.woogit.aicore.conversation.ContextProvider
import com.woogit.aicore.conversation.ConversationStore
import com.woogit.aicore.conversation.DefaultContextProvider
import com.woogit.aicore.domain.GenerationResult
import com.woogit.aicore.domain.InferenceSettings
import com.woogit.aicore.domain.InferenceSettingsStore
import java.util.UUID

sealed interface AgentSessionResult {
    data class Reply(
        val generation: GenerationResult,
        val actionPlan: ActionPlan? = null,
        val actionResult: String? = null,
    ) : AgentSessionResult
}

/** Application-level model → action → tool result → final answer loop. */
class AgentSession(
    private val orchestrator: AgentOrchestrator,
    private val conversationStore: ConversationStore,
    private val actionPlanCoordinator: ActionPlanCoordinator? = null,
    private val actionExecutor: (suspend (ActionPlan) -> ActionExecutionOutcome)? = null,
    private val eventSink: suspend (AgentEvent) -> Unit = {},
    private val contextProvider: ContextProvider = DefaultContextProvider(conversationStore, systemContext = { null }, persistentTaskContext = { null }, workspaceContext = { null }),
    /** Null means no artificial agent-step cap; a user/runtime setting may provide the cap explicitly. */
    private val maxActionSteps: Int? = null,
    private val conversationId: String? = null,
) {
    init { require(maxActionSteps == null || maxActionSteps >= 0) { "maxActionSteps must be non-negative" } }

    suspend fun send(content: String, settings: InferenceSettings, requestedRecentMessages: Int = 10): AgentSessionResult.Reply {
        require(content.isNotBlank()) { "content must not be blank" }
        val effectiveSettings = InferenceSettingsStore.current
        conversationStore.append(ConversationMessage(UUID.randomUUID().toString(), ConversationMessage.Role.USER, content, System.currentTimeMillis()))
        eventSink(AgentEvent.Started)
        return try {
            var result = orchestrator.generate(effectiveSettings, requestedRecentMessages) { token -> eventSink(AgentEvent.Token(token)) }
            var plan: ActionPlan? = null
            var actionResult: String? = null
            var steps = 0
            var resultPersisted = false
            while (actionPlanCoordinator != null && (maxActionSteps == null || steps < maxActionSteps)) {
                conversationStore.append(ConversationMessage(UUID.randomUUID().toString(), ConversationMessage.Role.ASSISTANT, result.text, System.currentTimeMillis()))
                resultPersisted = true
                plan = actionPlanCoordinator.prepare(contextProvider.build(requestedRecentMessages), conversationId) ?: break
                eventSink(AgentEvent.ActionPrepared(plan.prepared.executionId, plan.prepared.actionId))
                if (plan.prepared.risk == com.woogit.aicore.domain.RiskLevel.SENSITIVE) {
                    eventSink(AgentEvent.ApprovalRequired(plan.prepared.executionId))
                    break
                }
                val executor = actionExecutor ?: break
                val outcome = executor(plan)
                actionResult = outcome.message
                eventSink(AgentEvent.ActionExecuted(plan.prepared.executionId))
                conversationStore.append(ConversationMessage(UUID.randomUUID().toString(), ConversationMessage.Role.TOOL, outcome.toProtocolResult(plan), System.currentTimeMillis()))
                steps++
                // A real Agent result is successful only after verification. Never let an
                // unverified executor result drive another model step or be reported as success.
                if (!outcome.success || !outcome.verified) {
                    result = orchestrator.generate(effectiveSettings, requestedRecentMessages) { token -> eventSink(AgentEvent.Token(token)) }
                    resultPersisted = false
                    break
                }
                result = orchestrator.generate(effectiveSettings, requestedRecentMessages) { token -> eventSink(AgentEvent.Token(token)) }
                resultPersisted = false
            }
            if (!resultPersisted) conversationStore.append(ConversationMessage(UUID.randomUUID().toString(), ConversationMessage.Role.ASSISTANT, result.text, System.currentTimeMillis()))
            eventSink(AgentEvent.Completed)
            AgentSessionResult.Reply(result, plan, actionResult)
        } catch (t: Throwable) {
            eventSink(AgentEvent.Failed(t.message ?: t::class.simpleName.orEmpty()))
            throw t
        }
    }

    suspend fun resumeApproved(executionId: String, outcome: ActionExecutionOutcome, settings: InferenceSettings, requestedRecentMessages: Int = 24): AgentSessionResult.Reply {
        require(executionId.isNotBlank()) { "executionId must not be blank" }
        require(outcome.verified) { "Approved action result must be verified before resuming the agent" }
        val effectiveSettings = InferenceSettingsStore.current
        eventSink(AgentEvent.Started)
        conversationStore.append(ConversationMessage(UUID.randomUUID().toString(), ConversationMessage.Role.TOOL, outcome.toProtocolResult(executionId), System.currentTimeMillis()))
        return try {
            val result = orchestrator.generate(effectiveSettings, requestedRecentMessages) { token -> eventSink(AgentEvent.Token(token)) }
            conversationStore.append(ConversationMessage(UUID.randomUUID().toString(), ConversationMessage.Role.ASSISTANT, result.text, System.currentTimeMillis()))
            eventSink(AgentEvent.Completed)
            AgentSessionResult.Reply(result, null, outcome.message)
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
    fun toProtocolResult(plan: ActionPlan): String = toProtocolResult(plan.intent.requestId)

    fun toProtocolResult(requestId: String): String = buildString {
        append("{\"version\":1,\"actionId\":\"")
        append(requestId.replace("\\", "\\\\").replace("\"", "\\\""))
        append("\",\"success\":").append(success)
        append(",\"verified\":").append(verified)
        append(",\"data\":")
        append(data?.let { "\"${it.replace("\\", "\\\\").replace("\"", "\\\"")}\"" } ?: "null")
        append(",\"error\":")
        append(errorCode?.let { "\"${it.replace("\\", "\\\\").replace("\"", "\\\"")}\"" } ?: "null")
        append("}")
    }
}
