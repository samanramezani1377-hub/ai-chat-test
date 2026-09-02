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

/** Model → plan → validate/approve → execute → verify → recover/replan → final answer. */
class AgentSession(
    private val orchestrator: AgentOrchestrator,
    private val conversationStore: ConversationStore,
    private val actionPlanCoordinator: ActionPlanCoordinator? = null,
    private val actionExecutor: (suspend (ActionPlan) -> ActionExecutionOutcome)? = null,
    /** Optional lifecycle-aware retry. It must enforce idempotency/approval rules in the Action Core. */
    private val actionRetryExecutor: (suspend (ActionPlan) -> ActionExecutionOutcome)? = null,
    private val eventSink: suspend (AgentEvent) -> Unit = {},
    private val contextProvider: ContextProvider = DefaultContextProvider(conversationStore, systemContext = { null }, persistentTaskContext = { null }, workspaceContext = { null }),
    /** Optional runtime override. When absent, the supplied InferenceSettings value is used. */
    private val maxActionSteps: Int? = null,
    private val conversationId: String? = null,
) {
    init { require(maxActionSteps == null || maxActionSteps >= 0) { "maxActionSteps must be non-negative" } }

    suspend fun send(content: String, settings: InferenceSettings, requestedRecentMessages: Int? = null): AgentSessionResult.Reply {
        require(content.isNotBlank()) { "content must not be blank" }
        val effectiveSettings = settings
        val effectiveRecentMessages = requestedRecentMessages ?: effectiveSettings.recentMessages
        val effectiveMaxActionSteps = maxActionSteps ?: effectiveSettings.maxActionSteps
        conversationStore.append(ConversationMessage(UUID.randomUUID().toString(), ConversationMessage.Role.USER, content, System.currentTimeMillis()))
        eventSink(AgentEvent.Started)
        return try {
            var result = orchestrator.generate(effectiveSettings, effectiveRecentMessages) { token -> eventSink(AgentEvent.Token(token)) }
            var plan: ActionPlan? = null
            var actionResult: String? = null
            var steps = 0
            var resultPersisted = false
            while (actionPlanCoordinator != null && steps < effectiveMaxActionSteps) {
                conversationStore.append(ConversationMessage(UUID.randomUUID().toString(), ConversationMessage.Role.ASSISTANT, result.text, System.currentTimeMillis()))
                resultPersisted = true
                plan = actionPlanCoordinator.prepare(contextProvider.build(effectiveRecentMessages), conversationId) ?: break
                eventSink(AgentEvent.ActionPrepared(plan.prepared.executionId, plan.prepared.actionId))
                if (plan.prepared.risk == com.woogit.aicore.domain.RiskLevel.SENSITIVE) {
                    eventSink(AgentEvent.ApprovalRequired(plan.prepared.executionId))
                    break
                }
                val executor = actionExecutor ?: break
                var outcome = executor(plan)
                actionResult = outcome.message
                eventSink(AgentEvent.ActionExecuted(plan.prepared.executionId))
                conversationStore.append(ConversationMessage(UUID.randomUUID().toString(), ConversationMessage.Role.TOOL, outcome.toProtocolResult(plan), System.currentTimeMillis()))
                steps++

                if (!outcome.success || !outcome.verified) {
                    // First try a lifecycle-aware retry. If it is unavailable or fails,
                    // regenerate so the model can choose an alternative action/path.
                    val retry = actionRetryExecutor
                    if (retry != null && steps < effectiveMaxActionSteps) {
                        val retried = runCatching { retry(plan) }.getOrElse {
                            ActionExecutionOutcome(false, false, it.message ?: "Retry failed", errorCode = "RETRY_FAILED")
                        }
                        actionResult = retried.message
                        conversationStore.append(ConversationMessage(UUID.randomUUID().toString(), ConversationMessage.Role.TOOL, retried.toProtocolResult(plan), System.currentTimeMillis()))
                        steps++
                        outcome = retried
                    }
                    result = orchestrator.generate(effectiveSettings, effectiveRecentMessages) { token -> eventSink(AgentEvent.Token(token)) }
                    resultPersisted = false
                    if (outcome.success && outcome.verified) continue
                    continue
                }

                result = orchestrator.generate(effectiveSettings, effectiveRecentMessages) { token -> eventSink(AgentEvent.Token(token)) }
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

    suspend fun resumeApproved(executionId: String, outcome: ActionExecutionOutcome, settings: InferenceSettings, requestedRecentMessages: Int? = null): AgentSessionResult.Reply {
        require(executionId.isNotBlank()) { "executionId must not be blank" }
        require(outcome.success && outcome.verified) { "Approved action result must be successful and verified before resuming the agent" }
        val effectiveSettings = settings
        val effectiveRecentMessages = requestedRecentMessages ?: effectiveSettings.recentMessages
        eventSink(AgentEvent.Started)
        conversationStore.append(ConversationMessage(UUID.randomUUID().toString(), ConversationMessage.Role.TOOL, outcome.toProtocolResult(executionId), System.currentTimeMillis()))
        return try {
            val result = orchestrator.generate(effectiveSettings, effectiveRecentMessages) { token -> eventSink(AgentEvent.Token(token)) }
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
