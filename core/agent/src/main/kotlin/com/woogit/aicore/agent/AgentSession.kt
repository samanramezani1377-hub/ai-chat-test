package com.woogit.aicore.agent

import com.woogit.aicore.conversation.ConversationMessage
import com.woogit.aicore.conversation.ContextProvider
import com.woogit.aicore.conversation.ConversationStore
import com.woogit.aicore.conversation.DefaultContextProvider
import com.woogit.aicore.domain.GenerationResult
import com.woogit.aicore.domain.InferenceSettings
import java.util.UUID
import kotlinx.coroutines.CancellationException

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
            // Honor an explicit recent-message limit (e.g. real-time voice); callers that do not
            // specify one retain the complete history for normal chat/agent workflows.
            val initialContext = conversationStore.recent(
                if (requestedRecentMessages == null) Int.MAX_VALUE else effectiveRecentMessages
            )
            var result = generateTurn(effectiveSettings, effectiveRecentMessages, initialContext)
            var plan: ActionPlan? = null
            var actionResult: String? = null
            var steps = 0
            var repairAttempts = 0
            var resultPersisted = false
            while (actionPlanCoordinator != null && steps < effectiveMaxActionSteps) {
                conversationStore.append(ConversationMessage(UUID.randomUUID().toString(), ConversationMessage.Role.ASSISTANT, result.text, System.currentTimeMillis()))
                resultPersisted = true
                val context = contextProvider.build(Int.MAX_VALUE)
                val nextPlan = try {
                    actionPlanCoordinator.prepare(context, conversationId)
                } catch (cancelled: CancellationException) {
                    throw cancelled
                } catch (_: Exception) {
                    null
                }
                if (nextPlan == null) {
                    if (repairAttempts == 0 && actionPlanCoordinator.requiresAction(context)) {
                        repairAttempts++
                        conversationStore.append(ConversationMessage(
                            UUID.randomUUID().toString(),
                            ConversationMessage.Role.TOOL,
                            "ACTION_PROTOCOL_REPAIR: Previous output was not a valid ActionRequest v1. Do not output Bash, shell commands, prose, or Markdown. Return only one JSON object for a registered action, with version, actionId, action, and arguments. Never claim success without a verified tool result.",
                            System.currentTimeMillis()
                        ))
                        result = generateTurn(effectiveSettings, effectiveRecentMessages, conversationStore.recent(Int.MAX_VALUE))
                        resultPersisted = false
                        continue
                    }
                    if (repairAttempts > 0 && actionPlanCoordinator.requiresAction(context)) {
                        result = result.copy(text = "نتوانستم درخواست اجرایی را به فراخوانی ابزار معتبر تبدیل کنم؛ هیچ عملیاتی اجرا نشد. لطفاً درخواست را دوباره با نام و محتوای دقیق فایل ارسال کن.")
                        eventSink(AgentEvent.Token(result.text))
                        resultPersisted = false
                    }
                    break
                }
                plan = nextPlan
                eventSink(AgentEvent.ActionPrepared(nextPlan.prepared.executionId, nextPlan.prepared.actionId))
                if (nextPlan.prepared.risk == com.woogit.aicore.domain.RiskLevel.SENSITIVE) {
                    eventSink(AgentEvent.ApprovalRequired(nextPlan.prepared.executionId))
                    break
                }
                val executor = actionExecutor
                if (executor == null) {
                    result = result.copy(text = "ابزار آمادهٔ اجرا نیست؛ هیچ عملیاتی انجام نشد.")
                    eventSink(AgentEvent.Token(result.text))
                    resultPersisted = false
                    break
                }
                var outcome = executor(nextPlan)
                actionResult = outcome.message
                eventSink(AgentEvent.ActionExecuted(nextPlan.prepared.executionId))
                conversationStore.append(ConversationMessage(UUID.randomUUID().toString(), ConversationMessage.Role.TOOL, outcome.toProtocolResult(nextPlan), System.currentTimeMillis()))
                steps++

                if (!outcome.success || !outcome.verified) {
                    // First try a lifecycle-aware retry. If it is unavailable or fails,
                    // regenerate so the model can choose an alternative action/path.
                    val retry = actionRetryExecutor
                    if (retry != null && steps < effectiveMaxActionSteps) {
                        val retried = runCatching { retry(nextPlan) }.getOrElse {
                            ActionExecutionOutcome(false, false, it.message ?: "Retry failed", errorCode = "RETRY_FAILED")
                        }
                        actionResult = retried.message
                        conversationStore.append(ConversationMessage(UUID.randomUUID().toString(), ConversationMessage.Role.TOOL, retried.toProtocolResult(nextPlan), System.currentTimeMillis()))
                        steps++
                        outcome = retried
                    }
                    val retryContext = conversationStore.recent(Int.MAX_VALUE)
                    result = generateTurn(effectiveSettings, effectiveRecentMessages, retryContext)
                    resultPersisted = false
                    if (outcome.success && outcome.verified) continue
                    continue
                }

                val nextContext = conversationStore.recent(Int.MAX_VALUE)
                result = generateTurn(effectiveSettings, effectiveRecentMessages, nextContext)
                resultPersisted = false
            }
            if (!resultPersisted) conversationStore.append(ConversationMessage(UUID.randomUUID().toString(), ConversationMessage.Role.ASSISTANT, result.text, System.currentTimeMillis()))
            eventSink(AgentEvent.Completed)
            val userFacingResult = if (plan?.prepared?.risk == com.woogit.aicore.domain.RiskLevel.SENSITIVE) {
                result.copy(text = "برای اجرای عملیات «${plan.intent.actionId}» تأیید شما لازم است.")
            } else result
            AgentSessionResult.Reply(userFacingResult, plan, actionResult)
        } catch (t: Throwable) {
            eventSink(AgentEvent.Failed(t.message ?: t::class.simpleName.orEmpty()))
            throw t
        }
    }


    /**
     * Buffer native output until it is known whether the model produced a tool call.
     * This prevents internal reasoning and raw tool JSON from leaking into chat.
     */
    private suspend fun generateTurn(
        settings: InferenceSettings,
        recentMessages: Int,
        contextMessages: List<ConversationMessage>,
    ): GenerationResult {
        val result = orchestrator.generate(settings, recentMessages, { /* published after classification */ }, contextMessages)
        val cleaned = stripReasoning(result.text)
        val context = contextProvider.build(Int.MAX_VALUE)
        val actionRequired = actionPlanCoordinator?.requiresAction(context) == true
        val toolCall = ActionProtocolParser().parse(cleaned) != null
        if (!toolCall && !actionRequired && cleaned.isNotBlank()) eventSink(AgentEvent.Token(cleaned))
        return result.copy(text = cleaned)
    }

    private fun stripReasoning(raw: String): String {
        var text = raw.trim()
        val end = if (text.startsWith("{") || text.startsWith("```")) -1 else text.indexOf("</think>")
        if (end >= 0) text = text.substring(end + "</think>".length).trimStart()
        if (text.startsWith("<think>")) {
            val closing = text.indexOf("</think>")
            if (closing >= 0) text = text.substring(closing + "</think>".length).trimStart()
            else return ""
        }
        return text.trim()
    }

    suspend fun resumeApproved(executionId: String, outcome: ActionExecutionOutcome, settings: InferenceSettings, requestedRecentMessages: Int? = null): AgentSessionResult.Reply {
        require(executionId.isNotBlank()) { "executionId must not be blank" }
        require(outcome.success && outcome.verified) { "Approved action result must be successful and verified before resuming the agent" }
        val effectiveSettings = settings
        val effectiveRecentMessages = requestedRecentMessages ?: effectiveSettings.recentMessages
        eventSink(AgentEvent.Started)
        conversationStore.append(ConversationMessage(UUID.randomUUID().toString(), ConversationMessage.Role.TOOL, outcome.toProtocolResult(executionId), System.currentTimeMillis()))
        return try {
            val resumeContext = conversationStore.recent(Int.MAX_VALUE)
            val result = generateTurn(effectiveSettings, effectiveRecentMessages, resumeContext)
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
