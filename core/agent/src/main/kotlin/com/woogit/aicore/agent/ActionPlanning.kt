package com.woogit.aicore.agent

import com.woogit.aicore.actions.ActionLifecycle
import com.woogit.aicore.actions.PreparedAction
import com.woogit.aicore.conversation.ConversationContext
import com.woogit.aicore.domain.RiskLevel

/** Structured intent produced by an agent. It is data, not permission to execute. */
data class ActionIntent(
    val actionId: String,
    val input: Any,
    val requestedCapability: String? = null,
    val explanation: String? = null
)

data class ActionPlan(
    val intent: ActionIntent,
    val prepared: PreparedAction
)

interface ActionIntentPlanner {
    suspend fun plan(context: ConversationContext): ActionIntent?
}

/**
 * Converts an agent intent into a checkpointed action. It never executes the action.
 */
class ActionPlanCoordinator(
    private val lifecycle: ActionLifecycle,
    private val planner: ActionIntentPlanner
) {
    suspend fun prepare(context: ConversationContext): ActionPlan? {
        val intent = planner.plan(context) ?: return null
        val prepared = lifecycle.prepare(intent.actionId, intent.input)
        lifecycle.validate(prepared, intent.requestedCapability)
        return ActionPlan(intent, prepared.copy(risk = prepared.risk))
    }
}

/** Explicit representation used by UI for approval; no executable callback is exposed. */
data class ApprovalCard(
    val executionId: String,
    val actionId: String,
    val risk: RiskLevel,
    val explanation: String?
)

fun ActionPlan.toApprovalCard(): ApprovalCard = ApprovalCard(
    executionId = prepared.executionId,
    actionId = prepared.actionId,
    risk = prepared.risk,
    explanation = intent.explanation
)
