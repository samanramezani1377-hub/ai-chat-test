package com.woogit.aicore.agent

import com.woogit.aicore.actions.ActionLifecycle
import com.woogit.aicore.actions.PreparedAction
import com.woogit.aicore.conversation.ConversationContext
import com.woogit.aicore.domain.RiskLevel

/** Structured intent is never an execution authorization. */
data class ActionIntent(
    val actionId: String,
    val input: Any,
    val requestedCapability: String? = null,
    val explanation: String? = null,
    /** Protocol-level request identifier, separate from the registered action name. */
    val requestId: String = actionId,
)

data class ActionPlan(
    val intent: ActionIntent,
    val prepared: PreparedAction
) {
    fun approvalCard(): ApprovalCard = ApprovalCard(
        executionId = prepared.executionId,
        actionId = prepared.actionId,
        risk = prepared.risk,
        explanation = intent.explanation
    )
}

interface ActionIntentPlanner {
    suspend fun plan(context: ConversationContext): ActionIntent?
}

class ActionPlanCoordinator(
    private val lifecycle: ActionLifecycle,
    private val planner: ActionIntentPlanner
) {
    suspend fun prepare(context: ConversationContext): ActionPlan? {
        val intent = planner.plan(context) ?: return null
        val prepared = lifecycle.prepare(intent.actionId, intent.input)
        val validated = lifecycle.validate(prepared, intent.requestedCapability)
        return ActionPlan(intent, validated)
    }
}

data class ApprovalCard(
    val executionId: String,
    val actionId: String,
    val risk: RiskLevel,
    val explanation: String?
)
