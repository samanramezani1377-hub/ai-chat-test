package com.samanramezani.aichattest

import android.content.Context
import com.woogit.aicore.actions.ActionCheckpointStore
import com.woogit.aicore.actions.ActionExecutionState
import com.woogit.aicore.actions.PreparedAction
import com.woogit.aicore.domain.RiskLevel
import com.woogit.aicore.domain.VerificationResult
import org.json.JSONObject

/** Durable checkpoint store for prepared/approved/executing action state. */
class AndroidActionCheckpointStore(context: Context) : ActionCheckpointStore {
    private val preferences = context.applicationContext.getSharedPreferences("action_checkpoints", Context.MODE_PRIVATE)

    override suspend fun save(action: PreparedAction) {
        val json = JSONObject()
            .put("executionId", action.executionId)
            .put("actionId", action.actionId)
            .put("input", action.input.toString())
            .put("risk", action.risk.name)
            .put("retryCount", action.retryCount)
            .put("conversationId", action.conversationId)
            .put("state", encodeState(action.state))
        check(preferences.edit().putString(action.executionId, json.toString()).commit()) { "Unable to persist action checkpoint" }
    }

    override suspend fun get(executionId: String): PreparedAction? = decode(preferences.getString(executionId, null))

    override suspend fun list(): List<PreparedAction> = preferences.all.values
        .asSequence()
        .filterIsInstance<String>()
        .mapNotNull(::decode)
        .toList()

    private fun decode(raw: String?): PreparedAction? = runCatching {
        if (raw == null) return null
        val json = JSONObject(raw)
        PreparedAction(
            executionId = json.getString("executionId"),
            actionId = json.getString("actionId"),
            input = json.getString("input"),
            risk = RiskLevel.valueOf(json.getString("risk")),
            state = decodeState(json.getJSONObject("state")),
            retryCount = json.optInt("retryCount", 0).coerceAtLeast(0),
            conversationId = json.optString("conversationId").ifBlank { null },
        )
    }.getOrNull()

    private fun encodeState(state: ActionExecutionState): JSONObject = when (state) {
        ActionExecutionState.Prepared -> JSONObject().put("type", "prepared")
        ActionExecutionState.AwaitingApproval -> JSONObject().put("type", "awaiting_approval")
        ActionExecutionState.Approved -> JSONObject().put("type", "approved")
        ActionExecutionState.Executing -> JSONObject().put("type", "executing")
        ActionExecutionState.Rejected -> JSONObject().put("type", "rejected")
        is ActionExecutionState.Failed -> JSONObject().put("type", "failed").put("message", state.message)
        is ActionExecutionState.Completed -> JSONObject().put("type", "completed").put("success", state.verification.success).put("evidence", state.verification.evidence)
    }

    private fun decodeState(json: JSONObject): ActionExecutionState = when (json.getString("type")) {
        "prepared" -> ActionExecutionState.Prepared
        "awaiting_approval" -> ActionExecutionState.AwaitingApproval
        "approved" -> ActionExecutionState.Approved
        "executing" -> ActionExecutionState.Executing
        "rejected" -> ActionExecutionState.Rejected
        "failed" -> ActionExecutionState.Failed(json.optString("message", "Action failed"))
        "completed" -> ActionExecutionState.Completed(VerificationResult(json.optBoolean("success", false), json.optString("evidence").ifBlank { null }))
        else -> error("Unknown action checkpoint state")
    }
}
