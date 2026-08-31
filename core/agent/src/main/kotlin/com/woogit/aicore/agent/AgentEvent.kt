package com.woogit.aicore.agent

sealed interface AgentEvent {
    data object Started : AgentEvent
    data class Token(val value: String) : AgentEvent
    data class ActionPrepared(val executionId: String, val actionId: String) : AgentEvent
    data class ApprovalRequired(val executionId: String) : AgentEvent
    data class ActionExecuted(val executionId: String) : AgentEvent
    data class Failed(val message: String) : AgentEvent
    data object Completed : AgentEvent
}
