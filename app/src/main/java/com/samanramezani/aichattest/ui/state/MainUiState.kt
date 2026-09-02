package com.samanramezani.aichattest.ui.state

import com.woogit.aicore.domain.ChatMessage
import com.woogit.aicore.conversation.ConversationMessage
import com.woogit.aicore.conversation.ConversationRecord

internal data class UiMessage(
    val role: ChatMessage.Role,
    val text: String,
    val id: String,
) {
    constructor(role: ConversationMessage.Role, text: String, id: String) : this(
        role = when (role) {
            ConversationMessage.Role.SYSTEM -> ChatMessage.Role.SYSTEM
            ConversationMessage.Role.USER -> ChatMessage.Role.USER
            ConversationMessage.Role.ASSISTANT -> ChatMessage.Role.ASSISTANT
            ConversationMessage.Role.TOOL -> ChatMessage.Role.TOOL
        },
        text = text,
        id = id,
    )
}

internal data class DeletedConversation(
    val record: ConversationRecord,
)

internal data class ExecutionState(
    val id: String,
    val action: String,
    val status: String,
    val startedAt: Long,
    val finishedAt: Long? = null,
    val error: String? = null,
    val requestPreview: String = "",
    val resultPreview: String? = null,
    val approvalRequired: Boolean = false,
)
