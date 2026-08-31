package com.woogit.aicore.conversation

/** A message retained as part of the conversation source of truth. */
data class ConversationMessage(
    val id: String,
    val role: Role,
    val content: String,
    val timestampEpochMs: Long
) {
    enum class Role { SYSTEM, USER, ASSISTANT, TOOL }
}

/**
 * Source of truth for conversation history. UI must not own a second copy.
 */
interface ConversationStore {
    suspend fun append(message: ConversationMessage)
    suspend fun recent(limit: Int): List<ConversationMessage>
    suspend fun summary(): String?
    suspend fun replaceSummary(summary: String)
}

class InMemoryConversationStore : ConversationStore {
    private val messages = mutableListOf<ConversationMessage>()
    private var currentSummary: String? = null

    override suspend fun append(message: ConversationMessage) {
        messages += message
    }

    override suspend fun recent(limit: Int): List<ConversationMessage> {
        require(limit >= 0) { "limit must be non-negative" }
        return messages.takeLast(limit)
    }

    override suspend fun summary(): String? = currentSummary

    override suspend fun replaceSummary(summary: String) {
        currentSummary = summary
    }
}
