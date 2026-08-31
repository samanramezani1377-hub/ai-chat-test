package com.woogit.aicore.conversation

/**
 * Persistence boundary for conversation state.
 * Android/Data layers can provide the durable implementation without changing Core.
 */
interface PersistentConversationStore : ConversationStore {
    suspend fun flush()
}

/** Simple durable-implementation boundary for the current JVM module. */
class BufferedConversationStore(
    private val backing: ConversationStore
) : PersistentConversationStore {
    override suspend fun append(message: ConversationMessage) = backing.append(message)
    override suspend fun recent(limit: Int): List<ConversationMessage> = backing.recent(limit)
    override suspend fun summary(): String? = backing.summary()
    override suspend fun replaceSummary(summary: String) = backing.replaceSummary(summary)
    override suspend fun flush() = Unit
}
