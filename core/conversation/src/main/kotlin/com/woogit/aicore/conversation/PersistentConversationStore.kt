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

/**
 * Conversation identity and metadata used by the application history UI.
 * The store remains the source of truth for messages; this repository owns
 * only conversation-level metadata and ordering.
 */
data class ConversationRecord(
    val id: String,
    val title: String,
    val updatedAtEpochMs: Long,
    val messageCount: Int
)

interface ConversationHistoryRepository {
    suspend fun recent(limit: Int): List<ConversationRecord>
    suspend fun create(title: String, nowEpochMs: Long = System.currentTimeMillis()): ConversationRecord
    suspend fun rename(id: String, title: String, nowEpochMs: Long = System.currentTimeMillis()): Boolean
    suspend fun delete(id: String): ConversationRecord?
    suspend fun restore(record: ConversationRecord): Boolean
    suspend fun touch(id: String, messageCount: Int, nowEpochMs: Long = System.currentTimeMillis()): Boolean
}

/**
 * Thread-safe in-memory implementation for the Core boundary.
 * It deliberately does not pretend to be disk persistence; Android can replace
 * this implementation with a durable repository without changing the UI contract.
 */
class InMemoryConversationHistoryRepository : ConversationHistoryRepository {
    private val records = LinkedHashMap<String, ConversationRecord>()

    @Synchronized
    override suspend fun recent(limit: Int): List<ConversationRecord> {
        require(limit >= 0) { "limit must be non-negative" }
        return records.values.sortedByDescending { it.updatedAtEpochMs }.take(limit)
    }

    @Synchronized
    override suspend fun create(title: String, nowEpochMs: Long): ConversationRecord {
        val record = ConversationRecord(
            id = java.util.UUID.randomUUID().toString(),
            title = title.trim().ifEmpty { "گفت‌وگوی جدید" },
            updatedAtEpochMs = nowEpochMs,
            messageCount = 0
        )
        records[record.id] = record
        return record
    }

    @Synchronized
    override suspend fun rename(id: String, title: String, nowEpochMs: Long): Boolean {
        val current = records[id] ?: return false
        records[id] = current.copy(
            title = title.trim().ifEmpty { current.title },
            updatedAtEpochMs = nowEpochMs
        )
        return true
    }

    @Synchronized
    override suspend fun delete(id: String): ConversationRecord? = records.remove(id)

    @Synchronized
    override suspend fun restore(record: ConversationRecord): Boolean {
        if (records.containsKey(record.id)) return false
        records[record.id] = record
        return true
    }

    @Synchronized
    override suspend fun touch(id: String, messageCount: Int, nowEpochMs: Long): Boolean {
        val current = records[id] ?: return false
        records[id] = current.copy(updatedAtEpochMs = nowEpochMs, messageCount = messageCount)
        return true
    }
}
