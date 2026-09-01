package com.woogit.aicore.conversation

/** Persistence boundary for conversation state. */
interface PersistentConversationStore : ConversationStore {
    suspend fun flush()
}

class BufferedConversationStore(private val backing: ConversationStore) : PersistentConversationStore {
    override suspend fun append(message: ConversationMessage) = backing.append(message)
    override suspend fun recent(limit: Int): List<ConversationMessage> = backing.recent(limit)
    override suspend fun summary(): String? = backing.summary()
    override suspend fun replaceSummary(summary: String) = backing.replaceSummary(summary)
    override suspend fun flush() = Unit
}

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

    /** Returns persisted messages for one conversation, in chronological order. */
    suspend fun messages(id: String): List<ConversationMessage>

    /** Persists one message and advances conversation metadata atomically from the repository's perspective. */
    suspend fun append(id: String, message: ConversationMessage, nowEpochMs: Long = System.currentTimeMillis()): Boolean
}

class InMemoryConversationHistoryRepository : ConversationHistoryRepository {
    private val records = LinkedHashMap<String, ConversationRecord>()
    private val messages = LinkedHashMap<String, MutableList<ConversationMessage>>()

    @Synchronized
    override suspend fun recent(limit: Int): List<ConversationRecord> {
        require(limit >= 0) { "limit must be non-negative" }
        return records.values.sortedByDescending { it.updatedAtEpochMs }.take(limit)
    }

    @Synchronized
    override suspend fun create(title: String, nowEpochMs: Long): ConversationRecord {
        val record = ConversationRecord(java.util.UUID.randomUUID().toString(), title.trim().ifEmpty { "گفت‌وگوی جدید" }, nowEpochMs, 0)
        records[record.id] = record
        messages[record.id] = mutableListOf()
        return record
    }

    @Synchronized
    override suspend fun rename(id: String, title: String, nowEpochMs: Long): Boolean {
        val current = records[id] ?: return false
        records[id] = current.copy(title = title.trim().ifEmpty { current.title }, updatedAtEpochMs = nowEpochMs)
        return true
    }

    @Synchronized override suspend fun delete(id: String): ConversationRecord? {
        messages.remove(id)
        return records.remove(id)
    }

    @Synchronized override suspend fun restore(record: ConversationRecord): Boolean {
        if (records.containsKey(record.id)) return false
        records[record.id] = record
        messages.putIfAbsent(record.id, mutableListOf())
        return true
    }

    @Synchronized override suspend fun touch(id: String, messageCount: Int, nowEpochMs: Long): Boolean {
        val current = records[id] ?: return false
        records[id] = current.copy(updatedAtEpochMs = nowEpochMs, messageCount = messageCount)
        return true
    }

    @Synchronized override suspend fun messages(id: String): List<ConversationMessage> = messages[id]?.toList() ?: emptyList()

    @Synchronized override suspend fun append(id: String, message: ConversationMessage, nowEpochMs: Long): Boolean {
        val current = records[id] ?: return false
        val list = messages.getOrPut(id) { mutableListOf() }
        list += message
        records[id] = current.copy(updatedAtEpochMs = nowEpochMs, messageCount = list.size)
        return true
    }
}
