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
    /** Permanently removes retained data for a conversation after the undo window expires. */
    suspend fun purge(id: String): Boolean
    suspend fun touch(id: String, messageCount: Int, nowEpochMs: Long = System.currentTimeMillis()): Boolean
    suspend fun messages(id: String): List<ConversationMessage>
    suspend fun append(id: String, message: ConversationMessage, nowEpochMs: Long = System.currentTimeMillis()): Boolean
}

class InMemoryConversationHistoryRepository : ConversationHistoryRepository {
    private val lock = Any()
    private val records = LinkedHashMap<String, ConversationRecord>()
    private val messages = LinkedHashMap<String, MutableList<ConversationMessage>>()
    private val deletedMessages = LinkedHashMap<String, List<ConversationMessage>>()

    override suspend fun recent(limit: Int): List<ConversationRecord> = synchronized(lock) {
        require(limit >= 0) { "limit must be non-negative" }
        records.values.sortedByDescending { it.updatedAtEpochMs }.take(limit)
    }

    override suspend fun create(title: String, nowEpochMs: Long): ConversationRecord = synchronized(lock) {
        val record = ConversationRecord(java.util.UUID.randomUUID().toString(), title.trim().ifEmpty { "گفت‌وگوی جدید" }, nowEpochMs, 0)
        records[record.id] = record
        messages[record.id] = mutableListOf()
        record
    }

    override suspend fun rename(id: String, title: String, nowEpochMs: Long): Boolean = synchronized(lock) {
        val current = records[id] ?: return@synchronized false
        records[id] = current.copy(title = title.trim().ifEmpty { current.title }, updatedAtEpochMs = nowEpochMs)
        true
    }

    override suspend fun delete(id: String): ConversationRecord? = synchronized(lock) {
        val record = records.remove(id) ?: return@synchronized null
        deletedMessages[id] = messages.remove(id)?.toList() ?: emptyList()
        record
    }

    override suspend fun restore(record: ConversationRecord): Boolean = synchronized(lock) {
        if (records.containsKey(record.id)) return@synchronized false
        records[record.id] = record
        messages[record.id] = deletedMessages.remove(record.id)?.toMutableList() ?: mutableListOf()
        true
    }

    override suspend fun purge(id: String): Boolean = synchronized(lock) {
        deletedMessages.remove(id) != null
    }

    override suspend fun touch(id: String, messageCount: Int, nowEpochMs: Long): Boolean = synchronized(lock) {
        val current = records[id] ?: return@synchronized false
        records[id] = current.copy(updatedAtEpochMs = nowEpochMs, messageCount = messageCount)
        true
    }

    override suspend fun messages(id: String): List<ConversationMessage> = synchronized(lock) {
        messages[id]?.toList() ?: emptyList()
    }

    override suspend fun append(id: String, message: ConversationMessage, nowEpochMs: Long): Boolean = synchronized(lock) {
        val current = records[id] ?: return@synchronized false
        val list = messages.getOrPut(id) { mutableListOf() }
        list += message
        records[id] = current.copy(updatedAtEpochMs = nowEpochMs, messageCount = list.size)
        true
    }
}
