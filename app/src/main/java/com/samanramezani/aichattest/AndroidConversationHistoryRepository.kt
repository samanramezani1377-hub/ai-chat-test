package com.samanramezani.aichattest

import android.content.Context
import com.woogit.aicore.conversation.ConversationHistoryRepository
import com.woogit.aicore.conversation.ConversationRecord
import org.json.JSONArray
import org.json.JSONObject

/** Durable Android implementation for conversation metadata and message snapshots. */
class AndroidConversationHistoryRepository(context: Context) : ConversationHistoryRepository {
    private val preferences = context.getSharedPreferences("conversation_history", Context.MODE_PRIVATE)
    private val lock = Any()

    override suspend fun recent(limit: Int): List<ConversationRecord> = synchronized(lock) {
        require(limit >= 0) { "limit must be non-negative" }
        readAll().sortedByDescending { it.updatedAtEpochMs }.take(limit)
    }

    override suspend fun create(title: String, nowEpochMs: Long): ConversationRecord = synchronized(lock) {
        val record = ConversationRecord(java.util.UUID.randomUUID().toString(), title.trim().ifEmpty { "گفت‌وگوی جدید" }, nowEpochMs, 0)
        writeAll(readAll() + record)
        writeMessages(record.id, emptyList())
        record
    }

    override suspend fun rename(id: String, title: String, nowEpochMs: Long): Boolean = synchronized(lock) {
        val current = readAll()
        val index = current.indexOfFirst { it.id == id }
        if (index < 0) return@synchronized false
        val old = current[index]
        writeAll(current.toMutableList().also { it[index] = old.copy(title = title.trim().ifEmpty { old.title }, updatedAtEpochMs = nowEpochMs) })
        true
    }

    override suspend fun delete(id: String): ConversationRecord? = synchronized(lock) {
        val current = readAll()
        val deleted = current.firstOrNull { it.id == id } ?: return@synchronized null
        writeAll(current.filterNot { it.id == id })
        preferences.edit().remove(messagesKey(id)).apply()
        deleted
    }

    override suspend fun restore(record: ConversationRecord): Boolean = synchronized(lock) {
        val current = readAll()
        if (current.any { it.id == record.id }) return@synchronized false
        writeAll(current + record)
        true
    }

    override suspend fun touch(id: String, messageCount: Int, nowEpochMs: Long): Boolean = synchronized(lock) {
        val current = readAll()
        val index = current.indexOfFirst { it.id == id }
        if (index < 0) return@synchronized false
        writeAll(current.toMutableList().also { it[index] = current[index].copy(updatedAtEpochMs = nowEpochMs, messageCount = messageCount) })
        true
    }

    fun messages(id: String): List<StoredMessage> = synchronized(lock) { readMessages(id) }

    fun replaceMessages(id: String, messages: List<StoredMessage>) = synchronized(lock) {
        if (readAll().none { it.id == id }) return@synchronized
        writeMessages(id, messages)
    }

    private fun readAll(): List<ConversationRecord> {
        val raw = preferences.getString(KEY_RECORDS, null) ?: return emptyList()
        return runCatching {
            val array = JSONArray(raw)
            buildList(array.length()) {
                for (index in 0 until array.length()) {
                    val item = array.getJSONObject(index)
                    add(ConversationRecord(item.getString("id"), item.getString("title"), item.getLong("updatedAtEpochMs"), item.getInt("messageCount")))
                }
            }
        }.getOrDefault(emptyList())
    }

    private fun writeAll(records: List<ConversationRecord>) {
        val array = JSONArray()
        records.forEach { record -> array.put(JSONObject().put("id", record.id).put("title", record.title).put("updatedAtEpochMs", record.updatedAtEpochMs).put("messageCount", record.messageCount)) }
        preferences.edit().putString(KEY_RECORDS, array.toString()).apply()
    }

    private fun messagesKey(id: String) = "messages_$id"

    private fun readMessages(id: String): List<StoredMessage> {
        val raw = preferences.getString(messagesKey(id), null) ?: return emptyList()
        return runCatching {
            val array = JSONArray(raw)
            buildList(array.length()) {
                for (index in 0 until array.length()) {
                    val item = array.getJSONObject(index)
                    val role = runCatching { StoredMessage.Role.valueOf(item.getString("role")) }.getOrNull() ?: continue
                    add(StoredMessage(item.getString("id"), role, item.getString("content"), item.getLong("timestampEpochMs")))
                }
            }
        }.getOrDefault(emptyList())
    }

    private fun writeMessages(id: String, messages: List<StoredMessage>) {
        val array = JSONArray()
        messages.forEach { message -> array.put(JSONObject().put("id", message.id).put("role", message.role.name).put("content", message.content).put("timestampEpochMs", message.timestampEpochMs)) }
        preferences.edit().putString(messagesKey(id), array.toString()).apply()
    }

    private companion object { const val KEY_RECORDS = "records" }
}

data class StoredMessage(
    val id: String,
    val role: Role,
    val content: String,
    val timestampEpochMs: Long,
) { enum class Role { SYSTEM, USER, ASSISTANT, TOOL } }
