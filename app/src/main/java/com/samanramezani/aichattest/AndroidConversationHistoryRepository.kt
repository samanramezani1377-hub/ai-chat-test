package com.samanramezani.aichattest

import android.content.Context
import com.woogit.aicore.conversation.ConversationHistoryRepository
import com.woogit.aicore.conversation.ConversationMessage
import com.woogit.aicore.conversation.ConversationRecord
import org.json.JSONArray
import org.json.JSONObject

/** Durable Android implementation for conversation metadata and messages. */
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
        val current = readAll(); val index = current.indexOfFirst { it.id == id }
        if (index < 0) return@synchronized false
        writeAll(current.toMutableList().also { it[index] = current[index].copy(title = title.trim().ifEmpty { current[index].title }, updatedAtEpochMs = nowEpochMs) })
        true
    }

    override suspend fun delete(id: String): ConversationRecord? = synchronized(lock) {
        val current = readAll(); val deleted = current.firstOrNull { it.id == id } ?: return@synchronized null
        writeAll(current.filterNot { it.id == id }); preferences.edit().remove(messagesKey(id)).apply(); deleted
    }

    override suspend fun restore(record: ConversationRecord): Boolean = synchronized(lock) {
        val current = readAll(); if (current.any { it.id == record.id }) return@synchronized false
        writeAll(current + record); true
    }

    override suspend fun touch(id: String, messageCount: Int, nowEpochMs: Long): Boolean = synchronized(lock) {
        val current = readAll(); val index = current.indexOfFirst { it.id == id }
        if (index < 0) return@synchronized false
        writeAll(current.toMutableList().also { it[index] = current[index].copy(updatedAtEpochMs = nowEpochMs, messageCount = messageCount) }); true
    }

    override suspend fun messages(id: String): List<ConversationMessage> = synchronized(lock) { readMessages(id) }

    override suspend fun append(id: String, message: ConversationMessage, nowEpochMs: Long): Boolean = synchronized(lock) {
        val current = readAll(); val index = current.indexOfFirst { it.id == id }
        if (index < 0) return@synchronized false
        val updatedMessages = readMessages(id).toMutableList().also { it += message }
        writeMessages(id, updatedMessages)
        writeAll(current.toMutableList().also { it[index] = current[index].copy(updatedAtEpochMs = nowEpochMs, messageCount = updatedMessages.size) })
        true
    }

    private fun readAll(): List<ConversationRecord> {
        val raw = preferences.getString(KEY_RECORDS, null) ?: return emptyList()
        return runCatching {
            val array = JSONArray(raw)
            buildList(array.length()) {
                for (i in 0 until array.length()) {
                    val item = array.getJSONObject(i)
                    add(ConversationRecord(item.getString("id"), item.getString("title"), item.getLong("updatedAtEpochMs"), item.getInt("messageCount")))
                }
            }
        }.getOrDefault(emptyList())
    }

    private fun writeAll(records: List<ConversationRecord>) {
        val array = JSONArray()
        records.forEach { r -> array.put(JSONObject().put("id", r.id).put("title", r.title).put("updatedAtEpochMs", r.updatedAtEpochMs).put("messageCount", r.messageCount)) }
        preferences.edit().putString(KEY_RECORDS, array.toString()).apply()
    }

    private fun messagesKey(id: String) = "messages_$id"

    private fun readMessages(id: String): List<ConversationMessage> {
        val raw = preferences.getString(messagesKey(id), null) ?: return emptyList()
        return runCatching {
            val array = JSONArray(raw)
            buildList(array.length()) {
                for (i in 0 until array.length()) {
                    val item = array.getJSONObject(i)
                    val role = runCatching { ConversationMessage.Role.valueOf(item.getString("role")) }.getOrNull() ?: continue
                    add(ConversationMessage(item.getString("id"), role, item.getString("content"), item.getLong("timestampEpochMs")))
                }
            }
        }.getOrDefault(emptyList())
    }

    private fun writeMessages(id: String, messages: List<ConversationMessage>) {
        val array = JSONArray()
        messages.forEach { m -> array.put(JSONObject().put("id", m.id).put("role", m.role.name).put("content", m.content).put("timestampEpochMs", m.timestampEpochMs)) }
        preferences.edit().putString(messagesKey(id), array.toString()).apply()
    }

    private companion object { const val KEY_RECORDS = "records" }
}
