package com.samanramezani.aichattest

import android.content.Context
import com.woogit.aicore.conversation.ConversationHistoryRepository
import com.woogit.aicore.conversation.ConversationRecord
import org.json.JSONArray
import org.json.JSONObject

/** Durable Android implementation for conversation metadata. Messages remain owned by Core stores. */
class AndroidConversationHistoryRepository(context: Context) : ConversationHistoryRepository {
    private val preferences = context.getSharedPreferences("conversation_history", Context.MODE_PRIVATE)
    private val lock = Any()

    override suspend fun recent(limit: Int): List<ConversationRecord> = synchronized(lock) {
        require(limit >= 0) { "limit must be non-negative" }
        readAll().sortedByDescending { it.updatedAtEpochMs }.take(limit)
    }

    override suspend fun create(title: String, nowEpochMs: Long): ConversationRecord = synchronized(lock) {
        val record = ConversationRecord(
            id = java.util.UUID.randomUUID().toString(),
            title = title.trim().ifEmpty { "گفت‌وگوی جدید" },
            updatedAtEpochMs = nowEpochMs,
            messageCount = 0,
        )
        writeAll(readAll() + record)
        record
    }

    override suspend fun rename(id: String, title: String, nowEpochMs: Long): Boolean = synchronized(lock) {
        val current = readAll()
        val index = current.indexOfFirst { it.id == id }
        if (index < 0) return@synchronized false
        val old = current[index]
        val updated = old.copy(
            title = title.trim().ifEmpty { old.title },
            updatedAtEpochMs = nowEpochMs,
        )
        writeAll(current.toMutableList().also { it[index] = updated })
        true
    }

    override suspend fun delete(id: String): ConversationRecord? = synchronized(lock) {
        val current = readAll()
        val deleted = current.firstOrNull { it.id == id } ?: return@synchronized null
        writeAll(current.filterNot { it.id == id })
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
        val updated = current[index].copy(updatedAtEpochMs = nowEpochMs, messageCount = messageCount)
        writeAll(current.toMutableList().also { it[index] = updated })
        true
    }

    private fun readAll(): List<ConversationRecord> {
        val raw = preferences.getString(KEY_RECORDS, null) ?: return emptyList()
        return runCatching {
            val array = JSONArray(raw)
            buildList(array.length()) {
                for (index in 0 until array.length()) {
                    val item = array.getJSONObject(index)
                    add(
                        ConversationRecord(
                            id = item.getString("id"),
                            title = item.getString("title"),
                            updatedAtEpochMs = item.getLong("updatedAtEpochMs"),
                            messageCount = item.getInt("messageCount"),
                        )
                    )
                }
            }
        }.getOrDefault(emptyList())
    }

    private fun writeAll(records: List<ConversationRecord>) {
        val array = JSONArray()
        records.forEach { record ->
            array.put(
                JSONObject()
                    .put("id", record.id)
                    .put("title", record.title)
                    .put("updatedAtEpochMs", record.updatedAtEpochMs)
                    .put("messageCount", record.messageCount)
            )
        }
        preferences.edit().putString(KEY_RECORDS, array.toString()).apply()
    }

    private companion object {
        const val KEY_RECORDS = "records"
    }
}
