package com.samanramezani.aichattest.conversation

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject

/** Durable conversation history owned by the app data layer, not by Compose UI. */
class AndroidConversationHistoryStore(context: Context) {
    data class Conversation(
        val id: String,
        val title: String,
        val updatedAtEpochMs: Long,
        val messages: List<Message>,
    )

    data class Message(
        val role: String,
        val content: String,
        val timestampEpochMs: Long,
    )

    private val preferences = context.getSharedPreferences(FILE_NAME, Context.MODE_PRIVATE)

    @Synchronized
    fun list(limit: Int = 5): List<Conversation> {
        require(limit >= 0)
        return readAll().sortedByDescending { it.updatedAtEpochMs }.take(limit)
    }

    @Synchronized
    fun get(id: String): Conversation? = readAll().firstOrNull { it.id == id }

    @Synchronized
    fun save(conversation: Conversation) {
        val all = readAll().filterNot { it.id == conversation.id } + conversation
        writeAll(all)
    }

    @Synchronized
    fun delete(id: String): Conversation? {
        val all = readAll()
        val deleted = all.firstOrNull { it.id == id }
        if (deleted != null) writeAll(all.filterNot { it.id == id })
        return deleted
    }

    @Synchronized
    fun rename(id: String, title: String): Boolean {
        val trimmed = title.trim()
        if (trimmed.isEmpty()) return false
        val all = readAll()
        var changed = false
        val updated = all.map {
            if (it.id == id) {
                changed = true
                it.copy(title = trimmed, updatedAtEpochMs = System.currentTimeMillis())
            } else it
        }
        if (changed) writeAll(updated)
        return changed
    }

    private fun readAll(): List<Conversation> {
        val raw = preferences.getString(KEY, null) ?: return emptyList()
        return runCatching {
            val array = JSONArray(raw)
            buildList(array.length()) {
                for (i in 0 until array.length()) add(decode(array.getJSONObject(i)))
            }
        }.getOrDefault(emptyList())
    }

    private fun writeAll(conversations: List<Conversation>) {
        val array = JSONArray()
        conversations.forEach { array.put(encode(it)) }
        preferences.edit().putString(KEY, array.toString()).apply()
    }

    private fun encode(conversation: Conversation): JSONObject = JSONObject().apply {
        put("id", conversation.id)
        put("title", conversation.title)
        put("updatedAt", conversation.updatedAtEpochMs)
        put("messages", JSONArray().also { messages ->
            conversation.messages.forEach { message ->
                messages.put(JSONObject().apply {
                    put("role", message.role)
                    put("content", message.content)
                    put("timestamp", message.timestampEpochMs)
                })
            }
        })
    }

    private fun decode(value: JSONObject): Conversation {
        val messages = value.optJSONArray("messages") ?: JSONArray()
        return Conversation(
            id = value.getString("id"),
            title = value.optString("title", "گفت‌وگوی جدید"),
            updatedAtEpochMs = value.optLong("updatedAt", 0L),
            messages = buildList(messages.length()) {
                for (i in 0 until messages.length()) {
                    val item = messages.getJSONObject(i)
                    add(Message(item.optString("role", "USER"), item.optString("content"), item.optLong("timestamp", 0L)))
                }
            },
        )
    }

    private companion object {
        const val FILE_NAME = "conversation_history"
        const val KEY = "conversations"
    }
}
