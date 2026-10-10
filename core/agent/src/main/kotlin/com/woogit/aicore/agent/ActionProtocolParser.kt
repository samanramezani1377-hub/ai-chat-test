package com.woogit.aicore.agent

/** Parses only a complete ActionRequest v1 JSON object; surrounding prose is rejected. */
class ActionProtocolParser {
    fun parse(text: String): ActionIntent? {
        val trimmed = text.trim()
        if (trimmed.isEmpty()) return null

        // Qwen-style local models can emit their reasoning before </think> because
        // the runtime opens the reasoning section in the generation prompt. Only
        // discard that explicitly delimited prefix; arbitrary prose around a tool
        // request remains invalid and must never trigger execution.
        val json = if (trimmed.startsWith('{')) {
            trimmed
        } else {
            val reasoningEnd = trimmed.lastIndexOf("</think>")
            if (reasoningEnd < 0) return null
            trimmed.substring(reasoningEnd + "</think>".length).trim()
        }
        if (json.isEmpty() || json.first() != '{') return null
        val end = findObjectEnd(json, 0) ?: return null
        if (end != json.lastIndex) return null

        val version = stringOrNumber(json, "version")?.toIntOrNull() ?: return null
        if (version != 1) return null
        val action = string(json, "action") ?: return null
        val requestId = string(json, "actionId") ?: return null
        if (action.isBlank() || requestId.isBlank()) return null

        val arguments = objectValue(json, "arguments") ?: return null
        return ActionIntent(
            actionId = action,
            requestId = requestId,
            input = arguments,
            explanation = "ActionRequest v1: $action",
        )
    }

    private fun string(json: String, key: String): String? {
        val marker = Regex("\\\"${Regex.escape(key)}\\\"\\s*:").find(json) ?: return null
        var index = marker.range.last + 1
        while (index < json.length && json[index].isWhitespace()) index++
        if (index >= json.length || json[index] != '\"') return null
        val end = findStringEnd(json, index) ?: return null
        return decodeJsonString(json.substring(index + 1, end))
    }

    private fun stringOrNumber(json: String, key: String): String? {
        string(json, key)?.let { return it }

        val marker = Regex("\\\"${Regex.escape(key)}\\\"\\s*:").find(json) ?: return null
        var index = marker.range.last + 1
        while (index < json.length && json[index].isWhitespace()) index++
        val start = index
        while (index < json.length && json[index].isDigit()) index++
        if (index == start) return null
        return json.substring(start, index)
    }

    private fun objectValue(json: String, key: String): String? {
        val marker = Regex("\\\"${Regex.escape(key)}\\\"\\s*:").find(json) ?: return null
        var start = marker.range.last + 1
        while (start < json.length && json[start].isWhitespace()) start++
        if (start >= json.length || json[start] != '{') return null
        val end = findObjectEnd(json, start) ?: return null
        return json.substring(start, end + 1)
    }

    private fun findObjectEnd(text: String, start: Int): Int? {
        if (start >= text.length || text[start] != '{') return null
        var depth = 0
        var quoted = false
        var escaped = false
        for (i in start until text.length) {
            val c = text[i]
            if (quoted) {
                if (escaped) escaped = false
                else if (c == '\\') escaped = true
                else if (c == '"') quoted = false
                continue
            }
            when (c) {
                '"' -> quoted = true
                '{' -> depth++
                '}' -> {
                    depth--
                    if (depth == 0) return i
                    if (depth < 0) return null
                }
            }
        }
        return null
    }

    private fun findStringEnd(text: String, start: Int): Int? {
        var escaped = false
        for (i in start + 1 until text.length) {
            val c = text[i]
            if (escaped) escaped = false
            else if (c == '\\') escaped = true
            else if (c == '"') return i
        }
        return null
    }

    private fun decodeJsonString(value: String): String? {
        val out = StringBuilder(value.length)
        var i = 0
        while (i < value.length) {
            val c = value[i]
            if (c != '\\') {
                out.append(c)
                i++
                continue
            }
            if (++i >= value.length) return null
            when (val escaped = value[i]) {
                '"', '\\', '/' -> out.append(escaped)
                'b' -> out.append('\b')
                'f' -> out.append('\u000C')
                'n' -> out.append('\n')
                'r' -> out.append('\r')
                't' -> out.append('\t')
                'u' -> {
                    if (i + 4 >= value.length) return null
                    val hex = value.substring(i + 1, i + 5)
                    val code = hex.toIntOrNull(16) ?: return null
                    out.append(code.toChar())
                    i += 4
                }
                else -> return null
            }
            i++
        }
        return out.toString()
    }
}

class ProtocolActionIntentPlanner(
    private val parser: ActionProtocolParser = ActionProtocolParser()
) : ActionIntentPlanner {
    override suspend fun plan(context: com.woogit.aicore.conversation.ConversationContext): ActionIntent? {
        val assistant = context.recentMessages
            .asReversed()
            .firstOrNull { it.role == com.woogit.aicore.conversation.ConversationMessage.Role.ASSISTANT }
            ?: return null
        return parser.parse(assistant.content)
    }
}
