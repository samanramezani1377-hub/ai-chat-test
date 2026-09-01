package com.woogit.aicore.agent

/**
 * Parses the machine-readable ActionRequest emitted by the model.
 * The parser is deliberately strict: free-form text is never treated as an action.
 */
class ActionProtocolParser {
    fun parse(text: String): ActionIntent? {
        val start = text.indexOf('{')
        if (start < 0) return null
        val end = findObjectEnd(text, start) ?: return null
        val json = text.substring(start, end + 1)

        val version = stringOrNumber(json, "version")?.toIntOrNull() ?: return null
        if (version != 1) return null
        val action = string(json, "action") ?: return null
        val actionId = string(json, "actionId") ?: return null
        if (action.isBlank() || actionId.isBlank()) return null

        val arguments = object(json, "arguments") ?: return null
        return ActionIntent(
            actionId = action,
            input = arguments,
            explanation = "ActionRequest v1: $action"
        )
    }

    private fun string(json: String, key: String): String? =
        Regex("\\\"${Regex.escape(key)}\\\"\\s*:\\s*\\\"((?:\\\\.|[^\\\"])*)\\\"")
            .find(json)?.groupValues?.getOrNull(1)
            ?.replace("\\\"", "\"")
            ?.replace("\\\\", "\\")

    private fun stringOrNumber(json: String, key: String): String? =
        string(json, key) ?: Regex("\\\"${Regex.escape(key)}\\\"\\s*:\\s*(-?\\d+)")
            .find(json)?.groupValues?.getOrNull(1)

    private fun object(json: String, key: String): String? {
        val marker = Regex("\\\"${Regex.escape(key)}\\\"\\s*:").find(json) ?: return null
        val start = json.indexOf('{', marker.range.last + 1)
        if (start < 0) return null
        val end = findObjectEnd(json, start) ?: return null
        return json.substring(start, end + 1)
    }

    private fun findObjectEnd(text: String, start: Int): Int? {
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
                }
            }
        }
        return null
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
