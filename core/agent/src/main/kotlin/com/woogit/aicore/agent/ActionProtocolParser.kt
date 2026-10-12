package com.woogit.aicore.agent

import com.woogit.aicore.conversation.ConversationContext
import com.woogit.aicore.conversation.ConversationMessage

/**
 * Parses one complete ActionRequest v1 object.
 *
 * It tolerates the exact formatting mistakes commonly emitted by small local models
 * (a closed Qwen reasoning block or a JSON-only Markdown fence), but never extracts
 * a JSON-looking fragment from arbitrary prose or executes shell snippets.
 */
class ActionProtocolParser {
    fun parse(text: String): ActionIntent? {
        val candidate = extractCandidate(text) ?: return null
        val reader = JsonReader(candidate)
        val root = runCatching { reader.readRootObject() }.getOrNull() ?: return null
        if (reader.position != candidate.length) return null

        val version = (root["version"] as? JsonValue.NumberValue)?.raw?.toIntOrNull()
            ?: (root["version"] as? JsonValue.StringValue)?.value?.toIntOrNull()
            ?: return null
        if (version != 1) return null
        val action = (root["action"] as? JsonValue.StringValue)?.value ?: return null
        val requestId = (root["actionId"] as? JsonValue.StringValue)?.value ?: return null
        val arguments = root["arguments"] as? JsonValue.ObjectValue ?: return null
        if (action.isBlank() || requestId.isBlank()) return null

        return ActionIntent(
            actionId = action,
            requestId = requestId,
            input = arguments.raw,
            explanation = "ActionRequest v1: $action",
        )
    }

    private fun extractCandidate(text: String): String? {
        var candidate = text.trim()
        if (candidate.isEmpty()) return null

        // The Qwen prompt intentionally opens <think> before generation. The model
        // may therefore return reasoning without the opening tag, followed by </think>.
        val reasoningEnd = if (candidate.startsWith("{")) -1 else candidate.indexOf("</think>")
        if (reasoningEnd >= 0) candidate = candidate.substring(reasoningEnd + "</think>".length).trim()
        if (candidate.startsWith("<think>")) return null

        if (candidate.startsWith("```")) {
            val firstNewline = candidate.indexOf('\n')
            if (firstNewline < 0) return null
            val closing = candidate.lastIndexOf("```")
            if (closing <= firstNewline || candidate.substring(closing + 3).isNotBlank()) return null
            val language = candidate.substring(3, firstNewline).trim()
            if (language.isNotEmpty() && !language.equals("json", ignoreCase = true)) return null
            candidate = candidate.substring(firstNewline + 1, closing).trim()
        }
        if (!candidate.startsWith('{') || !candidate.endsWith('}')) return null
        return candidate
    }

    private sealed interface JsonValue {
        data class StringValue(val value: String) : JsonValue
        data class NumberValue(val raw: String) : JsonValue
        data class ObjectValue(val raw: String, val members: Map<String, JsonValue>) : JsonValue
        data class ArrayValue(val values: List<JsonValue>) : JsonValue
        data object Other : JsonValue
    }

    /** Small strict JSON reader: validates the whole document and rejects duplicate keys. */
    private class JsonReader(private val source: String) {
        var position: Int = 0
            private set

        fun readRootObject(): Map<String, JsonValue> {
            skipWhitespace()
            val value = readValue()
            require(value is JsonValue.ObjectValue) { "Root must be an object" }
            skipWhitespace()
            require(position == source.length) { "Trailing data" }
            return value.members
        }

        private fun readValue(): JsonValue {
            skipWhitespace()
            require(position < source.length) { "Unexpected end of JSON" }
            return when (source[position]) {
                '"' -> JsonValue.StringValue(readString())
                '{' -> readObject()
                '[' -> readArray()
                't' -> { literal("true"); JsonValue.Other }
                'f' -> { literal("false"); JsonValue.Other }
                'n' -> { literal("null"); JsonValue.Other }
                '-', in '0'..'9' -> JsonValue.NumberValue(readNumber())
                else -> error("Invalid JSON value")
            }
        }

        private fun readObject(): JsonValue.ObjectValue {
            val start = position
            expect('{')
            skipWhitespace()
            val members = linkedMapOf<String, JsonValue>()
            if (consume('}')) return JsonValue.ObjectValue(source.substring(start, position), members)
            while (true) {
                skipWhitespace()
                require(position < source.length && source[position] == '"') { "Object key must be a string" }
                val key = readString()
                skipWhitespace()
                expect(':')
                val value = readValue()
                require(!members.containsKey(key)) { "Duplicate JSON key: $key" }
                members[key] = value
                skipWhitespace()
                if (consume('}')) break
                expect(',')
            }
            return JsonValue.ObjectValue(source.substring(start, position), members)
        }

        private fun readArray(): JsonValue.ArrayValue {
            expect('[')
            skipWhitespace()
            val values = mutableListOf<JsonValue>()
            if (consume(']')) return JsonValue.ArrayValue(values)
            while (true) {
                values += readValue()
                skipWhitespace()
                if (consume(']')) break
                expect(',')
            }
            return JsonValue.ArrayValue(values)
        }

        private fun readString(): String {
            expect('"')
            val out = StringBuilder()
            while (position < source.length) {
                val c = source[position++]
                when {
                    c == '"' -> return out.toString()
                    c == '\\' -> {
                        require(position < source.length) { "Incomplete escape" }
                        when (val escaped = source[position++]) {
                            '"', '\\', '/' -> out.append(escaped)
                            'b' -> out.append('\b')
                            'f' -> out.append('\u000C')
                            'n' -> out.append('\n')
                            'r' -> out.append('\r')
                            't' -> out.append('\t')
                            'u' -> {
                                require(position + 4 <= source.length) { "Incomplete unicode escape" }
                                val code = source.substring(position, position + 4).toIntOrNull(16)
                                    ?: error("Invalid unicode escape")
                                out.append(code.toChar())
                                position += 4
                            }
                            else -> error("Invalid string escape")
                        }
                    }
                    c.code < 0x20 -> error("Unescaped control character")
                    else -> out.append(c)
                }
            }
            error("Unterminated string")
        }

        private fun readNumber(): String {
            val start = position
            consume('-')
            require(position < source.length) { "Incomplete number" }
            if (consume('0')) {
                require(position >= source.length || source[position] !in '0'..'9') { "Leading zero" }
            } else {
                require(source[position] in '1'..'9') { "Invalid number" }
                while (position < source.length && source[position] in '0'..'9') position++
            }
            if (consume('.')) {
                require(position < source.length && source[position] in '0'..'9') { "Invalid fraction" }
                while (position < source.length && source[position] in '0'..'9') position++
            }
            if (position < source.length && source[position] in "eE") {
                position++
                if (position < source.length && source[position] in "+-") position++
                require(position < source.length && source[position] in '0'..'9') { "Invalid exponent" }
                while (position < source.length && source[position] in '0'..'9') position++
            }
            return source.substring(start, position)
        }

        private fun literal(value: String) {
            require(source.startsWith(value, position)) { "Invalid literal" }
            position += value.length
        }

        private fun skipWhitespace() {
            while (position < source.length && source[position].isWhitespace()) position++
        }

        private fun expect(c: Char) {
            require(consume(c)) { "Expected '$c'" }
        }

        private fun consume(c: Char): Boolean {
            if (position < source.length && source[position] == c) {
                position++
                return true
            }
            return false
        }
    }
}

/**
 * Intent heuristic is used only to decide whether a malformed tool response deserves
 * one bounded repair attempt. It does not authorize or execute an action.
 */
class ProtocolActionIntentPlanner(
    private val parser: ActionProtocolParser = ActionProtocolParser()
) : ActionIntentPlanner {
    override suspend fun plan(context: ConversationContext): ActionIntent? {
        val assistant = context.recentMessages
            .asReversed()
            .firstOrNull { it.role == ConversationMessage.Role.ASSISTANT }
            ?: return null
        return parser.parse(assistant.content)
    }

    override fun requiresAction(context: ConversationContext): Boolean {
        val messages = context.recentMessages
        val lastUserIndex = messages.indexOfLast { it.role == ConversationMessage.Role.USER }
        if (lastUserIndex < 0) return false

        val laterToolMessages = messages.drop(lastUserIndex + 1)
            .filter { it.role == ConversationMessage.Role.TOOL }
        // A repair feedback message keeps the original request in action mode; a real
        // tool result means the requested operation has already been attempted.
        if (laterToolMessages.lastOrNull()?.content?.startsWith("ACTION_PROTOCOL_REPAIR:") == true) {
            // The latest tool message is repair feedback; AgentSession owns the retry bound.
        } else if (laterToolMessages.isNotEmpty()) {
            // A real action result supersedes earlier repair feedback.
            return false
        }

        val user = messages[lastUserIndex].content.lowercase()
        val indicators = listOf(
            "create_file", "read_file", "list_files", "delete_file", "calculate",
            "create a file", "make a file", "write to a file", "save this file", "delete the file",
            "بساز", "ایجاد کن", "بنویس", "ذخیره کن", "پاک کن", "حذف کن",
            "بخون", "بخوان", "فهرست کن", "محاسبه کن", "درست کن",
            "بساز برام", "ذخیره‌اش", "ذخیره اش", "اسمشو بزار", "اسمش را بگذار"
        )
        return indicators.any(user::contains)
    }
}
