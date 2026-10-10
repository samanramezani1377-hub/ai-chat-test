package com.woogit.aicore.runtime.android

import com.woogit.aicore.domain.ChatMessage

/**
 * Qwen3.8/Qwen3.5-compatible text-only chat template.
 *
 * Historical assistant turns preserve real reasoning when it is present, but
 * do not inject an empty <think> block. This keeps serialized history stable
 * for prefix/KV-cache reuse. The active generation prompt still starts inside
 * <think>.
 */
internal object Qwen3PromptFormatter {
    private const val IM_START = "<|im_start|>"
    private const val IM_END = "<|im_end|>"

    fun format(messages: List<ChatMessage>, enableThinking: Boolean = true): String {
        require(messages.isNotEmpty()) { "Conversation must contain at least one message" }

        return buildString {
            messages.forEach { message ->
                when (message.role) {
                    ChatMessage.Role.SYSTEM -> {
                        if (message.content.isNotBlank()) {
                            append(IM_START).append("system\n")
                            append(message.content.trim())
                            append(IM_END).append('\n')
                        }
                    }
                    ChatMessage.Role.USER -> {
                        append(IM_START).append("user\n")
                        append(message.content.trim())
                        append(IM_END).append('\n')
                    }
                    ChatMessage.Role.ASSISTANT -> {
                        append(IM_START).append("assistant\n")
                        appendAssistantHistory(message.content)
                        append(IM_END).append('\n')
                    }
                    ChatMessage.Role.TOOL -> {
                        append(IM_START).append("user\n")
                        append("<tool_response>\n")
                        append(message.content)
                        append("\n</tool_response>")
                        append(IM_END).append('\n')
                    }
                }
            }
            append(IM_START).append("assistant\n")
            if (enableThinking) append("<think>\n")
        }
    }

    private fun StringBuilder.appendAssistantHistory(content: String) {
        val text = content.trim()
        val thinkStart = "<think>"
        val thinkEnd = "</think>"
        val start = text.indexOf(thinkStart)
        val explicitEnd = if (start >= 0) text.indexOf(thinkEnd, start + thinkStart.length) else -1
        val end = if (explicitEnd >= 0) explicitEnd else text.indexOf(thinkEnd)

        if (end >= 0) {
            val reasoningStart = if (start >= 0) start + thinkStart.length else 0
            val reasoning = text.substring(reasoningStart, end).trim()
            if (reasoning.isNotEmpty()) {
                append("<think>\n")
                append(reasoning)
                append("\n</think>\n\n")
            }
            append(text.substring(end + thinkEnd.length).trimStart())
        } else {
            append(text)
        }
    }
}
