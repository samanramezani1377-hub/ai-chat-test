package com.woogit.aicore.runtime.android

import com.woogit.aicore.domain.ChatMessage

/**
 * Qwen3.8 text-only chat template.
 *
 * Qwen3.8 is a Qwen3.5/qwen35 hybrid model and its embedded template is not
 * equivalent to the older Qwen3 ChatML template. In particular, historical
 * assistant turns carry an explicit empty/preserved <think> block and the
 * generation prompt starts inside <think>. Keeping those markers identical
 * is required both for model behavior and for exact prefix/KV reuse.
 */
internal object Qwen3PromptFormatter {
    private const val IM_START = "<|im_start|>"
    private const val IM_END = "<|im_end|>"

    fun format(messages: List<ChatMessage>): String {
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

            // Qwen3.8 reasoning is enabled. The model begins its next answer
            // inside the think block, exactly like the embedded template.
            append(IM_START).append("assistant\n")
            append("<think>\n")
        }
    }

    /**
     * Preserve the reasoning that was actually generated in a previous turn.
     * If the runtime/UI has already removed the think block, emit the exact
     * empty reasoning wrapper expected by Qwen3.8.
     */
    private fun StringBuilder.appendAssistantHistory(content: String) {
        val text = content.trim()
        val thinkStart = "<think>"
        val thinkEnd = "</think>"
        val start = text.indexOf(thinkStart)
        val end = if (start >= 0) text.indexOf(thinkEnd, start + thinkStart.length) else -1

        append("<think>\n")
        if (start >= 0 && end >= 0) {
            val reasoning = text.substring(start + thinkStart.length, end).trim()
            if (reasoning.isNotEmpty()) append(reasoning).append('\n')
        }
        append("</think>\n\n")

        if (start >= 0 && end >= 0) {
            val answer = text.substring(end + thinkEnd.length).trimStart()
            append(answer)
        } else {
            append(text)
        }
    }
}
