package com.woogit.aicore.runtime.android

import com.woogit.aicore.domain.ChatMessage

/** Formats MiniCPM5 conversations according to the model's official ChatML template. */
internal object MiniCpm5PromptFormatter {
    private const val IM_START = "<|im_start|>"
    private const val IM_END = "<|im_end|>"

    fun format(messages: List<ChatMessage>): String {
        require(messages.isNotEmpty()) { "Conversation must contain at least one message" }

        return buildString {
            messages.forEach { message ->
                when (message.role) {
                    ChatMessage.Role.ASSISTANT -> {
                        append(IM_START).append("assistant\n")
                        val content = message.content
                        if ("<think>" in content || "</think>" in content) {
                            append(content)
                        } else {
                            append("<think>\n\n</think>\n\n")
                            append(content)
                        }
                        append(IM_END).append('\n')
                    }
                    ChatMessage.Role.TOOL -> {
                        append(IM_START).append("user\n<tool_response>\n")
                        append(message.content)
                        append("\n</tool_response>")
                        append(IM_END).append('\n')
                    }
                    else -> {
                        append(IM_START).append(roleName(message.role)).append('\n')
                        append(message.content)
                        append(IM_END).append('\n')
                    }
                }
            }
            append(IM_START).append("assistant\n")
            append("<think>\n")
        }
    }

    private fun roleName(role: ChatMessage.Role): String = when (role) {
        ChatMessage.Role.SYSTEM -> "system"
        ChatMessage.Role.USER -> "user"
        ChatMessage.Role.ASSISTANT -> "assistant"
        ChatMessage.Role.TOOL -> "user"
    }
}
