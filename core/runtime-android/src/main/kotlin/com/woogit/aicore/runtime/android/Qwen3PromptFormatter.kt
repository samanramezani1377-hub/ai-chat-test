package com.woogit.aicore.runtime.android

import com.woogit.aicore.domain.ChatMessage

/** Formats the conversation using Qwen3's ChatML control-token contract. */
internal object Qwen3PromptFormatter {
    private const val IM_START = "<|im_start|>"
    private const val IM_END = "<|im_end|>"

    fun format(messages: List<ChatMessage>): String {
        require(messages.isNotEmpty()) { "Conversation must contain at least one message" }

        val output = buildString {
            messages.forEach { message ->
                append(IM_START)
                append(roleName(message.role))
                append('\n')
                append(message.content)
                append(IM_END)
                append('\n')
            }
            append(IM_START)
            append("assistant\n")
        }
        return output
    }

    private fun roleName(role: ChatMessage.Role): String = when (role) {
        ChatMessage.Role.SYSTEM -> "system"
        ChatMessage.Role.USER -> "user"
        ChatMessage.Role.ASSISTANT -> "assistant"
        ChatMessage.Role.TOOL -> "tool"
    }
}
