package com.woogit.aicore.runtime.android

import com.woogit.aicore.domain.ChatMessage

/** Formats conversations using Liquid AI's official LFM2 ChatML-like template. */
internal object Lfm2PromptFormatter {
    private const val BOS = "<|startoftext|>"
    private const val IM_START = "<|im_start|>"
    private const val IM_END = "<|im_end|>"

    fun format(messages: List<ChatMessage>): String {
        require(messages.isNotEmpty()) { "Conversation must contain at least one message" }

        return buildString {
            append(BOS)
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
    }

    private fun roleName(role: ChatMessage.Role): String = when (role) {
        ChatMessage.Role.SYSTEM -> "system"
        ChatMessage.Role.USER -> "user"
        ChatMessage.Role.ASSISTANT -> "assistant"
        ChatMessage.Role.TOOL -> "tool"
    }
}
