package com.woogit.aicore.runtime.android

import com.woogit.aicore.domain.ChatMessage

/**
 * Qwen3.8/Qwen3.5-compatible text-only chat template.
 *
 * Voice generation passes enableThinking=false: the active assistant prefix is
 * explicitly routed to the final channel and historical reasoning blocks are
 * omitted from the prompt. Normal chat keeps the model's normal thinking mode.
 */
internal object Qwen3PromptFormatter {
    private const val IM_START = "<|im_start|>"
    private const val IM_END = "<|im_end|>"

    /**
     * Exact serialized system-message prefix used by format(). It deliberately excludes
     * the assistant generation marker so native code can snapshot only the stable prefix.
     */
    fun formatSystemPrefix(messages: List<ChatMessage>): String = buildString {
        // Cache the invariant global system instruction only. Later SYSTEM messages can
        // contain per-session summaries, task state, or workspace context and must remain
        // part of the variable suffix when a new session is started.
        messages.firstOrNull { it.role == ChatMessage.Role.SYSTEM }
            ?.takeIf { it.content.isNotBlank() }
            ?.let { message ->
                append(IM_START).append("system\n")
                append(message.content.trim())
                append(IM_END).append('\n')
            }
    }

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
                        appendAssistantHistory(message.content, includeReasoning = enableThinking)
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
            if (enableThinking) {
                append("<think>\n")
            } else {
                // Qwen's explicit final channel disables reasoning for voice turns.
                append("<|channel|>final\n")
            }
        }
    }

    private fun StringBuilder.appendAssistantHistory(content: String, includeReasoning: Boolean) {
        val text = content.trim()
        val thinkStart = "<think>"
        val thinkEnd = "</think>"
        val start = text.indexOf(thinkStart)
        val explicitEnd = if (start >= 0) text.indexOf(thinkEnd, start + thinkStart.length) else -1
        val end = if (explicitEnd >= 0) explicitEnd else text.indexOf(thinkEnd)

        if (end >= 0) {
            val reasoningStart = if (start >= 0) start + thinkStart.length else 0
            val reasoning = text.substring(reasoningStart, end).trim()
            val finalText = text.substring(end + thinkEnd.length).trimStart()
            if (includeReasoning && reasoning.isNotEmpty()) {
                append("<think>\n")
                append(reasoning)
                append("\n</think>\n\n")
            }
            append(finalText)
        } else if (includeReasoning) {
            append(text)
        } else {
            // Avoid replaying unclosed reasoning or analysis text in no-think mode.
            append(text.substringBefore("<think>").trim())
        }
    }
}
