package com.woogit.aicore.runtime.android

import com.woogit.aicore.domain.ChatMessage
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class Qwen3PromptFormatterTest {
    @Test
    fun systemPrefixIsExactAndExcludesSessionSpecificMessages() {
        val messages = listOf(
            ChatMessage(ChatMessage.Role.SYSTEM, "  Global instructions  "),
            ChatMessage(ChatMessage.Role.SYSTEM, "Session summary that may change"),
            ChatMessage(ChatMessage.Role.USER, "Hello"),
        )

        val prefix = Qwen3PromptFormatter.formatSystemPrefix(messages)
        val fullPrompt = Qwen3PromptFormatter.format(messages)

        assertEquals("<|im_start|>system\nGlobal instructions<|im_end|>\n", prefix)
        assertTrue(fullPrompt.startsWith(prefix))
        assertFalse(prefix.contains("Session summary"))
        assertFalse(prefix.contains("<|im_start|>assistant"))
    }

    @Test
    fun formatsSystemUserHistoryAndAssistantGenerationPrompt() {
        val prompt = Qwen3PromptFormatter.format(
            listOf(
                ChatMessage(ChatMessage.Role.SYSTEM, "You are helpful."),
                ChatMessage(ChatMessage.Role.USER, "سلام"),
                ChatMessage(ChatMessage.Role.ASSISTANT, "سلام!"),
                ChatMessage(ChatMessage.Role.USER, "حالت چطوره؟"),
            )
        )

        assertEquals(
            "<|im_start|>system\nYou are helpful.<|im_end|>\n" +
                "<|im_start|>user\nسلام<|im_end|>\n" +
                "<|im_start|>assistant\nسلام!<|im_end|>\n" +
                "<|im_start|>user\nحالت چطوره؟<|im_end|>\n" +
                "<|im_start|>assistant\n<think>\n",
            prompt
        )
    }

    @Test
    fun noThinkingModeSelectsFinalChannelWithoutThinkBlock() {
        val prompt = Qwen3PromptFormatter.format(
            listOf(ChatMessage(ChatMessage.Role.USER, "سلام")),
            enableThinking = false,
        )

        assertEquals(
            "<|im_start|>user\nسلام<|im_end|>\n<|im_start|>assistant\n<|channel|>final\n",
            prompt,
        )
        assertFalse(prompt.contains("<think>"))
    }

    @Test
    fun noThinkingModeRemovesHistoricalReasoningAndKeepsFinalAnswer() {
        val prompt = Qwen3PromptFormatter.format(
            listOf(
                ChatMessage(ChatMessage.Role.USER, "سوال قبلی"),
                ChatMessage(
                    ChatMessage.Role.ASSISTANT,
                    "<think>private reasoning</think>پاسخ قبلی",
                ),
                ChatMessage(ChatMessage.Role.USER, "سوال جدید"),
            ),
            enableThinking = false,
        )

        assertFalse(prompt.contains("private reasoning"))
        assertFalse(prompt.contains("<think>"))
        assertTrue(prompt.contains("پاسخ قبلی"))
        assertTrue(prompt.endsWith("<|im_start|>assistant\n<|channel|>final\n"))
    }

    @Test
    fun noThinkingModeDropsUnclosedHistoricalReasoning() {
        val prompt = Qwen3PromptFormatter.format(
            listOf(
                ChatMessage(ChatMessage.Role.ASSISTANT, "متن قبل <think>reasoning that never closed"),
            ),
            enableThinking = false,
        )

        assertFalse(prompt.contains("reasoning that never closed"))
        assertFalse(prompt.contains("<think>"))
    }

    @Test
    fun preservesToolMessages() {
        val prompt = Qwen3PromptFormatter.format(
            listOf(ChatMessage(ChatMessage.Role.TOOL, "tool-result"))
        )

        assertEquals(
            "<|im_start|>user\n<tool_response>\ntool-result\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n",
            prompt
        )
    }

    @Test
    fun preservesHistoricalReasoningWhenThinkingIsEnabled() {
        val prompt = Qwen3PromptFormatter.format(
            listOf(
                ChatMessage(
                    ChatMessage.Role.ASSISTANT,
                    "<think>\nreasoning\n</think>\n\nfinal answer"
                )
            )
        )

        assertEquals(
            "<|im_start|>assistant\n<think>\nreasoning\n</think>\n\nfinal answer<|im_end|>\n" +
                "<|im_start|>assistant\n<think>\n",
            prompt
        )
    }
}
