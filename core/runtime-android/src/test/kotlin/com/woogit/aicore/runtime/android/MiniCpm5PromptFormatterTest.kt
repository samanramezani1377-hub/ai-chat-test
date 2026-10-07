package com.woogit.aicore.runtime.android

import com.woogit.aicore.domain.ChatMessage
import kotlin.test.Test
import kotlin.test.assertEquals

class MiniCpm5PromptFormatterTest {
    @Test
    fun formatsHistoryAndReasoningGenerationPrompt() {
        val prompt = MiniCpm5PromptFormatter.format(
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
                "<|im_start|>assistant\n<think>\n\n</think>\n\nسلام!<|im_end|>\n" +
                "<|im_start|>user\nحالت چطوره؟<|im_end|>\n" +
                "<|im_start|>assistant\n<think>\n",
            prompt
        )
    }
}
