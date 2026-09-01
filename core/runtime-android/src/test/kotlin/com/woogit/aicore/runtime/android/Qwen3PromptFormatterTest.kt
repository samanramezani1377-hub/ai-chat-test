package com.woogit.aicore.runtime.android

import com.woogit.aicore.domain.ChatMessage
import kotlin.test.Test
import kotlin.test.assertEquals

class Qwen3PromptFormatterTest {
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
                "<|im_start|>assistant\n",
            prompt
        )
    }

    @Test
    fun preservesToolMessages() {
        val prompt = Qwen3PromptFormatter.format(
            listOf(ChatMessage(ChatMessage.Role.TOOL, "tool-result"))
        )

        assertEquals(
            "<|im_start|>tool\ntool-result<|im_end|>\n<|im_start|>assistant\n",
            prompt
        )
    }
}
