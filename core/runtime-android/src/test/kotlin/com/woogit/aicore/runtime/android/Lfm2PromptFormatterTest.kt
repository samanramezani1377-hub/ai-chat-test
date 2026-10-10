package com.woogit.aicore.runtime.android

import com.woogit.aicore.domain.ChatMessage
import kotlin.test.Test
import kotlin.test.assertEquals

class Lfm2PromptFormatterTest {
    @Test
    fun formatsOfficialLfm2ChatTemplate() {
        val prompt = Lfm2PromptFormatter.format(
            listOf(
                ChatMessage(ChatMessage.Role.USER, "سلام"),
                ChatMessage(ChatMessage.Role.ASSISTANT, "سلام!"),
                ChatMessage(ChatMessage.Role.USER, "خوبی؟"),
            )
        )

        assertEquals(
            "<|startoftext|>" +
                "<|im_start|>user\nسلام<|im_end|>\n" +
                "<|im_start|>assistant\nسلام!<|im_end|>\n" +
                "<|im_start|>user\nخوبی؟<|im_end|>\n" +
                "<|im_start|>assistant\n",
            prompt
        )
    }
}
