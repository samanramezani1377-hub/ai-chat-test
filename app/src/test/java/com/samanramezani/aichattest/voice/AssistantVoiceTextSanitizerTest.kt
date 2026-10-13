package com.samanramezani.aichattest.voice

import com.samanramezani.aichattest.ui.voice.sanitizeAssistantVoiceText
import org.junit.Assert.assertEquals
import org.junit.Test

class AssistantVoiceTextSanitizerTest {
    @Test fun dropsThoughtChannelWhenNoFinalAnswerExists() {
        assertEquals("", sanitizeAssistantVoiceText("<|channel|>analysis\nI should think through this carefully."))
    }

    @Test fun extractsOnlyFinalChannel() {
        assertEquals(
            "ساعت ۹ است.",
            sanitizeAssistantVoiceText("<|channel|>analysis\nprivate reasoning<|channel|>final\nساعت ۹ است.")
        )
    }

    @Test fun stripsThinkTagsBeforeSpeech() {
        assertEquals("پاسخ نهایی", sanitizeAssistantVoiceText("<think>تفکر داخلی</think>پاسخ نهایی"))
    }

    @Test fun preservesOrdinaryPlainText() {
        assertEquals("سلام، چطور کمک کنم؟", sanitizeAssistantVoiceText("سلام، چطور کمک کنم؟"))
    }
}
