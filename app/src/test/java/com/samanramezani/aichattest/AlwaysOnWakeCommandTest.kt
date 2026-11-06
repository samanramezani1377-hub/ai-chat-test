package com.samanramezani.aichattest

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class AlwaysOnWakeCommandTest {
    @Test
    fun stopConversationPhrasesKeepWakeListenerEnabled() {
        assertTrue(isDinaSleepCommand("خاموش شو"))
        assertTrue(isDinaSleepCommand("خاموش"))
        assertTrue(isDinaSleepCommand("برو بخواب"))
        assertTrue(isDinaSleepCommand("فعلا خاموش شو"))
    }

    @Test
    fun unrelatedCommandsAreNotTreatedAsWakeShutdown() {
        assertFalse(isDinaSleepCommand("هوا چطوره"))
        assertFalse(isDinaSleepCommand("صدای گوشی رو زیاد کن"))
    }

    @Test
    fun arabicPersianLetterVariantsAndTrailingPunctuationAreNormalized() {
        assertTrue(isDinaSleepCommand("خاموش شو؟"))
        assertTrue(isDinaSleepCommand("خاموش‌شو"))
    }
}
