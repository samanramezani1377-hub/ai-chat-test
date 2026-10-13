package com.samanramezani.aichattest.voice

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class AlwaysOnKoochikTest {
    @Test
    fun extractsOnlySpeechAfterPersianWakeWord() {
        assertEquals("سلام چطوری", extractDinaCommand("دینا، سلام چطوری"))
        assertEquals("ساعت چنده؟", extractDinaCommand("دینا ساعت چنده؟"))
        assertEquals("برو خانه", extractDinaCommand("سلام دینا برو خانه"))
    }

    @Test
    fun acceptsArabicPersianAndLatinSpellingVariants() {
        assertEquals("سلام", extractDinaCommand("دينا سلام"))
        assertEquals("سلام", extractDinaCommand("دينه سلام"))
        assertEquals("hello", extractDinaCommand("Dina, hello"))
    }

    @Test
    fun distinguishesWakeWordAloneFromNoWakeWord() {
        assertEquals("", extractDinaCommand("دینا"))
        assertEquals("", extractDinaCommand("دینا،"))
        assertNull(extractDinaCommand("سلام خوبی"))
        assertNull(extractDinaCommand("مدینا سلام"))
    }
}
