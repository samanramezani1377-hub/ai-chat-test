package com.samanramezani.aichattest.voice

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class WakeWordModelsTest {
    @Test
    fun acceptsOnlyDinaLabels() {
        assertTrue(isDinaWakeKeywordLabel("دینا"))
        assertTrue(isDinaWakeKeywordLabel("  دینا  "))
        assertTrue(isDinaWakeKeywordLabel("@Dina"))
        assertTrue(isDinaWakeKeywordLabel("D I N A"))
        assertFalse(isDinaWakeKeywordLabel(""))
        assertFalse(isDinaWakeKeywordLabel("hello"))
        assertFalse(isDinaWakeKeywordLabel("نزدیک دینا"))
    }

    @Test
    fun serviceStaysForegroundOnlyForAnActiveConversationAfterWakeDetection() {
        assertTrue(
            shouldKeepWakeServiceForConversation(
                armed = true,
                listeningEnabled = false,
                phraseDetected = true,
                controllerActive = true,
            ),
        )
        assertFalse(
            shouldKeepWakeServiceForConversation(
                armed = true,
                listeningEnabled = false,
                phraseDetected = false,
                controllerActive = false,
            ),
        )
        // Pausing the listener must not leave an idle foreground service/wake lock behind.
        assertFalse(
            shouldKeepWakeServiceForConversation(
                armed = true,
                listeningEnabled = false,
                phraseDetected = false,
                controllerActive = false,
            ),
        )
        assertFalse(
            shouldKeepWakeServiceForConversation(
                armed = true,
                listeningEnabled = true,
                phraseDetected = true,
                controllerActive = true,
            ),
        )
    }
}
