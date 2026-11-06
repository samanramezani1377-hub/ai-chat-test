package com.samanramezani.aichattest.ui.voice

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class VoiceBargeInGateTest {
    @Test
    fun ambientNoiseDuringModelGenerationDoesNotInterruptTurn() {
        assertFalse(
            shouldInterruptForVoiceBargeIn(
                turnRunning = true,
                assistantSpeaking = false,
                speech = true,
                rms = 0.12f,
                threshold = 0.035f,
                aecActive = true,
            )
        )
    }

    @Test
    fun playbackCannotBeInterruptedWhenEchoCancellationIsUnavailable() {
        assertFalse(
            shouldInterruptForVoiceBargeIn(
                turnRunning = true,
                assistantSpeaking = true,
                speech = true,
                rms = 0.2f,
                threshold = 0.06f,
                aecActive = false,
            )
        )
    }

    @Test
    fun quietFrameDoesNotTriggerBargeIn() {
        assertFalse(
            shouldInterruptForVoiceBargeIn(
                turnRunning = true,
                assistantSpeaking = true,
                speech = true,
                rms = 0.04f,
                threshold = 0.06f,
                aecActive = true,
            )
        )
    }

    @Test
    fun sufficientlyLoudSpeechCanInterruptDuringPlaybackWithAec() {
        assertTrue(
            shouldInterruptForVoiceBargeIn(
                turnRunning = true,
                assistantSpeaking = true,
                speech = true,
                rms = 0.09f,
                threshold = 0.06f,
                aecActive = true,
            )
        )
    }
}
