package com.samanramezani.aichattest.voice

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class DinaCommandClassifierTest {
    @Test fun recognizesTimeWithoutCallingChatModel() {
        assertEquals(DinaIntent.GET_TIME, DinaCommandClassifier.classify("دینا ساعت چنده؟").intent)
    }

    @Test fun parsesPersianNumeralsForAbsoluteVolume() {
        val command = DinaCommandClassifier.classify("دینا صدا رو بذار روی ۳")
        assertEquals(DinaIntent.SET_VOLUME, command.intent)
        assertEquals(3, command.value)
    }

    @Test fun recognizesConversationalVolumeDecrease() {
        assertEquals(DinaIntent.DECREASE_VOLUME, DinaCommandClassifier.classify("دینا صداشو کم کن").intent)
    }

    @Test fun parsesBrightnessPercentage() {
        val command = DinaCommandClassifier.classify("دینا نور صفحه رو بذار روی ۳۰ درصد")
        assertEquals(DinaIntent.SET_BRIGHTNESS, command.intent)
        assertEquals(30, command.value)
    }

    @Test fun unknownRequestFallsThroughToChatUnchanged() {
        assertEquals(DinaIntent.CHAT, DinaCommandClassifier.classify("دینا درباره تاریخ ایران توضیح بده").intent)
    }
}
