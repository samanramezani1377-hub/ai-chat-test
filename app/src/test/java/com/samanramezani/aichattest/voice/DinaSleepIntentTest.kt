package com.samanramezani.aichattest.voice

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class DinaSleepIntentTest {
    private fun cmd(s: String) = DinaSleepIntent.matchesCommand(s)
    private fun utt(s: String, wake: Boolean = true) = DinaSleepIntent.matchesUtterance(s, wake)

    @Test fun keepsOriginalPhrases() {
        assertTrue(cmd("خاموش شو")); assertTrue(cmd("خاموش")); assertTrue(cmd("برو بخواب"))
        assertTrue(cmd("فعلا خاموش شو")); assertTrue(cmd("خاموش شو؟")); assertTrue(cmd("خاموش\u200Cشو"))
    }

    @Test fun acceptsVerbFormsAndFillers() {
        assertTrue(cmd("خاموش بشو")); assertTrue(cmd("خاموش باش")); assertTrue(cmd("خاموش کن"))
        assertTrue(cmd("لطفاً خاموش شو")); assertTrue(cmd("خاموش شو دیگه")); assertTrue(cmd("خب خاموش شو بابا"))
        assertTrue(cmd("خودتو خاموش کن")); assertTrue(cmd("ساکت شو")); assertTrue(cmd("بخواب"))
    }

    @Test fun toleratesOneLetterRecognitionErrors() {
        assertTrue(cmd("خاموشو")); assertTrue(cmd("خامش شو")); assertTrue(cmd("خاموس شو")); assertTrue(cmd("خاموشش شو"))
    }

    @Test fun acceptsWakeWordAtEndOrMisheard() {
        assertTrue(utt("خاموش شو دینا")); assertTrue(utt("دینا لطفا خاموش شو")); assertTrue(utt("دنیا خاموش شو"))
        assertTrue(utt("دی نا خاموش شو")); assertTrue(utt("دانا، خاموش شو."))
    }

    @Test fun requiresWakeOnlyWhenAsked() {
        assertFalse(utt("خاموش شو", wake = true))
        assertTrue(utt("خاموش شو", wake = false))
    }

    @Test fun ignoresOrdinarySentences() {
        assertFalse(cmd("هوا چطوره")); assertFalse(cmd("صدای گوشی رو زیاد کن"))
        assertFalse(cmd("چراغ رو خاموش کن و برو"))
        assertFalse(cmd("کامپیوتر خاموش شد و من رفتم بخوابم"))
        assertFalse(utt("دینا چرا تلویزیون خاموش شد", wake = true))
        assertFalse(utt("دینا ساعت چنده", wake = true))
        assertFalse(cmd(""))
    }
}
