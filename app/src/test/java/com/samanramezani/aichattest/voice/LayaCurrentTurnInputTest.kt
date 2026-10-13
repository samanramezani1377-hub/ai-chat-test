package com.samanramezani.aichattest.voice

import org.junit.Assert.assertEquals
import org.junit.Test

class LayaCurrentTurnInputTest {
    @Test fun extractsOnlyLastNumberedTurnFromSerializedTranscript() {
        assertEquals(
            "چطور می‌توانم بروم کرج؟",
            extractCurrentLayaTurn("۱. سلام ۲. سلام چطور می‌توانم کمکتان کنم؟ ۳. ساعت چنده؟ ۴. ساعت بیست‌ودو و سی دقیقه ۵. چطور می‌توانم بروم کرج؟")
        )
    }

    @Test fun stripsSingleLeadingTurnNumber() {
        assertEquals("ساعت چنده؟", extractCurrentLayaTurn("۳. ساعت چنده؟"))
    }

    @Test fun selectsLatestUserRoleAndDropsAssistantHistory() {
        assertEquals(
            "چطور می‌توانم بروم کرج؟",
            extractCurrentLayaTurn("User: سلام\nAssistant: سلام\nUser: چطور می‌توانم بروم کرج؟")
        )
    }

    @Test fun preservesPlainCurrentUtterance() {
        assertEquals("چطور می‌توانم بروم کرج؟", extractCurrentLayaTurn("چطور می‌توانم بروم کرج؟"))
    }

    @Test fun refusesTranscriptWithNoUserTurn() {
        assertEquals("", extractCurrentLayaTurn("Assistant: ساعت بیست‌ودو و سی دقیقه"))
    }
}
