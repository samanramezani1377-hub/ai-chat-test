package com.samanramezani.aichattest.voice

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Test

class PersianSpeechTextTest {
    private fun n(s: String) = PersianSpeechText.normalize(s)

    @Test fun keepsPersianPunctuationAsPauses() = assertEquals("سلام, چطور کمک کنم?", n("سلام، چطور کمک کنم؟"))
    @Test fun removesEmoji() = assertEquals("سلام", n("😊 سلام ❤️"))
    @Test fun emojiOnlyBecomesNotSpeakable() = assertFalse(PersianSpeechText.hasSpeakableContent(n("😊🎉")))
    @Test fun replacesUrlInsteadOfSpellingIt() = assertEquals("سایت لینک را ببین", n("سایت https://example.com/a را ببین"))
    @Test fun readsJalaliDateWithoutEnglishSlash() = assertEquals("25 مهر 1405", n("۱۴۰۵/۰۷/۲۵"))
    @Test fun readsFractionWithoutEnglishSlash() = assertEquals("3 بر 4", n("۳/۴"))
    @Test fun keepsDecimalPoint() = assertEquals("3.5 درصد", n("۳.۵ درصد"))
    @Test fun usesLexiconForKnownLatinWords() = assertEquals("از پایتون استفاده کن", n("از Python استفاده کن"))
    @Test fun spellsUnknownAcronyms() = assertEquals("ان پی یو", n("NPU"))
    @Test fun transliteratesUnknownLatinWords() {
        assertEquals("سرور", n("server"))
        assertEquals("توکن", n("token"))
        assertEquals("داتا", n("data"))
    }
    @Test fun turnsParenthesesIntoPauses() = assertEquals("مدل, قدیمی, بود", n("مدل (قدیمی) بود"))
    @Test fun collapsesEllipsisAndPunctuationRuns() = assertEquals("سلام, خوبی?", n("سلام... خوبی؟!"))
    @Test fun preservesZwnj() = assertEquals("می\u200Cخواهم", n("می\u200Cخواهم"))
    @Test fun normalizesArabicLetterVariants() = assertEquals("کتاب یک", n("كتاب يك"))
}
