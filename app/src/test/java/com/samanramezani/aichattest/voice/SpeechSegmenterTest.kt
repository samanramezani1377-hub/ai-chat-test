package com.samanramezani.aichattest.voice

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class SpeechSegmenterTest {
    private fun run(vararg tokens: String): List<String> {
        val segmenter = SpeechSegmenter()
        val out = ArrayList<String>()
        for (t in tokens) out += segmenter.feed(t)
        out += segmenter.flush()
        return out
    }

    @Test fun splitsOnSentenceEnds() = assertEquals(listOf("سلام.", "خوبی؟"), run("سلام. خوبی؟"))
    @Test fun doesNotSplitDecimalAcrossTokens() = assertEquals(listOf("قیمت 3.5 درصد است."), run("قیمت 3", ".", "5 درصد است."))
    @Test fun doesNotSplitDomain() = assertEquals(listOf("برو به example.com بعد."), run("برو به example", ".com بعد."))
    @Test fun keepsEllipsisTogether() = assertEquals(listOf("خب...", "بله."), run("خب... بله."))
    @Test fun dropsSegmentsWithoutSpeakableContent() = assertEquals(listOf("بله!"), run("بله!", "!", " "))
    @Test fun handlesDotArrivingAsOwnToken() = assertEquals(listOf("یک.", "دو."), run("یک", ".", " دو", "."))
    @Test fun splitsOnNewline() = assertEquals(listOf("الف", "ب"), run("الف\nب"))
    @Test fun breaksVeryLongRunsAtSpaces() = assertTrue(run("کلمه ".repeat(40)).size >= 2)
}
