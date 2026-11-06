package com.samanramezani.aichattest.voice

/**
 * Splits streamed assistant text into sentences for incremental TTS.
 *
 * Unlike a plain "flush on every '.'" rule, a single '.' only ends a sentence when it is followed
 * by whitespace/end of text. That keeps "3.5", "1.2.3" and "example.com" in one segment instead of
 * cutting them mid-token (which resets prosody and made numbers sound broken). Segments that carry
 * no letters or digits (stray "!", "»", "...") are dropped rather than sent to the synthesizer.
 */
internal class SpeechSegmenter(private val softLimit: Int = 120) {
    private val buffer = StringBuilder()
    private var dotRun = 0

    fun feed(text: String): List<String> {
        val ready = ArrayList<String>()
        for (c in text) {
            if (dotRun > 0) {
                when {
                    c == '.' -> { buffer.append(c); dotRun++; continue }
                    c.isWhitespace() -> { emit(ready); continue }
                    dotRun > 1 -> emit(ready)
                    else -> dotRun = 0 // single '.' inside a token: decimal point, version, domain
                }
            }
            buffer.append(c)
            when {
                c == '.' -> dotRun = 1
                c == '!' || c == '?' || c == '؟' || c == '؛' || c == '\n' -> emit(ready)
                c == ' ' && buffer.length >= softLimit -> emit(ready)
            }
        }
        return ready
    }

    fun flush(): List<String> {
        val ready = ArrayList<String>(1)
        emit(ready)
        return ready
    }

    private fun emit(out: MutableList<String>) {
        val segment = buffer.toString().trim()
        buffer.setLength(0)
        dotRun = 0
        if (PersianSpeechText.hasSpeakableContent(segment)) out.add(segment)
    }
}
