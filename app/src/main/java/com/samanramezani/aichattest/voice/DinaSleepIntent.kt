package com.samanramezani.aichattest.voice

import java.text.Normalizer
import java.util.Locale

/**
 * Tolerant matcher for the «دینا خاموش شو» control phrase.
 *
 * The previous check compared the text after «دینا» against four exact strings, so it failed on
 * everyday variants and on ordinary ASR slips: wake word at the end («خاموش شو دینا»), polite or
 * filler words («دینا لطفاً خاموش شو دیگه»), other verb forms («خاموش بشو», «خاموش باش»),
 * one-letter recognition errors («خاموشو», «خامش شو») and a mis-heard wake word («دنیا»).
 *
 * Only an utterance that is *entirely* a sleep request (plus fillers) matches, so a normal
 * sentence that merely contains «خاموش» never ends the conversation. Pure Kotlin, no Android deps.
 */
internal object DinaSleepIntent {

    /** [text] is the speech after the wake word was already removed. */
    fun matchesCommand(text: String): Boolean = matchesTokens(tokenize(text))

    /**
     * [transcript] is a whole finalized utterance. With [requireWake] the (fuzzy) wake word must be
     * present, anywhere in the utterance; without it (already inside an active dialogue) a bare
     * «خاموش شو» is accepted as well.
     */
    fun matchesUtterance(transcript: String, requireWake: Boolean): Boolean {
        val tokens = tokenize(transcript)
        val rest = ArrayList<String>(tokens.size)
        var wakeFound = false
        var i = 0
        while (i < tokens.size) {
            val t = tokens[i]
            if (isWake(t)) { wakeFound = true; i++; continue }
            if (t == "دی" && i + 1 < tokens.size && tokens[i + 1] == "نا") { wakeFound = true; i += 2; continue }
            rest += t
            i++
        }
        if (requireWake && !wakeFound) return false
        return matchesTokens(rest)
    }

    private fun matchesTokens(all: List<String>): Boolean {
        var from = 0
        var to = all.size
        while (from < to && all[from] in FILLERS) from++
        while (to > from && all[to - 1] in FILLERS) to--
        val t = all.subList(from, to)
        return when (t.size) {
            1 -> isHead(t[0]) || t[0] == "بخواب"
            2 -> (isHead(t[0]) && t[1] in TAILS) || (t[0] == "برو" && t[1] == "بخواب")
            else -> false
        }
    }

    private fun isHead(token: String): Boolean = token == "ساکت" || editDistanceAtMost1(token, "خاموش")

    private fun isWake(token: String): Boolean =
        token in WAKE_VARIANTS || editDistanceAtMost1(token, "دینا")

    private fun tokenize(raw: String): List<String> {
        val sb = StringBuilder(raw.length)
        for (c in Normalizer.normalize(raw, Normalizer.Form.NFC).lowercase(Locale.ROOT)) {
            when {
                c == 'ي' || c == 'ى' -> sb.append('ی')
                c == 'ك' -> sb.append('ک')
                c == 'ة' -> sb.append('ه')
                c == '\u0640' || c in '\u064B'..'\u065F' || c == '\u0670' -> Unit
                c == '\u200C' || c == '\u200D' -> sb.append(' ')
                c.isLetterOrDigit() -> sb.append(c)
                else -> sb.append(' ')
            }
        }
        return sb.toString().split(' ').filter { it.isNotEmpty() }
    }

    /** True when the Levenshtein distance between [a] and [b] is 0 or 1. */
    private fun editDistanceAtMost1(a: String, b: String): Boolean {
        if (a == b) return true
        if (kotlin.math.abs(a.length - b.length) > 1) return false
        var i = 0
        while (i < a.length && i < b.length && a[i] == b[i]) i++
        return when {
            a.length == b.length -> a.substring(i + 1) == b.substring(i + 1)
            a.length < b.length -> a.substring(i) == b.substring(i + 1)
            else -> a.substring(i + 1) == b.substring(i)
        }
    }

    private val FILLERS = setOf(
        "لطفا", "خب", "خیلی", "باشه", "پس", "حالا", "دیگه", "فعلا", "الان", "ممنون", "مرسی",
        "بابا", "جان", "عزیزم", "عزیز", "ok", "okay", "اوکی", "هی", "یه", "لحظه", "خودت", "خودتو",
        "خودتون", "رو", "را",
    )

    private val TAILS = setOf(
        "شو", "شود", "شوو", "بشو", "بشود", "شوید", "شین", "شید", "شه", "بشه", "باش", "باشید",
        "کن", "کنید", "چو", "جو",
    )

    private val WAKE_VARIANTS = setOf("دینا", "دینه", "دنیا", "دانا", "dina", "deena", "dena")
}
