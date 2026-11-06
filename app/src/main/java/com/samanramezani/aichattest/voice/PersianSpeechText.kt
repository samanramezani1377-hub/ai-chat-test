package com.samanramezani.aichattest.voice

import java.text.Normalizer

/**
 * Prepares assistant text for the Persian Piper/espeak-ng TTS path.
 *
 * Measured against espeak-ng's `fa` voice, these inputs break Persian speech:
 *  - emoji are read out as Persian words ("لبخند خجالتی ..."),
 *  - Latin words switch espeak to English phonemes, which a Persian-only VITS voice never saw
 *    in training, so they come out garbled and give the whole utterance a foreign accent,
 *  - URLs / e-mail addresses are spelled out character by character,
 *  - "/" is read as English "slash" (Jalali dates such as ۱۴۰۵/۰۷/۲۵ and fractions),
 *  - an emoji-only or symbol-only segment becomes blank and used to abort the whole reply.
 *
 * Everything espeak already handles well (Persian punctuation pauses, Persian/Arabic digits,
 * numbers, percent) is intentionally left alone. This object is pure Kotlin: no Android deps.
 */
internal object PersianSpeechText {

    fun hasSpeakableContent(text: String): Boolean = text.any { it.isLetterOrDigit() }

    fun normalize(raw: String): String {
        var s = Normalizer.normalize(raw, Normalizer.Form.NFC)
        s = stripInvisibleAndVariants(s)
        s = stripPictographs(s)
        s = URL_REGEX.replace(s, " لینک ")
        s = EMAIL_REGEX.replace(s, " ایمیل ")
        s = normalizeDigits(s)
        s = JALALI_DATE_REGEX.replace(s) { m ->
            val month = m.groupValues[2].toInt()
            "${m.groupValues[3].toInt()} ${JALALI_MONTHS[month - 1]} ${m.groupValues[1]}"
        }
        s = DIGIT_SLASH_DIGIT_REGEX.replace(s, "$1 بر $2")
        s = s.replace('/', ' ').replace('\\', ' ')
        s = LATIN_WORD_REGEX.replace(s) { m -> " ${latinToPersian(m.value)} " }
        s = LETTER_DIGIT_BOUNDARY_REGEX.replace(s, " ")
        s = s.replace("&", " و ")
            .replace('|', ',')
            .replace(Regex("[#*_~`^<>=]+"), " ")
            .replace(Regex("[()\\[\\]{}]"), ",")
            .replace(Regex("[«»“”\"‘’']"), "")
            .replace(Regex("\\s[-–—]+\\s|[–—]+"), ", ")
            .replace(Regex("\\.{2,}|…+"), ", ")
            .replace('،', ',')
            .replace('؛', ';')
            .replace('؟', '?')
        s = PUNCT_RUN_REGEX.replace(s) { m -> m.groupValues[1] }
        s = s.replace(Regex("\\s+([,;:.!?])"), "$1")
            .replace(Regex("([,;!?])(?=[^\\s\\d,;:.!?])"), "$1 ")
            .replace(Regex("[ \\t\\r\\n\\u000B\\u000C]+"), " ")
            .trim()
        return s.trimStart(',', ';', ':', '.', '!', '?', ' ')
    }

    private fun stripInvisibleAndVariants(value: String): String {
        val out = StringBuilder(value.length)
        for (c in value) {
            when (c) {
                '\u200B', '\u200D', '\u200E', '\u200F', '\uFEFF', '\u00AD', '\u0640' -> Unit
                in '\u202A'..'\u202E', in '\u2066'..'\u2069' -> Unit
                '\u00A0' -> out.append(' ')
                'ي', 'ى' -> out.append('ی')
                'ك' -> out.append('ک')
                'ة' -> out.append('ه')
                'ۀ' -> out.append("هٔ")
                else -> out.append(c)
            }
        }
        return out.toString()
    }

    /** Removes emoji, pictographs, dingbats and variation selectors (kept out of speech entirely). */
    private fun stripPictographs(value: String): String {
        val out = StringBuilder(value.length)
        var i = 0
        while (i < value.length) {
            val cp = value.codePointAt(i)
            i += Character.charCount(cp)
            val drop = cp in 0x1F000..0x1FAFF || cp in 0x2190..0x21FF || cp in 0x2300..0x23FF ||
                cp in 0x2460..0x24FF || cp in 0x25A0..0x27BF || cp in 0x2900..0x2BFF ||
                cp in 0xFE00..0xFE0F || cp in 0xE0000..0xE007F || cp == 0x20E3 || cp == 0x00A9 ||
                cp == 0x00AE || cp == 0x2122
            if (!drop) out.appendCodePoint(cp)
        }
        return out.toString()
    }

    private fun normalizeDigits(value: String): String {
        val out = StringBuilder(value.length)
        for (c in value) {
            when (c) {
                in '\u06F0'..'\u06F9' -> out.append('0' + (c - '\u06F0'))
                in '\u0660'..'\u0669' -> out.append('0' + (c - '\u0660'))
                '\u066B' -> out.append('.')
                '\u066C' -> Unit
                else -> out.append(c)
            }
        }
        return out.toString()
    }

    private fun latinToPersian(token: String): String {
        val word = token.replace("'", "").replace("’", "")
        if (word.isEmpty()) return ""
        LEXICON[word.lowercase()]?.let { return it }
        if (word.length in 2..6 && word.all { it in 'A'..'Z' }) {
            return word.map { LETTER_NAMES.getValue(it) }.joinToString(" ")
        }
        return transliterate(word.lowercase())
    }

    /** Conservative phonetic approximation for unknown Latin words; Persian script omits short vowels. */
    private fun transliterate(w: String): String {
        val sb = StringBuilder()
        var i = 0
        val n = w.length
        while (i < n) {
            val rest = w.substring(i)
            val digraph = DIGRAPHS.firstOrNull { rest.startsWith(it.first) }
            if (digraph != null) {
                sb.append(digraph.second)
                i += digraph.first.length
                continue
            }
            val c = w[i]
            if (i > 0 && c == w[i - 1] && c !in "aeiouy") { i++; continue }
            when (c) {
                'a' -> sb.append('ا')
                'e' -> if (i == 0) sb.append('ا')
                'i' -> sb.append(if (i == 0) "ای" else "ی")
                'o' -> sb.append(if (i == 0) "او" else "و")
                'u' -> sb.append(if (i == 0) "ا" else "و")
                'y' -> sb.append('ی')
                'c' -> sb.append(if (i + 1 < n && w[i + 1] in "eiy") 'س' else 'ک')
                'x' -> sb.append("کس")
                else -> sb.append(CONSONANTS[c] ?: "")
            }
            i++
        }
        return sb.toString()
    }

    private val URL_REGEX = Regex("""(?i)\b(?:https?://|www\.)\S+""")
    private val EMAIL_REGEX = Regex("""[\w.+-]+@[\w-]+(?:\.[\w-]+)+""")
    private val JALALI_DATE_REGEX = Regex("""(?<!\d)(1[34]\d{2})/(0?[1-9]|1[0-2])/(0?[1-9]|[12]\d|3[01])(?!\d)""")
    // "3/4" -> "3 بر 4" (espeak would otherwise say English "slash")
    private val DIGIT_SLASH_DIGIT_REGEX = Regex("""(\d+)\s*/\s*(\d+)""")
    private val LATIN_WORD_REGEX = Regex("""[A-Za-z]+(?:['’][A-Za-z]+)*""")
    private val LETTER_DIGIT_BOUNDARY_REGEX = Regex("""(?<=[\u0600-\u06FF])(?=\d)|(?<=\d)(?=[\u0600-\u06FF])""")
    private val PUNCT_RUN_REGEX = Regex("""([,;:.!?])(?:\s*[,;:.!?])+""")

    private val JALALI_MONTHS = listOf(
        "فروردین", "اردیبهشت", "خرداد", "تیر", "مرداد", "شهریور",
        "مهر", "آبان", "آذر", "دی", "بهمن", "اسفند",
    )

    private val LETTER_NAMES = mapOf(
        'A' to "ای", 'B' to "بی", 'C' to "سی", 'D' to "دی", 'E' to "ای", 'F' to "اف", 'G' to "جی",
        'H' to "اچ", 'I' to "آی", 'J' to "جی", 'K' to "کی", 'L' to "ال", 'M' to "ام", 'N' to "ان",
        'O' to "او", 'P' to "پی", 'Q' to "کیو", 'R' to "آر", 'S' to "اس", 'T' to "تی", 'U' to "یو",
        'V' to "وی", 'W' to "دبلیو", 'X' to "اکس", 'Y' to "وای", 'Z' to "زد",
    )

    private val DIGRAPHS = listOf(
        "tion" to "شن", "sh" to "ش", "ch" to "چ", "kh" to "خ", "gh" to "ق", "ph" to "ف",
        "th" to "ت", "ck" to "ک", "oo" to "و", "ee" to "ی", "ou" to "او", "qu" to "کو",
    )

    private val CONSONANTS = mapOf(
        'b' to "ب", 'd' to "د", 'f' to "ف", 'g' to "گ", 'h' to "ه", 'j' to "ج", 'k' to "ک",
        'l' to "ل", 'm' to "م", 'n' to "ن", 'p' to "پ", 'q' to "ک", 'r' to "ر", 's' to "س",
        't' to "ت", 'v' to "و", 'w' to "و", 'z' to "ز",
    )

    private val LEXICON = mapOf(
        "python" to "پایتون", "java" to "جاوا", "javascript" to "جاوااسکریپت", "kotlin" to "کاتلین",
        "android" to "اندروید", "ios" to "آی‌اواس", "windows" to "ویندوز", "linux" to "لینوکس",
        "google" to "گوگل", "openai" to "اوپن‌ای‌آی", "claude" to "کلود", "chatgpt" to "چت‌جی‌پی‌تی",
        "github" to "گیت‌هاب", "git" to "گیت", "wifi" to "وای‌فای", "bluetooth" to "بلوتوث",
        "internet" to "اینترنت", "email" to "ایمیل", "online" to "آنلاین", "offline" to "آفلاین",
        "app" to "اپ", "ok" to "اوکی", "hi" to "های", "hello" to "هلو", "whatsapp" to "واتس‌اپ",
        "telegram" to "تلگرام", "instagram" to "اینستاگرام", "youtube" to "یوتیوب", "api" to "ای‌پی‌آی",
        "ai" to "ای‌آی", "gpu" to "جی‌پی‌یو", "cpu" to "سی‌پی‌یو", "usb" to "یواس‌بی",
    )
}
