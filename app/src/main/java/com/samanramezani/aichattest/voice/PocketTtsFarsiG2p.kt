package com.samanramezani.aichattest.voice

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.Closeable
import java.nio.LongBuffer

/**
 * Android implementation of the exported Persian ByT5 G2P graphs.
 *
 * This is the first executable stage of Pocket TTS: UTF-8 bytes are mapped to
 * the ByT5 vocabulary (byte + 3), encoded by g2p_encoder.onnx, then greedily
 * decoded by g2p_decoder.onnx. The model's G2P output is phonemes, not audio.
 */
internal class PocketTtsFarsiG2p(private val modelDir: java.io.File) : Closeable {
    private val env = OrtEnvironment.getEnvironment()
    private val options = OrtSession.SessionOptions()
    private val encoder: OrtSession
    private val decoder: OrtSession

    init {
        options.setIntraOpNumThreads(2)
        options.setInterOpNumThreads(1)
        try {
            encoder = env.createSession(java.io.File(modelDir, "g2p_encoder.onnx").absolutePath, options)
            decoder = env.createSession(java.io.File(modelDir, "g2p_decoder.onnx").absolutePath, options)
        } catch (t: Throwable) {
            options.close()
            throw IllegalStateException("POCKET-TTS-G2P-001: بارگذاری گراف‌های G2P ناموفق بود: ${t.message}", t)
        }
    }

    @Synchronized
    fun phonemise(input: String): String {
        val text = normalize(transliterate(input))
        if (text.isBlank()) return ""
        val windows = splitWindows(text)
        return windows.mapNotNull { window -> decodeWindow(window).takeIf { it.isNotBlank() } }
            .joinToString(" ")
            .replace(Regex("\\s+"), " ")
            .trim()
            .map { PHONEME_REPLACEMENTS[it] ?: it }
            .joinToString("")
            .split(Regex("\\s+"))
            .joinToString(" ") { PRONUNCIATION_FIXES[it] ?: it }
    }

    private fun decodeWindow(text: String): String {
        val bytes = text.toByteArray(Charsets.UTF_8)
        if (bytes.isEmpty()) return ""
        val inputIds = LongArray(bytes.size) { (bytes[it].toInt() and 0xff).toLong() + 3L }
        val encoded = tensor(inputIds)
        // Copy the encoder result before closing the Result. ONNX Runtime may expose
        // a tensor's value as a nested primitive array (for example float[][][] / [[[F),
        // rather than an OnnxTensor instance. The decoder needs an owned Tensor that
        // remains valid after the encoder Result is closed.
        val hidden = try {
            encoder.run(mapOf("input_ids" to encoded)).use { result ->
                require(result.size() > 0) { "گراف encoder هیچ خروجی‌ای برنگرداند" }
                val outputName = encoder.outputNames.firstOrNull()
                    ?: error("گراف encoder هیچ خروجی نام‌گذاری‌شده‌ای ندارد")
                val outputValue = result.get(outputName)
                    ?: error("خروجی encoder پیدا نشد: $outputName")
                val output = outputValue as? OnnxTensor
                    ?: throw IllegalStateException(
                        "POCKET-TTS-G2P-002: خروجی encoder از نوع Tensor نیست؛ " +
                            "outputClass=${outputValue.javaClass.name}, outputs=${encoder.outputNames.joinToString()}"
                    )
                copyFloatTensor(output)
            }
        } finally {
            encoded.close()
        }

        val generated = ArrayList<Long>(48)
        generated += 0L // ByT5 decoder_start_token_id = PAD
        try {
            repeat(MAX_DECODE_TOKENS) {
                val ids = tensor(generated.toLongArray())
                val logits = try {
                    decoder.run(mapOf("decoder_input_ids" to ids, "encoder_hidden" to hidden)).use { result ->
                        require(result.size() > 0) { "گراف decoder هیچ خروجی‌ای برنگرداند" }
                        // Read the ONNX value itself. Accessing result[0].value materializes
                        // a JVM array (or another wrapper) and then incorrectly treats that
                        // Java representation as the model output object. ONNX Runtime already
                        // owns the actual tensor in the Result; consume its typed buffer while
                        // the Result is open instead of round-tripping through value/createTensor.
                        val outputName = decoder.outputNames.firstOrNull()
                            ?: error("گراف decoder هیچ خروجی نام‌گذاری‌شده‌ای ندارد")
                        val outputValue = result.get(outputName)
                            ?: error("خروجی decoder پیدا نشد: $outputName")
                        val outputTensor = outputValue as? OnnxTensor
                            ?: throw IllegalStateException(
                                "POCKET-TTS-G2P-003: خروجی decoder از نوع Tensor نیست؛ " +
                                    "outputClass=${outputValue.javaClass.name}, " +
                                    "outputNames=${decoder.outputNames.joinToString()}"
                            )
                        val shape = outputTensor.info.shape
                        require(shape.size == 3 && shape[0] == 1L && shape[1] >= generated.size) {
                            "ابعاد logits نامعتبر است: ${shape.contentToString()}، decoderTokens=${generated.size}"
                        }
                        require(outputTensor.info.type == ai.onnxruntime.OnnxJavaType.FLOAT) {
                            "نوع logits پشتیبانی نمی‌شود: ${outputTensor.info.type}"
                        }
                        val buffer = outputTensor.floatBuffer.duplicate()
                        val values = FloatArray(buffer.remaining())
                        buffer.get(values)
                        val vocab = shape[2].toInt()
                        // The next token is predicted from the last supplied decoder token.
                        val lastOffset = (generated.size - 1) * vocab
                        require(vocab > 0 && lastOffset >= 0 && lastOffset + vocab <= values.size) {
                            "خروجی logits ناقص است: shape=${shape.contentToString()}, values=${values.size}"
                        }
                        values.copyOfRange(lastOffset, lastOffset + vocab)
                    }
                } finally {
                    ids.close()
                }
                var bestId = 0
                var best = Float.NEGATIVE_INFINITY
                logits.forEachIndexed { index, score ->
                    if (score > best) { best = score; bestId = index }
                }
                if (bestId == EOS_ID) return decodeBytes(generated.drop(1))
                generated += bestId.toLong()
            }
            throw IllegalStateException("تولید واج از سقف $MAX_DECODE_TOKENS توکن عبور کرد.")
        } finally {
            hidden.close()
        }
    }

    /** Copy the encoder hidden-state tensor before its Result is closed. */
    private fun copyFloatTensor(source: OnnxTensor): OnnxTensor {
        require(source.info.type == ai.onnxruntime.OnnxJavaType.FLOAT) {
            "نوع hidden-state پشتیبانی نمی‌شود: ${source.info.type}"
        }
        val shape = source.info.shape
        val buffer = source.floatBuffer.duplicate()
        val values = FloatArray(buffer.remaining())
        buffer.get(values)
        return OnnxTensor.createTensor(env, java.nio.FloatBuffer.wrap(values), shape)
    }

    private fun tensor(values: LongArray): OnnxTensor {
        val buffer = LongBuffer.allocate(values.size)
        buffer.put(values)
        buffer.flip()
        return OnnxTensor.createTensor(env, buffer, longArrayOf(1L, values.size.toLong()))
    }

    private fun decodeBytes(ids: List<Long>): String {
        val bytes = ids.filter { it in 3L..258L }.map { (it - 3L).toByte() }.toByteArray()
        return bytes.toString(Charsets.UTF_8)
    }

    private fun transliterate(raw: String): String = LATIN_WORD.replace(raw) { match ->
        val word = match.value
        if ('-' in word.trim('-')) {
            val parts = word.split('-').filter(String::isNotEmpty)
            if (parts.size > 1) return@replace parts.joinToString(" ") { transliterate(it) }
        }
        val lower = word.lowercase(java.util.Locale.ROOT)
        LATIN_EXCEPTIONS[lower] ?: if (word.length in 2..5 && word.all { it.isUpperCase() }) {
            word.lowercase(java.util.Locale.ROOT).map { LETTER_NAMES[it] ?: "" }.joinToString("\u200c")
        } else {
            val out = StringBuilder()
            var i = 0
            while (i < lower.length) {
                val pair = lower.substring(i, minOf(i + 2, lower.length))
                val digraph = LATIN_DIGRAPHS[pair]
                if (digraph != null) { out.append(digraph); i += 2 }
                else { out.append(LATIN_CHARS[lower[i]] ?: ""); i++ }
            }
            var result = out.toString()
            if (result.length >= 2 && result[0] == 'س' && result[1] in CLUSTER_START) result = "ا$result"
            result.ifBlank { word }
        }
    }

    /**
     * Kotlin port of normalize_fa.normalize_for_model() from the upstream
     * ParSiGo/Pocket-TTS Farsi v2 pipeline. Keep this before ByT5 encoding:
     * digits and Arabic letter variants are otherwise silently lost by G2P.
     */
    private fun normalize(raw: String): String {
        if (raw.none { it in PERSIAN_LETTERS } && raw.any { it.code < 128 && it.isLetter() }) {
            return raw.trim() // upstream normalize_for_model passes phoneme strings through
        }
        var text = java.text.Normalizer.normalize(raw, java.text.Normalizer.Form.NFC)
        text = text.replace(INVISIBLES, "")
            .replace(Regex("[\\u064B-\\u0655\\u0670\\u06D6-\\u06ED]"), "")
        text = text.map { CHAR_MAP[it] ?: it }.joinToString("")
        text = text.map { ch ->
            when (ch) {
                in '۰'..'۹' -> ('0'.code + (ch.code - '۰'.code)).toChar()
                in '٠'..'٩' -> ('0'.code + (ch.code - '٠'.code)).toChar()
                else -> ch
            }
        }.joinToString("")
            .replace("٬", "")
            .replace("٫", ".")
            .replace(Regex("[٪%]"), " درصد ")
        text = spellNumbers(text)
        text = text.map { PUNCT_MAP[it] ?: it }.joinToString("")
        text = text.map { ch ->
            if (ch in PERSIAN_LETTERS || ch == '\u200c' || ch in KEPT_PUNCT || ch.isWhitespace()) ch else ' '
        }.joinToString("")
        text = text.replace(Regex("\\u200c+"), "\u200c")
            .replace(Regex("\\s*\\u200c\\s*"), " ")
            .replace(Regex("(?<![$PERSIAN_LETTERS])\\u200c|\\u200c(?![$PERSIAN_LETTERS])"), "")
            .replace(Regex("\\s+([.،؛؟!:])"), "$1")
            .replace(Regex("([.،؛؟!:])\\1+"), "$1")
            .replace(Regex("\\s+"), " ")
            .trim()
            .trimStart('.', '،', '؛', '؟', '!', ':', ' ')
        return text.replace(Regex("[؟?:]"), "").replace(Regex("\\s+"), " ").trim()
    }

    private fun spellNumbers(input: String): String {
        val compact = input.replace(Regex("(?<=\\d),(?=\\d{3}\\b)"), "")
        return Regex("(\\d+)(?:[.](\\d+))?").replace(compact) { match ->
            val whole = match.groupValues[1]
            var fraction = match.groupValues.getOrElse(2) { "" }
            if (whole.length > 12 || (whole.startsWith('0') && whole.length > 1)) {
                val digits = whole.map(::digitWord).joinToString(" ")
                val frac = if (fraction.isEmpty()) "" else " " + fraction.map(::digitWord).joinToString(" ")
                " $digits$frac "
            } else {
                var words = numberToWords((whole.trimStart('0').ifEmpty { "0" }).toLong())
                fraction = fraction.trimEnd('0')
                if (fraction.isNotEmpty()) {
                    val unit = when (fraction.length) { 1 -> "دهم"; 2 -> "صدم"; 3 -> "هزارم"; else -> "" }
                    words += if (unit.isNotEmpty()) " ممیز ${numberToWords(fraction.toLong())} $unit"
                    else " ممیز " + fraction.map(::digitWord).joinToString(" ")
                }
                " $words "
            }
        }
    }

    private fun digitWord(digit: Char): String = when (digit) {
        '0' -> "صفر"; '1' -> "یک"; '2' -> "دو"; '3' -> "سه"; '4' -> "چهار"
        '5' -> "پنج"; '6' -> "شش"; '7' -> "هفت"; '8' -> "هشت"; else -> "نه"
    }

    private fun numberToWords(value: Long): String {
        if (value == 0L) return "صفر"
        val ones = listOf("", "یک", "دو", "سه", "چهار", "پنج", "شش", "هفت", "هشت", "نه")
        val teens = listOf("ده", "یازده", "دوازده", "سیزده", "چهارده", "پانزده", "شانزده", "هفده", "هجده", "نوزده")
        val tens = listOf("", "", "بیست", "سی", "چهل", "پنجاه", "شصت", "هفتاد", "هشتاد", "نود")
        val hundreds = listOf("", "صد", "دویست", "سیصد", "چهارصد", "پانصد", "ششصد", "هفتصد", "هشتصد", "نهصد")
        fun underThousand(number: Int): String {
            var n = number
            val parts = mutableListOf<String>()
            if (n >= 100) { parts += hundreds[n / 100]; n %= 100 }
            if (n >= 20) { parts += tens[n / 10]; n %= 10 }
            if (n in 10..19) { parts += teens[n - 10]; n = 0 }
            if (n > 0) parts += ones[n]
            return parts.joinToString(" و ")
        }
        var n = value
        val parts = mutableListOf<String>()
        val scales = listOf(1_000_000_000_000L to "تریلیون", 1_000_000_000L to "میلیارد", 1_000_000L to "میلیون", 1_000L to "هزار")
        for ((scale, name) in scales) {
            if (n >= scale) {
                val count = n / scale
                n %= scale
                val head = if (count == 1L && scale == 1_000L) "" else underThousand(count.toInt()) + " "
                parts += (head + name).trim()
            }
        }
        if (n > 0) parts += underThousand(n.toInt())
        return parts.joinToString(" و ")
    }

    private fun splitWindows(text: String): List<String> {
        val words = text.split(Regex("\\s+")).filter { it.isNotBlank() }
        if (words.size <= WINDOW_WORDS) return listOf(text)
        val cut = (WINDOW_WORDS / 2 until minOf(words.size - 4, WINDOW_WORDS - 3))
            .lastOrNull { words[it] in CONJUNCTIONS } ?: WINDOW_WORDS / 2
        return splitWindows(words.take(cut).joinToString(" ")) +
            splitWindows(words.drop(cut).joinToString(" "))
    }

    override fun close() {
        runCatching { encoder.close() }
        runCatching { decoder.close() }
        options.close()
    }

    companion object {
        private const val EOS_ID = 1
        private const val MAX_DECODE_TOKENS = 512
        private const val WINDOW_WORDS = 30
        private val CONJUNCTIONS = setOf("و", "یا", "ولی", "اما", "که", "زیرا", "چون", "پس", "سپس", "بنابراین", "همچنین")
        private val PHONEME_REPLACEMENTS = mapOf('/' to 'a', 'a' to 'A', '@' to '?', '
        private val LATIN_WORD = Regex("[A-Za-z][A-Za-z'-]*")
        private val LATIN_EXCEPTIONS = mapOf(
            "echo" to "اکو", "state" to "استیت", "network" to "نت‌ورک",
            "windows" to "ویندوز", "notebook" to "نوت‌بوک", "photoshop" to "فتوشاپ",
            "python" to "پایتون", "recurrency" to "ریکارانسی"
        )
        private val LATIN_DIGRAPHS = mapOf(
            "sh" to "ش", "ch" to "چ", "th" to "ث", "ph" to "ف", "gh" to "غ",
            "kh" to "خ", "wh" to "و", "ck" to "ک", "qu" to "کو", "oo" to "و",
            "ee" to "ی", "ea" to "ی", "ou" to "او", "au" to "او", "ai" to "ای",
            "ay" to "ای", "ey" to "ای", "oi" to "اوی", "oy" to "اوی"
        )
        private val LATIN_CHARS = mapOf(
            'a' to "ا", 'b' to "ب", 'c' to "ک", 'd' to "د", 'e' to "", 'f' to "ف",
            'g' to "گ", 'h' to "ه", 'i' to "ی", 'j' to "ج", 'k' to "ک", 'l' to "ل",
            'm' to "م", 'n' to "ن", 'o' to "او", 'p' to "پ", 'q' to "ق", 'r' to "ر",
            's' to "س", 't' to "ت", 'u' to "و", 'v' to "و", 'w' to "و", 'x' to "کس",
            'y' to "ی", 'z' to "ز"
        )
        private val LETTER_NAMES = mapOf(
            'a' to "ای", 'b' to "بی", 'c' to "سی", 'd' to "دی", 'e' to "ای",
            'f' to "اف", 'g' to "جی", 'h' to "اچ", 'i' to "آی", 'j' to "جی",
            'k' to "کی", 'l' to "ال", 'm' to "ام", 'n' to "ان", 'o' to "او",
            'p' to "پی", 'q' to "کیو", 'r' to "آر", 's' to "اس", 't' to "تی",
            'u' to "یو", 'v' to "وی", 'w' to "دبلیو", 'x' to "ایکس", 'y' to "وای", 'z' to "زد"
        )
        private const val PERSIAN_LETTERS = "آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهیئ"
        private val INVISIBLES = Regex("[\\u200b\\u200d\\u200e\\u200f\\u202a-\\u202e\\u2066-\\u2069\\ufeff\\u00ad]")
        private val CHAR_MAP = mapOf(
            'ي' to 'ی', 'ى' to 'ی', 'ۍ' to 'ی', 'ې' to 'ی', 'ك' to 'ک', 'ڪ' to 'ک',
            'ة' to 'ه', 'ۀ' to 'ه', 'أ' to 'ا', 'إ' to 'ا', 'ٱ' to 'ا', 'ٲ' to 'ا',
            'ٳ' to 'ا', 'ؤ' to 'و', 'ۂ' to 'ه', 'ۃ' to 'ه', 'ء' to ' ', 'ـ' to ' '
        )
        private val PUNCT_MAP = mapOf(
            ',' to '،', '?' to '؟', ';' to '؛', '٬' to ' ', '٫' to '.', '…' to '.',
            '»' to ' ', '«' to ' ', '“' to ' ', '”' to ' ', '‘' to ' ', '’' to ' ',
            '"' to ' ', '\'' to ' ', '`' to ' ', '(' to ' ', ')' to ' ', '[' to ' ',
            ']' to ' ', '{' to ' ', '}' to ' ', '–' to ' ', '—' to ' ', '−' to ' ',
            '-' to ' ', '‐' to ' ', '/' to ' ', '\\\\' to ' ', '*' to ' ', '_' to ' ',
            '|' to ' ', '×' to ' ', '=' to ' ', '+' to ' '
        )
        private val KEPT_PUNCT = setOf('.', '،', '؛', '؟', '!', ':')
        private val CLUSTER_START = setOf('پ','ت','ک','ب','ج','چ','ذ','ژ','ز','ص','ض','ث','ف','گ','س','ش')
    }
}
 to 'S', 'c' to 'C')
        private val PRONUNCIATION_FIXES = mapOf("bo?d" to "ba?d")
        private val LATIN_WORD = Regex("[A-Za-z][A-Za-z'-]*")
        private val LATIN_EXCEPTIONS = mapOf(
            "echo" to "اکو", "state" to "استیت", "network" to "نت‌ورک",
            "windows" to "ویندوز", "notebook" to "نوت‌بوک", "photoshop" to "فتوشاپ",
            "python" to "پایتون", "recurrency" to "ریکارانسی"
        )
        private val LATIN_DIGRAPHS = mapOf(
            "sh" to "ش", "ch" to "چ", "th" to "ث", "ph" to "ف", "gh" to "غ",
            "kh" to "خ", "wh" to "و", "ck" to "ک", "qu" to "کو", "oo" to "و",
            "ee" to "ی", "ea" to "ی", "ou" to "او", "au" to "او", "ai" to "ای",
            "ay" to "ای", "ey" to "ای", "oi" to "اوی", "oy" to "اوی"
        )
        private val LATIN_CHARS = mapOf(
            'a' to "ا", 'b' to "ب", 'c' to "ک", 'd' to "د", 'e' to "", 'f' to "ف",
            'g' to "گ", 'h' to "ه", 'i' to "ی", 'j' to "ج", 'k' to "ک", 'l' to "ل",
            'm' to "م", 'n' to "ن", 'o' to "او", 'p' to "پ", 'q' to "ق", 'r' to "ر",
            's' to "س", 't' to "ت", 'u' to "و", 'v' to "و", 'w' to "و", 'x' to "کس",
            'y' to "ی", 'z' to "ز"
        )
        private val LETTER_NAMES = mapOf(
            'a' to "ای", 'b' to "بی", 'c' to "سی", 'd' to "دی", 'e' to "ای",
            'f' to "اف", 'g' to "جی", 'h' to "اچ", 'i' to "آی", 'j' to "جی",
            'k' to "کی", 'l' to "ال", 'm' to "ام", 'n' to "ان", 'o' to "او",
            'p' to "پی", 'q' to "کیو", 'r' to "آر", 's' to "اس", 't' to "تی",
            'u' to "یو", 'v' to "وی", 'w' to "دبلیو", 'x' to "ایکس", 'y' to "وای", 'z' to "زد"
        )
        private val CLUSTER_START = setOf('پ','ت','ک','ب','ج','چ','ذ','ژ','ز','ص','ض','ث','ف','گ','س','ش')
    }
}
