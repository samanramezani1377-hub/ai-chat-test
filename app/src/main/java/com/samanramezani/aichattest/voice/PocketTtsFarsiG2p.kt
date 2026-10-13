package com.samanramezani.aichattest.voice

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OnnxSequence
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
            .map { PHONEME_REPLACEMENTS[it] ?: it }
            .joinToString("")
            .replace(Regex("\\s+"), " ")
            .trim()
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
                val output = result[0].value
                try {
                    copyHiddenTensor(output)
                } catch (t: Throwable) {
                    throw IllegalStateException(
                        "POCKET-TTS-G2P-002: خروجی رمزگذار قابل تبدیل به Tensor نیست؛ " +
                            "output index 0, valueClass=${output.javaClass.name}: ${t.message}",
                        t
                    )
                }
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
                        val tensor = result[0].value as? OnnxTensor ?: error("خروجی decoder از نوع Tensor نیست")
                        val shape = tensor.info.shape
                        require(shape.size == 3 && shape[0] == 1L && shape[1] >= generated.size) {
                            "ابعاد logits نامعتبر است: ${shape.contentToString()}"
                        }
                        val values = FloatArray(tensor.floatBuffer.remaining())
                        tensor.floatBuffer.get(values)
                        val vocab = shape[2].toInt()
                        val lastOffset = (shape[1].toInt() - 1) * vocab
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

    /**
     * Materialize an independent tensor from the Java value representation returned by
     * ONNX Runtime. This handles direct tensors, sequence-wrapped values, and primitive
     * multidimensional arrays such as float[][][] (whose JVM class name is [[[F).
     */
    private fun copyHiddenTensor(value: Any): OnnxTensor = when (value) {
        is OnnxTensor -> OnnxTensor.createTensor(env, value.value)
        is OnnxSequence -> {
            val first = value.value.firstOrNull()
                ?: error("خروجی sequence رمزگذار خالی است")
            copyHiddenTensor(first)
        }
        is Array<*> -> OnnxTensor.createTensor(env, value)
        else -> error("نمایش خروجی برای ساخت Tensor پشتیبانی نمی‌شود: ${value.javaClass.name}")
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

    private fun normalize(raw: String): String = raw
        .replace('ي', 'ی')
        .replace('ى', 'ی')
        .replace('ك', 'ک')
        .replace('ۀ', 'ه')
        .replace('ة', 'ه')
        .replace('\u0640'.toString(), "")
        .replace(Regex("[\\u064B-\\u065F\\u0670]"), "")
        .replace(Regex("[؟?:]+"), " ")
        .replace(Regex("\\s+"), " ")
        .trim()

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
        private val PHONEME_REPLACEMENTS = mapOf('/' to 'a', 'a' to 'A', '@' to '?', '$' to 'S', 'c' to 'C')
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
