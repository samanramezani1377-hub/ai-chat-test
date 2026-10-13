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
        val text = normalize(input)
        if (text.isBlank()) return ""
        val windows = splitWindows(text)
        return windows.mapNotNull { window -> decodeWindow(window).takeIf { it.isNotBlank() } }
            .joinToString(" ")
            .replace('/', 'a')
            .replace('@', '?')
            .replace('$', 'S')
            .replace('c', 'C')
            .replace(Regex("\\s+"), " ")
            .trim()
    }

    private fun decodeWindow(text: String): String {
        val bytes = text.toByteArray(Charsets.UTF_8)
        if (bytes.isEmpty()) return ""
        val inputIds = LongArray(bytes.size) { (bytes[it].toInt() and 0xff).toLong() + 3L }
        val encoded = tensor(inputIds)
        val hidden = try {
            encoder.run(mapOf("input_ids" to encoded)).use { result ->
                (result[0].value as? OnnxTensor)
                    ?: error("خروجی encoder از نوع Tensor نیست")
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
    }
}
