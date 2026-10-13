package com.samanramezani.aichattest.voice

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.LongBuffer
import java.util.Locale
import kotlin.math.exp

/**
 * Runs a fine-tuned sequence-classification ONNX model on-device.
 *
 * Bundle contract:
 * - model.onnx: logits output [1, number_of_labels]
 * - tokenizer.json: Hugging Face fast-tokenizer JSON with a WordPiece model vocab
 * - labels.json: {"labels":["CHAT","GET_TIME",...]} in the same order as model logits,
 *   or {"0":"CHAT","1":"GET_TIME",...}.
 *
 * The generic MiniLM embedding model is not sufficient; model.onnx must have a
 * sequence-classification head trained for these intent labels.
 */
internal class MiniLmOnnxIntentClassifier(private val root: File) : AutoCloseable {
    companion object {
        private const val MAX_LENGTH = 64
        private const val MIN_CONFIDENCE = 0.78
        private const val MIN_MARGIN = 0.18
        private val ALLOWED_LABELS = setOf(
            "CHAT", "GET_TIME", "GET_DATE", "INCREASE_VOLUME", "DECREASE_VOLUME",
            "SET_VOLUME", "INCREASE_BRIGHTNESS", "DECREASE_BRIGHTNESS", "SET_BRIGHTNESS",
        )
    }

    private val lock = Any()
    @Volatile private var cachedModelPath: String? = null
    private var environment: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var tokenizer: WordPieceTokenizer? = null
    private var labels: List<String> = emptyList()

    fun isReady(): Boolean = File(root, "model.onnx").isFile &&
        File(root, "tokenizer.json").isFile && File(root, "labels.json").isFile

    /** Returns a high-confidence model prediction, or null to use the safe rules fallback. */
    fun classify(text: String): DinaCommand? = synchronized(lock) {
        if (!isReady() || text.isBlank()) return@synchronized null
        try {
            ensureLoaded()
            val env = environment ?: return@synchronized null
            val activeSession = session ?: return@synchronized null
            val activeTokenizer = tokenizer ?: return@synchronized null
            val encoded = activeTokenizer.encode(text, MAX_LENGTH)
            val inputs = linkedMapOf<String, OnnxTensor>()
            try {
                activeSession.inputNames.forEach { inputName ->
                    val tensor = when {
                        inputName.contains("input_ids", ignoreCase = true) ->
                            OnnxTensor.createTensor(env, LongBuffer.wrap(encoded.ids), longArrayOf(1, MAX_LENGTH.toLong()))
                        inputName.contains("attention_mask", ignoreCase = true) ->
                            OnnxTensor.createTensor(env, LongBuffer.wrap(encoded.mask), longArrayOf(1, MAX_LENGTH.toLong()))
                        inputName.contains("token_type_ids", ignoreCase = true) ->
                            OnnxTensor.createTensor(env, LongBuffer.wrap(encoded.typeIds), longArrayOf(1, MAX_LENGTH.toLong()))
                        else -> return@synchronized null
                    }
                    inputs[inputName] = tensor
                }
                if (inputs.none { it.key.contains("input_ids", ignoreCase = true) } ||
                    inputs.none { it.key.contains("attention_mask", ignoreCase = true) }) {
                    return@synchronized null
                }
                activeSession.run(inputs).use { result ->
                    val output = result.firstOrNull()?.value as? OnnxTensor ?: return@synchronized null
                    val buffer = output.floatBuffer
                    if (buffer.remaining() != labels.size || labels.isEmpty()) return@synchronized null
                    val logits = FloatArray(buffer.remaining())
                    buffer.get(logits)
                    val maxLogit = logits.maxOrNull() ?: return@synchronized null
                    val expScores = logits.map { exp((it - maxLogit).toDouble()) }
                    val sum = expScores.sum().takeIf { it > 0.0 } ?: return@synchronized null
                    val ranked = logits.indices.sortedByDescending { expScores[it] / sum }
                    val best = ranked.first()
                    val confidence = expScores[best] / sum
                    val second = ranked.getOrNull(1)?.let { expScores[it] / sum } ?: 0.0
                    if (confidence < MIN_CONFIDENCE || confidence - second < MIN_MARGIN) return@synchronized null
                    commandFor(labels[best], text)
                }
            } finally {
                inputs.values.forEach { runCatching { it.close() } }
            }
        } catch (_: Throwable) {
            // Bad/unsupported model bundles must never break the voice conversation.
            null
        }
    }

    private fun commandFor(label: String, text: String): DinaCommand? {
        if (label !in ALLOWED_LABELS) return null
        if (label == "CHAT") return DinaCommand(DinaIntent.CHAT)
        val parsed = DinaCommandClassifier.classify(text)
        val modelIntent = runCatching { DinaIntent.valueOf(label) }.getOrNull() ?: return null
        // Absolute values must be grounded in the actual utterance, not fabricated by the model.
        val value = when (modelIntent) {
            DinaIntent.SET_VOLUME, DinaIntent.SET_BRIGHTNESS -> parsed.value ?: return null
            else -> null
        }
        return DinaCommand(modelIntent, value, modelIntent == DinaIntent.SET_BRIGHTNESS || parsed.valueIsPercent)
    }

    private fun ensureLoaded() {
        val model = File(root, "model.onnx")
        if (session != null && cachedModelPath == model.absolutePath + ":" + model.lastModified() + ":" + model.length()) return
        closeLoaded()
        val tokenizerJson = JSONObject(File(root, "tokenizer.json").readText(Charsets.UTF_8))
        val modelJson = tokenizerJson.optJSONObject("model") ?: error("tokenizer.json فاقد بخش model است.")
        require(modelJson.optString("type") == "WordPiece") {
            "MINILM-INFER-001: tokenizer فعلاً باید از نوع WordPiece باشد."
        }
        val vocabObject = modelJson.optJSONObject("vocab") ?: error("واژگان WordPiece پیدا نشد.")
        val vocab = HashMap<String, Int>(vocabObject.length())
        val keys = vocabObject.keys()
        while (keys.hasNext()) {
            val key = keys.next()
            vocab[key] = vocabObject.getInt(key)
        }
        val addedTokens = tokenizerJson.optJSONArray("added_tokens") ?: JSONArray()
        var cls = "[CLS]"
        var sep = "[SEP]"
        var unk = modelJson.optString("unk_token", "[UNK]")
        var pad = "[PAD]"
        for (i in 0 until addedTokens.length()) {
            val token = addedTokens.optJSONObject(i) ?: continue
            when (token.optString("content")) {
                "[CLS]" -> cls = token.optString("content")
                "[SEP]" -> sep = token.optString("content")
                "[UNK]" -> unk = token.optString("content")
                "[PAD]" -> pad = token.optString("content")
            }
        }
        require(vocab.containsKey(cls) && vocab.containsKey(sep) && vocab.containsKey(unk) && vocab.containsKey(pad)) {
            "MINILM-INFER-002: واژگان باید توکن‌های CLS، SEP، UNK و PAD را داشته باشد."
        }
        tokenizer = WordPieceTokenizer(vocab, cls, sep, unk, pad)
        labels = parseLabels(File(root, "labels.json").readText(Charsets.UTF_8))
        require(labels.isNotEmpty() && labels.all { it in ALLOWED_LABELS } && labels.contains("CHAT")) {
            "MINILM-INFER-003: labels.json باید فقط برچسب‌های معتبر و CHAT را شامل شود."
        }
        val env = OrtEnvironment.getEnvironment()
        val options = OrtSession.SessionOptions()
        try {
            options.setIntraOpNumThreads(2)
            options.setInterOpNumThreads(1)
            session = env.createSession(model.absolutePath, options)
        } finally {
            options.close()
        }
        environment = env
        cachedModelPath = model.absolutePath + ":" + model.lastModified() + ":" + model.length()
        val outputCount = session?.outputInfo?.size ?: 0
        require(outputCount > 0) { "MINILM-INFER-004: خروجی مدل ONNX پیدا نشد." }
    }

    private fun parseLabels(raw: String): List<String> {
        val json = raw.trim()
        if (json.startsWith("[")) {
            val array = JSONArray(json)
            return (0 until array.length()).map { array.getString(it).uppercase(Locale.ROOT) }
        }
        val objectJson = JSONObject(json)
        val array = objectJson.optJSONArray("labels")
        if (array != null) return (0 until array.length()).map { array.getString(it).uppercase(Locale.ROOT) }
        return (0 until objectJson.length()).map { objectJson.optString(it.toString()).uppercase(Locale.ROOT) }
    }

    private fun closeLoaded() {
        runCatching { session?.close() }
        session = null
        tokenizer = null
        labels = emptyList()
        cachedModelPath = null
        // OrtEnvironment is a process-wide singleton; do not close it here.
        environment = null
    }

    override fun close() = synchronized(lock) { closeLoaded() }

    private data class Encoded(val ids: LongArray, val mask: LongArray, val typeIds: LongArray)

    private class WordPieceTokenizer(
        private val vocab: Map<String, Int>,
        private val clsToken: String,
        private val sepToken: String,
        private val unkToken: String,
        private val padToken: String,
    ) {
        fun encode(input: String, maxLength: Int): Encoded {
            val tokens = ArrayList<String>()
            tokens += clsToken
            for (basic in basicTokenize(input)) {
                val pieces = wordPiece(basic)
                tokens.addAll(pieces)
                if (tokens.size >= maxLength - 1) break
            }
            tokens += sepToken
            val ids = LongArray(maxLength) { vocab.getValue(padToken).toLong() }
            val mask = LongArray(maxLength)
            val types = LongArray(maxLength)
            for (i in 0 until minOf(tokens.size, maxLength)) {
                ids[i] = (vocab[tokens[i]] ?: vocab.getValue(unkToken)).toLong()
                mask[i] = 1L
            }
            return Encoded(ids, mask, types)
        }

        private fun basicTokenize(input: String): List<String> {
            val out = ArrayList<String>()
            val current = StringBuilder()
            fun flush() {
                if (current.isNotEmpty()) {
                    out += current.toString()
                    current.setLength(0)
                }
            }
            for (char in input.lowercase(Locale.ROOT)) {
                when {
                    char.isWhitespace() || char.isISOControl() -> flush()
                    char.isPunctuationLike() -> {
                        flush()
                        out += char.toString()
                    }
                    else -> current.append(char)
                }
            }
            flush()
            return out
        }

        private fun Char.isPunctuationLike(): Boolean =
            this in ".,!?;:()[]{}<>/\\\"'،؛؟٪٬٫ـ—–…@#$%^&*+=|"

        private fun wordPiece(word: String): List<String> {
            if (word.isEmpty()) return emptyList()
            if (word.length > 100) return listOf(unkToken)
            val pieces = ArrayList<String>()
            var start = 0
            while (start < word.length) {
                var end = word.length
                var found: String? = null
                while (start < end) {
                    val candidate = (if (start == 0) "" else "##") + word.substring(start, end)
                    if (vocab.containsKey(candidate)) {
                        found = candidate
                        break
                    }
                    end--
                }
                if (found == null) return listOf(unkToken)
                pieces += found
                start = end
            }
            return pieces
        }
    }
}
