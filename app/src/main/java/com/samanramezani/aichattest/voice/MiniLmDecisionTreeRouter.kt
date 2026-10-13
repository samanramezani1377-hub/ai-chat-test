package com.samanramezani.aichattest.voice

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale
import kotlin.math.sqrt

/**
 * Runs an imported sentence-embedding MiniLM against a bounded, staged decision tree.
 *
 * IMPORTANT: all-MiniLM-L6-v2 is English-trained, not multilingual. Persian/Arabic-script
 * utterances deliberately return null so the caller can use Dina's Persian-safe rules instead
 * of executing a command from meaningless cross-lingual similarity scores.
 *
 * Bundle: model.onnx + tokenizer.json + decision_trees.json.
 * decision_trees.json stores candidate descriptions and examples for each decision stage.
 */
internal class MiniLmDecisionTreeRouter(private val root: File) : AutoCloseable {
    companion object {
        private const val MAX_LENGTH = 64
        private const val DEFAULT_MIN_SIMILARITY = 0.42
        private const val DEFAULT_MIN_MARGIN = 0.035
        private const val EMBEDDING_SIZE = 384
    }

    @Volatile var lastError: String? = null
        private set

    private val lock = Any()
    private var environment: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var vocab: Map<String, Int> = emptyMap()
    private var clsToken = "[CLS]"
    private var sepToken = "[SEP]"
    private var unkToken = "[UNK]"
    private var padToken = "[PAD]"
    private var nodes: JSONObject? = null
    private var minSimilarity = DEFAULT_MIN_SIMILARITY
    private var minMargin = DEFAULT_MIN_MARGIN
    private val candidateCache = HashMap<String, FloatArray>()

    fun isReady(): Boolean = File(root, "model.onnx").isFile &&
        File(root, "tokenizer.json").isFile && File(root, "decision_trees.json").isFile

    fun classify(rawText: String): DinaCommand? = synchronized(lock) {
        if (!isReady() || rawText.isBlank()) return@synchronized null
        // all-MiniLM-L6-v2 is an English model. Persian tokens mostly become unknown pieces,
        // so cosine ranking is not semantically valid for Persian input.
        if (rawText.any { it in '\u0600'..'\u06FF' || it in '\u0750'..'\u077F' }) {
            lastError = "MINILM-TREE-003: مدل all-MiniLM-L6-v2 انگلیسی است؛ فرمان فارسی به قواعد امن دینا سپرده شد."
            return@synchronized null
        }
        try {
            ensureLoaded()
            val query = embed(rawText)
            val domain = choose("domain", query) ?: return@synchronized null
            val command = when (domain) {
                "GET_TIME" -> DinaCommand(DinaIntent.GET_TIME)
                "GET_DATE" -> DinaCommand(DinaIntent.GET_DATE)
                "AUDIO" -> routeDeviceCommand("audio", query, rawText)
                "DISPLAY" -> routeDeviceCommand("display", query, rawText)
                "CHAT" -> DinaCommand(DinaIntent.CHAT)
                else -> null
            }
            if (command != null) lastError = null
            command
        } catch (t: Throwable) {
            lastError = "MINILM-TREE-001: ${t.message ?: t.javaClass.simpleName}"
            null
        }
    }

    private fun routeDeviceCommand(node: String, query: FloatArray, rawText: String): DinaCommand? {
        val operation = choose(node, query) ?: return null
        // Numeric values are grounded in the utterance by the existing deterministic Persian-safe
        // parser; embeddings are never allowed to invent a target value.
        val parsed = DinaCommandClassifier.classify(rawText)
        return when (operation) {
            "INCREASE_VOLUME" -> DinaCommand(DinaIntent.INCREASE_VOLUME)
            "DECREASE_VOLUME" -> DinaCommand(DinaIntent.DECREASE_VOLUME)
            "SET_VOLUME" -> {
                // The decision tree selects levels 1-10 itself; no number extraction is needed
                // for this common scale. Explicit percentages/other scales retain safe parsing.
                val level = if (nodes?.optJSONObject("volume_level") != null) choose("volume_level", query) else null
                val selected = level?.takeIf { it.startsWith("LEVEL_") }
                    ?.removePrefix("LEVEL_")?.toIntOrNull()
                when {
                    selected != null && selected in 1..10 -> DinaCommand(DinaIntent.SET_VOLUME, selected * 10, valueIsPercent = true)
                    level == "OTHER_EXPLICIT_VALUE" || level == null ->
                        parsed.takeIf { it.intent == DinaIntent.SET_VOLUME && it.value != null }
                    else -> null
                }
            }
            "MUTE" -> DinaCommand(DinaIntent.MUTE)
            "UNMUTE" -> DinaCommand(DinaIntent.UNMUTE)
            "INCREASE_BRIGHTNESS" -> DinaCommand(DinaIntent.INCREASE_BRIGHTNESS)
            "DECREASE_BRIGHTNESS" -> DinaCommand(DinaIntent.DECREASE_BRIGHTNESS)
            "SET_BRIGHTNESS" -> parsed.takeIf { it.intent == DinaIntent.SET_BRIGHTNESS && it.value != null }
            else -> null
        }
    }

    private fun choose(nodeId: String, query: FloatArray): String? {
        val node = nodes?.optJSONObject(nodeId) ?: error("درخت تصمیم فاقد گره '$nodeId' است.")
        val choices = node.optJSONArray("choices") ?: error("گره '$nodeId' گزینه ندارد.")
        val ranked = ArrayList<Pair<String, Float>>(choices.length())
        for (i in 0 until choices.length()) {
            val choice = choices.getJSONObject(i)
            val id = choice.getString("id")
            // Embed one compact prototype per option instead of running ONNX once per example.
            // This keeps the first decision fast while retaining examples as semantic anchors.
            val prototype = buildString {
                append(choice.optString("description", id.replace('_', ' ').lowercase(Locale.ROOT)))
                val examples = choice.optJSONArray("examples") ?: JSONArray()
                for (j in 0 until minOf(examples.length(), 3)) {
                    append(". ")
                    append(examples.getString(j))
                }
            }
            val vector = candidateCache.getOrPut("$nodeId:$id:$prototype") { embed(prototype) }
            ranked += id to cosine(query, vector)
        }
        val ordered = ranked.sortedByDescending { it.second }
        val best = ordered.firstOrNull() ?: return null
        val second = ordered.getOrNull(1)?.second ?: -1f
        if (best.second < minSimilarity || best.second - second < minMargin) {
            lastError = "MINILM-TREE-002: تصمیم نامطمئن در گره '$nodeId' (similarity=${"%.3f".format(Locale.US, best.second)}, margin=${"%.3f".format(Locale.US, best.second - second)})."
            return null
        }
        return best.first
    }

    private fun ensureLoaded() {
        if (session != null && nodes != null) return
        val tokenizerJson = JSONObject(File(root, "tokenizer.json").readText(Charsets.UTF_8))
        val model = tokenizerJson.optJSONObject("model") ?: error("tokenizer.json فاقد model است.")
        require(model.optString("type") == "WordPiece") {
            "فقط tokenizer از نوع WordPiece برای all-MiniLM-L6-v2 پشتیبانی می‌شود."
        }
        val vocabulary = model.optJSONObject("vocab") ?: error("واژگان WordPiece پیدا نشد.")
        val parsedVocab = HashMap<String, Int>(vocabulary.length())
        val keys = vocabulary.keys()
        while (keys.hasNext()) {
            val key = keys.next()
            parsedVocab[key] = vocabulary.getInt(key)
        }
        vocab = parsedVocab
        clsToken = if ("[CLS]" in vocab) "[CLS]" else error("توکن [CLS] پیدا نشد.")
        sepToken = if ("[SEP]" in vocab) "[SEP]" else error("توکن [SEP] پیدا نشد.")
        unkToken = if ("[UNK]" in vocab) "[UNK]" else error("توکن [UNK] پیدا نشد.")
        padToken = if ("[PAD]" in vocab) "[PAD]" else error("توکن [PAD] پیدا نشد.")

        val config = JSONObject(File(root, "decision_trees.json").readText(Charsets.UTF_8))
        require(config.optString("model_type") == "sentence_embedding") {
            "decision_trees.json باید model_type=sentence_embedding داشته باشد."
        }
        nodes = config.optJSONObject("nodes") ?: error("decision_trees.json فاقد nodes است.")
        minSimilarity = config.optDouble("min_similarity", DEFAULT_MIN_SIMILARITY)
        minMargin = config.optDouble("min_margin", DEFAULT_MIN_MARGIN)
        require(minSimilarity in -1.0..1.0 && minMargin >= 0.0) { "آستانه‌های شباهت نامعتبرند." }
        for (nodeId in listOf("domain", "audio", "display")) {
            require(nodes?.optJSONObject(nodeId)?.optJSONArray("choices")?.length()?.let { it > 0 } == true) {
                "درخت تصمیم گره معتبر '$nodeId' را ندارد."
            }
        }
        val env = OrtEnvironment.getEnvironment()
        val options = OrtSession.SessionOptions()
        try {
            options.setIntraOpNumThreads(2)
            options.setInterOpNumThreads(1)
            session = env.createSession(File(root, "model.onnx").absolutePath, options)
        } finally {
            options.close()
        }
        environment = env
        val active = session ?: error("نشست ONNX ایجاد نشد.")
        val names = active.inputNames
        require(names.any { it.contains("input_ids", true) } &&
            names.any { it.contains("attention_mask", true) }) {
            "مدل باید ورودی input_ids و attention_mask داشته باشد."
        }
    }

    private fun embed(text: String): FloatArray {
        val env = environment ?: OrtEnvironment.getEnvironment()
        val active = session ?: error("مدل ONNX بارگذاری نشده است.")
        val encoded = encode(text)
        val inputs = linkedMapOf<String, OnnxTensor>()
        try {
            active.inputNames.forEach { name ->
                val values = when {
                    name.contains("input_ids", true) -> encoded.ids
                    name.contains("attention_mask", true) -> encoded.mask
                    name.contains("token_type_ids", true) -> LongArray(MAX_LENGTH)
                    else -> error("ورودی ONNX پشتیبانی‌نشده: $name")
                }
                val buffer = ByteBuffer.allocateDirect(values.size * java.lang.Long.BYTES)
                    .order(ByteOrder.nativeOrder()).asLongBuffer()
                buffer.put(values); buffer.rewind()
                inputs[name] = OnnxTensor.createTensor(env, buffer, longArrayOf(1, MAX_LENGTH.toLong()))
            }
            active.run(inputs).use { result ->
                val output = result.firstOrNull()?.value as? OnnxTensor
                    ?: error("مدل خروجی tensor ندارد.")
                val shape = output.info.shape
                val data = output.floatBuffer
                val values = FloatArray(data.remaining())
                data.get(values)
                val vector = when {
                    values.size == EMBEDDING_SIZE -> values
                    shape.size == 3 && shape.last() == EMBEDDING_SIZE.toLong() -> {
                        val seqLength = shape[1].toInt()
                        require(values.size == seqLength * EMBEDDING_SIZE) { "ابعاد خروجی embedding ناسازگار است." }
                        val pooled = FloatArray(EMBEDDING_SIZE)
                        var count = 0
                        for (token in 0 until minOf(seqLength, encoded.mask.size)) {
                            if (encoded.mask[token] == 0L) continue
                            count++
                            val offset = token * EMBEDDING_SIZE
                            for (d in 0 until EMBEDDING_SIZE) pooled[d] += values[offset + d]
                        }
                        require(count > 0) { "خروجی tokenizer هیچ توکن معتبری ندارد." }
                        for (d in pooled.indices) pooled[d] /= count
                        pooled
                    }
                    else -> error("خروجی مدل embedding مورد انتظار نیست؛ shape=${shape.contentToString()}")
                }
                return normalize(vector)
            }
        } finally {
            inputs.values.forEach { runCatching { it.close() } }
        }
    }

    private data class Encoded(val ids: LongArray, val mask: LongArray)

    private fun encode(raw: String): Encoded {
        val tokens = ArrayList<String>()
        val normalized = raw.lowercase(Locale.ROOT)
        val basic = ArrayList<String>()
        val current = StringBuilder()
        fun flush() { if (current.isNotEmpty()) { basic += current.toString(); current.setLength(0) } }
        for (ch in normalized) {
            when {
                ch.isWhitespace() || ch.isISOControl() -> flush()
                ch in ".,!?;:()[]{}<>/\\\"'@#$%^&*+=|،؛؟" -> { flush(); basic += ch.toString() }
                else -> current.append(ch)
            }
        }
        flush()
        for (word in basic) {
            val pieces = wordPiece(word)
            tokens.addAll(pieces)
            if (tokens.size >= MAX_LENGTH - 2) break
        }
        val full = listOf(clsToken) + tokens.take(MAX_LENGTH - 2) + sepToken
        val ids = LongArray(MAX_LENGTH) { vocab.getValue(padToken).toLong() }
        val mask = LongArray(MAX_LENGTH)
        for (i in full.indices.take(MAX_LENGTH)) {
            ids[i] = (vocab[full[i]] ?: vocab.getValue(unkToken)).toLong()
            mask[i] = 1
        }
        return Encoded(ids, mask)
    }

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
                if (candidate in vocab) { found = candidate; break }
                end--
            }
            if (found == null) return listOf(unkToken)
            pieces += found
            start = end
        }
        return pieces
    }

    private fun cosine(a: FloatArray, b: FloatArray): Float {
        var dot = 0.0
        for (i in a.indices) dot += a[i] * b[i]
        return dot.toFloat()
    }

    private fun normalize(vector: FloatArray): FloatArray {
        var norm = 0.0
        for (value in vector) norm += value * value
        val denominator = sqrt(norm).toFloat().takeIf { it > 1e-12f } ?: return vector
        return FloatArray(vector.size) { vector[it] / denominator }
    }

    override fun close() = synchronized(lock) {
        runCatching { session?.close() }
        session = null
        environment = null
        nodes = null
        vocab = emptyMap()
        candidateCache.clear()
    }
}
