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
import kotlin.math.exp

/**
 * Laya's typed-choice ONNX runner. It performs all decisions on-device and never generates
 * assistant text. A decision is accepted only when its softmax confidence and margin clear
 * conservative thresholds; otherwise the caller may use the existing safe command parser.
 */
internal class LayaDecisionEngine(private val root: File) : AutoCloseable {
    data class Decision(val command: DinaCommand, val route: String, val confidence: Double, val trace: List<String>)

    companion object {
        private const val MODEL_FILE = LayaDecisionModelStore.MODEL_FILE
        private const val MAX_SEQUENCE = 512
        private const val MIN_CONFIDENCE = 0.43
        private const val MIN_MARGIN = 0.09
    }

    @Volatile var lastError: String? = null
        private set
    @Volatile var lastTrace: List<String> = emptyList()
        private set

    private val lock = Any()
    private var environment: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var tokenizer: LayaTokenizer? = null
    private var cachedModelKey: String? = null

    fun isReady(): Boolean = File(root, MODEL_FILE).isFile &&
        File(root, "$MODEL_FILE.data").isFile &&
        File(root, "tokenizer/tokenizer.json").isFile

    fun classify(rawText: String, currentVolume: Int? = null, maxVolume: Int? = null): Decision? = synchronized(lock) {
        lastTrace = emptyList()
        if (rawText.isBlank() || !isReady()) return@synchronized null
        val trace = mutableListOf<String>()
        try {
            ensureLoaded()
            val domain = choose(rawText, "Which action category best matches the user's intent?",
                listOf(
                    Option("GET_TIME", "The user asks for the current local clock time, asks what time it is, or says to announce the time."),
                    Option("GET_DATE", "The user asks for today's date."),
                    Option("AUDIO", "The user wants to change media volume, make sound louder or quieter, or mute/unmute."),
                    Option("DISPLAY", "The user wants to change screen brightness."),
                    Option("CHAT", "The user is asking a general question or wants a normal conversation, not a device action."),
                )) ?: return@synchronized null
            trace += "domain=${domain.id} confidence=${"%.3f".format(Locale.US, domain.confidence)}"
            val command = when (domain.id) {
                "GET_TIME" -> DinaCommand(DinaIntent.GET_TIME)
                "GET_DATE" -> DinaCommand(DinaIntent.GET_DATE)
                "CHAT" -> DinaCommand(DinaIntent.CHAT)
                "AUDIO" -> {
                    val operation = choose(rawText, "Which media-volume operation does the user want?",
                        listOf(
                            Option("INCREASE", "Increase the current media volume or make the sound louder."),
                            Option("DECREASE", "Decrease the current media volume or make the sound quieter."),
                            Option("SET", "Set media volume to a requested exact level or explicit percentage."),
                            Option("MUTE", "Mute or silence media sound."),
                            Option("UNMUTE", "Restore sound after mute."),
                        )) ?: return@synchronized null
                    trace += "audio_operation=${operation.id} confidence=${"%.3f".format(Locale.US, operation.confidence)}"
                    when (operation.id) {
                        "MUTE" -> DinaCommand(DinaIntent.MUTE)
                        "UNMUTE" -> DinaCommand(DinaIntent.UNMUTE)
                        "INCREASE", "DECREASE", "SET" -> {
                            val parsed = DinaCommandClassifier.classify(rawText)
                            val explicitPercent = Regex("(?:%|درصد|percent)", RegexOption.IGNORE_CASE).containsMatchIn(rawText)
                            run {
                                val actual = currentVolume
                                val maximum = maxVolume
                                val currentLevel = if (actual != null && maximum != null && maximum > 0) {
                                    (actual * 10.0 / maximum).toInt().coerceIn(0, 10)
                                } else null
                                val question = when (operation.id) {
                                    "INCREASE" -> "The user requests louder sound. Choose a target level on a 1-10 scale, greater than the current level."
                                    "DECREASE" -> "The user requests quieter sound. Choose a target level on a 1-10 scale, lower than the current level."
                                    else -> "Choose the exact target media volume level from 1 to 10 requested by the user. If no level is specified, do not guess."
                                }
                                val state = buildString {
                                    append(rawText)
                                    if (currentLevel != null) append("\nCurrent media volume level: ").append(currentLevel).append(" of 10.")
                                }
                                val level = choose(state, question, (1..10).map { n ->
                                    Option("LEVEL_$n", "Set the target media volume to level $n out of 10.")
                                } + if (operation.id == "SET") listOf(
                                    Option("OTHER_EXPLICIT_VALUE", "The user explicitly gave a percentage or a numeric target that is not a level from 1 to 10.")
                                ) else emptyList()) ?: return@synchronized null
                                trace += "volume_target=${level.id} confidence=${"%.3f".format(Locale.US, level.confidence)} current=${currentLevel ?: "unknown"}"
                                when {
                                    level.id == "OTHER_EXPLICIT_VALUE" && operation.id == "SET" &&
                                        parsed.intent == DinaIntent.SET_VOLUME && parsed.value != null -> parsed
                                    level.id.startsWith("LEVEL_") && !explicitPercent -> {
                                        val targetLevel = level.id.removePrefix("LEVEL_").toIntOrNull() ?: return@synchronized null
                                        if (operation.id == "INCREASE" && currentLevel != null && targetLevel <= currentLevel) return@synchronized null
                                        if (operation.id == "DECREASE" && currentLevel != null && targetLevel >= currentLevel) return@synchronized null
                                        DinaCommand(DinaIntent.SET_VOLUME, targetLevel * 10, valueIsPercent = true)
                                    }
                                    else -> return@synchronized null
                                }
                            }
                        }
                        else -> return@synchronized null
                    }
                }
                "DISPLAY" -> {
                    val operation = choose(rawText, "Which screen-brightness operation does the user want?",
                        listOf(
                            Option("INCREASE", "Increase screen brightness."),
                            Option("DECREASE", "Decrease screen brightness."),
                            Option("SET", "Set screen brightness to a specified percentage."),
                        )) ?: return@synchronized null
                    trace += "display_operation=${operation.id} confidence=${"%.3f".format(Locale.US, operation.confidence)}"
                    val parsed = DinaCommandClassifier.classify(rawText)
                    when (operation.id) {
                        "INCREASE" -> DinaCommand(DinaIntent.INCREASE_BRIGHTNESS)
                        "DECREASE" -> DinaCommand(DinaIntent.DECREASE_BRIGHTNESS)
                        "SET" -> parsed.takeIf { it.intent == DinaIntent.SET_BRIGHTNESS && it.value != null } ?: return@synchronized null
                        else -> return@synchronized null
                    }
                }
                else -> return@synchronized null
            }
            lastTrace = trace.toList()
            lastError = null
            Decision(command, domain.id, trace.lastOrNull()?.substringAfter("confidence=")?.toDoubleOrNull() ?: 1.0, trace.toList())
        } catch (t: Throwable) {
            lastError = "LAYA-INFER-001: ${t.message ?: t.javaClass.simpleName}"
            lastTrace = trace.toList()
            null
        }
    }

    private data class Option(val id: String, val description: String)
    private data class Ranked(val id: String, val confidence: Double)

    private fun choose(state: String, instructions: String, options: List<Option>): Ranked? {
        require(options.size in 2..16) { "LAYA-QUESTION-001: تعداد گزینه‌های تصمیم باید بین ۲ و ۱۶ باشد." }
        val env = environment ?: error("محیط ONNX آماده نیست.")
        val active = session ?: error("نشست ONNX آماده نیست.")
        val tok = tokenizer ?: error("توکن‌ساز Laya آماده نیست.")
        val ids = ArrayList<Long>()
        val markers = ArrayList<Long>()
        ids += tok.clsId
        ids += tok.encode("choice question: $instructions", MAX_SEQUENCE)
        ids += tok.sepId
        for (option in options) {
            markers += ids.size.toLong()
            ids += tok.maskId
            ids += tok.encode(" ${option.id}: ${option.description}", MAX_SEQUENCE)
            if (ids.size >= MAX_SEQUENCE - 2) error("LAYA-QUESTION-002: سربرگ گزینه‌ها از ظرفیت مدل بیشتر شد.")
        }
        ids += tok.sepId
        val remaining = (MAX_SEQUENCE - ids.size - 1).coerceAtLeast(0)
        ids += tok.encode(state, remaining)
        ids += tok.sepId
        require(markers.isNotEmpty() && ids.size <= MAX_SEQUENCE) { "LAYA-QUESTION-003: طول ورودی نامعتبر است." }

        val feeds = linkedMapOf<String, OnnxTensor>()
        try {
            active.inputNames.forEach { name ->
                feeds[name] = when {
                    name.equals("input_ids", true) -> longTensor(env, ids.toLongArray(), longArrayOf(1, ids.size.toLong()))
                    name.equals("attention_mask", true) -> longTensor(env, LongArray(ids.size) { 1L }, longArrayOf(1, ids.size.toLong()))
                    name.equals("marker_pos", true) -> longTensor(env, markers.toLongArray(), longArrayOf(1, markers.size.toLong()))
                    name.equals("marker_mask", true) -> OnnxTensor.createTensor(env, arrayOf(BooleanArray(markers.size) { true }))
                    name.equals("qtype", true) -> longTensor(env, longArrayOf(0L), longArrayOf(1))
                    else -> error("LAYA-INFER-002: ورودی ناشناخته مدل ONNX: $name")
                }
            }
            active.run(feeds).use { result ->
                val output = result.firstOrNull()?.value as? OnnxTensor
                    ?: error("LAYA-INFER-003: مدل خروجی tensor ندارد.")
                val buffer = output.floatBuffer
                val logits = FloatArray(buffer.remaining())
                buffer.get(logits)
                require(logits.size == options.size) {
                    "LAYA-INFER-004: خروجی choice با تعداد گزینه‌ها همخوان نیست؛ output=${logits.size}, options=${options.size}, shape=${output.info.shape.contentToString()}"
                }
                val max = logits.maxOrNull() ?: return null
                val exps = logits.map { exp((it - max).coerceIn(-80f, 80f).toDouble()) }
                val sum = exps.sum().takeIf { it > 0.0 } ?: return null
                val probs = exps.map { it / sum }
                val ranked = probs.indices.sortedByDescending { probs[it] }
                val best = ranked.first()
                val margin = probs[best] - (ranked.getOrNull(1)?.let { probs[it] } ?: 0.0)
                if (probs[best] < MIN_CONFIDENCE || margin < MIN_MARGIN) {
                    lastError = "LAYA-INFER-005: تصمیم کم‌اطمینان؛ confidence=${"%.3f".format(Locale.US, probs[best])}, margin=${"%.3f".format(Locale.US, margin)}"
                    return null
                }
                return Ranked(options[best].id, probs[best])
            }
        } finally {
            feeds.values.forEach { runCatching { it.close() } }
        }
    }

    private fun longTensor(env: OrtEnvironment, values: LongArray, shape: LongArray): OnnxTensor {
        val buffer = ByteBuffer.allocateDirect(values.size * java.lang.Long.BYTES)
            .order(ByteOrder.nativeOrder()).asLongBuffer()
        buffer.put(values)
        buffer.rewind()
        return OnnxTensor.createTensor(env, buffer, shape)
    }

    private fun ensureLoaded() {
        val model = File(root, MODEL_FILE)
        val key = "${model.absolutePath}:${model.length()}:${model.lastModified()}:${File(root, "$MODEL_FILE.data").length()}"
        if (session != null && cachedModelKey == key) return
        closeLoaded()
        val env = OrtEnvironment.getEnvironment()
        val options = OrtSession.SessionOptions()
        try {
            options.setIntraOpNumThreads(2)
            options.setInterOpNumThreads(1)
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            session = env.createSession(model.absolutePath, options)
        } finally {
            options.close()
        }
        environment = env
        tokenizer = LayaTokenizer(File(root, "tokenizer/tokenizer.json"))
        val names = session!!.inputNames
        val required = setOf("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
        require(names.containsAll(required)) {
            "LAYA-INFER-006: ورودی‌های مدل با قالب Laya سازگار نیستند؛ inputs=${names.joinToString()}"
        }
        cachedModelKey = key
    }

    private fun closeLoaded() {
        runCatching { session?.close() }
        session = null
        tokenizer = null
        cachedModelKey = null
    }

    override fun close() = synchronized(lock) {
        closeLoaded()
        environment = null
    }
}

/** Small tokenizer implementation for the WordPiece/BPE tokenizer formats used by Laya bundles. */
private class LayaTokenizer(file: File) {
    private val vocab = HashMap<String, Int>()
    private val merges = HashMap<String, Int>()
    private val modelType: String
    private val lowercase: Boolean
    private val byteLevel: Boolean
    private val metaspace: Boolean
    private val rootPrependNever: Boolean
    private val unkToken: String
    val clsId: Long
    val sepId: Long
    val maskId: Long

    init {
        val root = JSONObject(file.readText(Charsets.UTF_8))
        val model = root.optJSONObject("model") ?: error("LAYA-TOKENIZER-001: tokenizer model گم شده است.")
        modelType = model.optString("type")
        require(modelType in setOf("BPE", "WordPiece", "Unigram")) {
            "LAYA-TOKENIZER-002: نوع tokenizer پشتیبانی نمی‌شود: $modelType"
        }
        val rawVocab = model.optJSONObject("vocab")
        if (rawVocab != null) {
            val keys = rawVocab.keys()
            while (keys.hasNext()) {
                val key = keys.next()
                vocab[key] = rawVocab.getInt(key)
            }
        } else {
            val arr = model.optJSONArray("vocab") ?: error("LAYA-TOKENIZER-003: واژگان پیدا نشد.")
            for (i in 0 until arr.length()) {
                val item = arr.optJSONArray(i) ?: continue
                if (item.length() >= 2) vocab[item.getString(0)] = i
            }
        }
        val mergesArray = model.optJSONArray("merges") ?: JSONArray()
        for (i in 0 until mergesArray.length()) {
            val merge = mergesArray.opt(i)
            val pair = when (merge) {
                is String -> merge.trim().split(Regex("\\s+"), limit = 2).takeIf { it.size == 2 }
                is JSONArray -> if (merge.length() >= 2) listOf(merge.getString(0), merge.getString(1)) else null
                else -> null
            }
            if (pair != null) merges["${pair[0]} ${pair[1]}"] = i
        }
        val normalizer = root.optJSONObject("normalizer")
        lowercase = normalizer?.optBoolean("lowercase", false) == true
        val pre = root.optJSONObject("pre_tokenizer")?.toString().orEmpty()
        byteLevel = pre.contains("ByteLevel")
        metaspace = pre.contains("Metaspace")
        rootPrependNever = Regex("\\"prepend_scheme\\"\\s*:\\s*\\"never\\"").containsMatchIn(pre)
        unkToken = model.optString("unk_token", "<unk>")
        val special = root.optJSONArray("added_tokens") ?: JSONArray()
        val specialIds = HashMap<String, Int>()
        for (i in 0 until special.length()) {
            val item = special.optJSONObject(i) ?: continue
            val content = item.optString("content")
            if (content.isNotEmpty()) specialIds[content] = item.optInt("id", vocab[content] ?: -1)
        }
        fun id(vararg candidates: String): Long {
            for (candidate in candidates) {
                val found = vocab[candidate] ?: specialIds[candidate]
                if (found != null && found >= 0) return found.toLong()
            }
            error("LAYA-TOKENIZER-004: توکن ویژه پیدا نشد: ${candidates.joinToString()}")
        }
        clsId = id("<s>", "[CLS]", "<|begin_of_text|>")
        sepId = id("</s>", "[SEP]", "<|end_of_text|>")
        maskId = id("<mask>", "[MASK]", "<|mask|>")
    }

    fun encode(raw: String, maxTokens: Int): List<Long> {
        if (maxTokens <= 0 || raw.isEmpty()) return emptyList()
        val text = if (lowercase) raw.lowercase(Locale.ROOT) else raw
        val tokens = when (modelType) {
            "WordPiece" -> encodeWordPiece(text)
            "BPE" -> encodeBpe(text)
            "Unigram" -> encodeUnigram(text)
            else -> emptyList()
        }
        return tokens.take(maxTokens).map { (vocab[it] ?: vocab[unkToken] ?: 0).toLong() }
    }

    private fun encodeWordPiece(text: String): List<String> {
        val result = ArrayList<String>()
        for (word in text.trim().split(Regex("\\s+"))) {
            if (word.isEmpty()) continue
            var start = 0
            val pieces = ArrayList<String>()
            while (start < word.length) {
                var end = word.length
                var match: String? = null
                while (end > start) {
                    val candidate = (if (start == 0) "" else "##") + word.substring(start, end)
                    if (candidate in vocab) { match = candidate; break }
                    end--
                }
                if (match == null) { pieces.clear(); pieces += unkToken; break }
                pieces += match
                start = end
            }
            result += pieces
        }
        return result
    }

    private fun encodeBpe(text: String): List<String> {
        val result = ArrayList<String>()
        val preTokens = if (metaspace) {
            text.trim().split(Regex("\\s+")).filter { it.isNotEmpty() }.mapIndexed { i, word ->
                (if (i == 0 && rootPrependNever) "" else "▁") + word
            }
        } else {
            Regex("[\\p{L}\\p{M}]+|[\\p{N}]+|[^\\s\\p{L}\\p{M}\\p{N}]|\\s+")
                .findAll(text).map { it.value }.toList()
        }
        for (part in preTokens) {
            if (part.isBlank()) continue
            val symbols = if (byteLevel) byteEncode(part).map { it.toString() } else part.map { it.toString() }
            if (symbols.isEmpty()) continue
            val pieces = symbols.toMutableList()
            while (pieces.size > 1) {
                var bestRank = Int.MAX_VALUE
                var bestIndex = -1
                for (i in 0 until pieces.lastIndex) {
                    val rank = merges["${pieces[i]} ${pieces[i + 1]}"] ?: Int.MAX_VALUE
                    if (rank < bestRank) { bestRank = rank; bestIndex = i }
                }
                if (bestIndex < 0) break
                pieces[bestIndex] = pieces[bestIndex] + pieces[bestIndex + 1]
                pieces.removeAt(bestIndex + 1)
            }
            result += pieces
        }
        return result
    }

    private fun encodeUnigram(text: String): List<String> {
        // SentencePiece/Unigram: Viterbi over the actual vocabulary. Scores are optional in
        // tokenizer.json; longest-match fallback is deterministic for bundles without scores.
        val pieces = vocab.keys.filter { it.isNotEmpty() }.sortedByDescending { it.length }
        val output = ArrayList<String>()
        val words = text.trim().split(Regex("\\s+")).filter { it.isNotEmpty() }
        for ((index, word) in words.withIndex()) {
            val normalizedWord = (if (metaspace && !(index == 0 && rootPrependNever)) "▁" else "") + word
            var offset = 0
            while (offset < normalizedWord.length) {
                val found = pieces.firstOrNull { normalizedWord.startsWith(it, offset) }
                if (found == null) {
                    output += unkToken
                    offset++
                } else {
                    output += found
                    offset += found.length
                }
            }
        }
        return output
    }

    private fun byteEncode(text: String): String {
        val bytes = text.toByteArray(Charsets.UTF_8)
        val result = StringBuilder(bytes.size)
        val visible = ArrayList<Int>()
        for (b in 33..126) visible += b
        for (b in 161..172) visible += b
        for (b in 174..255) visible += b
        val mapped = HashMap<Int, Char>()
        var extra = 0
        for (b in 0..255) {
            val code = if (b in visible) b else {
                while (visible.contains(256 + extra)) extra++
                256 + extra++
            }
            mapped[b] = code.toChar()
        }
        for (byte in bytes) result.append(mapped[byte.toInt() and 0xff])
        return result.toString()
    }
}
