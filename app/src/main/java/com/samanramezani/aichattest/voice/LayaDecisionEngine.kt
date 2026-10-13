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
import java.text.Normalizer
import kotlin.math.exp
import kotlin.math.ln

/**
 * Laya's typed-choice ONNX runner. It performs all decisions on-device and never generates
 * assistant text. A decision is accepted only when its softmax confidence and margin clear
 * conservative thresholds; otherwise no device action is inferred and the original text is routed to chat.
 */
internal class LayaDecisionEngine(private val root: File) : AutoCloseable {
    data class Decision(val command: DinaCommand, val route: String, val confidence: Double, val trace: List<String>)

    companion object {
        private const val MODEL_FILE = LayaDecisionModelStore.MODEL_FILE
        private const val DEFAULT_MAX_SEQUENCE = 1024
        private const val DEFAULT_HEAD_MAX_SEQUENCE = 256
        private const val MIN_CONFIDENCE = 0.43
        private const val MIN_MARGIN = 0.09
        private const val MIN_TEMPERATURE = 0.5
        private const val MAX_TEMPERATURE = 5.0
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
    private var maxSequence = DEFAULT_MAX_SEQUENCE
    private var headMaxSequence = DEFAULT_HEAD_MAX_SEQUENCE
    private var choiceTemperature = 1.0
    private var temperatureSource = "missing; neutral fallback 1.0"
    private val optionTemperatures = mutableMapOf<String, Double>()

    fun isReady(): Boolean = File(root, MODEL_FILE).isFile &&
        File(root, "$MODEL_FILE.data").isFile &&
        File(root, "tokenizer/tokenizer.json").isFile &&
        File(root, "tokenizer/tokenizer_config.json").isFile &&
        File(root, "encoder/config.json").isFile &&
        File(root, "rl_agent_config.json").isFile

    fun classify(rawText: String, currentVolume: Int? = null, maxVolume: Int? = null): Decision? = synchronized(lock) {
        lastTrace = emptyList()
        lastError = null
        if (rawText.isBlank()) return@synchronized null
        if (!isReady()) {
            closeLoaded()
            lastError = "LAYA-MODEL-001: مدل یا فایل‌های لازم آن نصب نیستند؛ تصمیم‌گیری محلی انجام نشد."
            return@synchronized null
        }
        val trace = mutableListOf<String>()
        var retainLoadedSession = false
        try {
            ensureLoaded()
            val domain = choose(rawText, "درخواست کاربر به کدام دسته مربوط است؟ Choose the user's request category.",
                listOf(
                    Option("AUDIO", "کنترل صدا: تنظیم، افزایش، کاهش، قطع یا وصل صدا. Audio volume control."),
                    Option("DISPLAY", "کنترل نمایشگر و روشنایی صفحه. Display and brightness control."),
                    Option("APPLICATION", "کنترل برنامه‌ها؛ این حوزه فعلاً غیرفعال است و نباید فرمانی اجرا کند. Application control, currently disabled."),
                    Option("MEDIA", "پخش و کنترل رسانه؛ این حوزه فعلاً غیرفعال است و نباید فرمانی اجرا کند. Media playback and control, currently disabled."),
                    Option("SYSTEM", "تنظیمات سیستم؛ این حوزه فعلاً غیرفعال است و نباید فرمانی اجرا کند. System settings, currently disabled."),
                    Option("INFORMATION", "پرسش اطلاعاتی که باید به مدل مکالمه فرستاده شود. Information request for the chat model."),
                    Option("OTHER", "سایر درخواست‌ها؛ بدون اجرای فرمان دستگاه. Other request, do not execute a device command."),
                )) ?: return@synchronized null
            trace += "domain=${domain.id} confidence=${"%.3f".format(Locale.US, domain.confidence)}"
            val command = when (domain.id) {
                "INFORMATION" -> {
                    // Keep the requested top-level taxonomy exact. Laya makes a second
                    // decision to distinguish local clock/date actions from ordinary chat.
                    val info = choose(
                        rawText,
                        "این درخواست اطلاعاتی است. آیا کاربر ساعت فعلی، تاریخ امروز یا پاسخ گفت‌وگویی معمولی می‌خواهد؟ فقط بهترین گزینه را انتخاب کن.",
                        listOf(
                            Option("GET_TIME", "کاربر ساعت فعلی دستگاه را می‌خواهد؛ زمان محلی را اعلام کن. Current local time."),
                            Option("GET_DATE", "کاربر تاریخ امروز را می‌خواهد. Today's local date."),
                            Option("CHAT", "پرسش اطلاعاتی معمولی است و باید به مدل گفت‌وگو برسد. Normal information request for chat.")
                        )
                    ) ?: return@synchronized null
                    trace += "information_action=${info.id} confidence=${"%.3f".format(Locale.US, info.confidence)}"
                    when (info.id) {
                        "GET_TIME" -> DinaCommand(DinaIntent.GET_TIME)
                        "GET_DATE" -> DinaCommand(DinaIntent.GET_DATE)
                        else -> DinaCommand(DinaIntent.CHAT)
                    }
                }
                "OTHER", "APPLICATION", "MEDIA", "SYSTEM" -> DinaCommand(DinaIntent.CHAT)
                "GET_TIME" -> DinaCommand(DinaIntent.GET_TIME)
                "GET_DATE" -> DinaCommand(DinaIntent.GET_DATE)
                "AUDIO" -> {
                    val operation = choose(rawText, "کاربر برای صدای رسانه چه عملی می‌خواهد؟ Choose the media-volume operation.",
                        listOf(
                            Option("INCREASE", "صدای رسانه را بیشتر کن یا بلندتر کن. Increase or turn up the volume."),
                            Option("DECREASE", "صدای رسانه را کمتر کن یا آرام‌تر کن. Decrease or turn down the volume."),
                            Option("SET", "صدای رسانه را روی سطح مشخص یا درصد معین تنظیم کن. Set a specific level or percentage."),
                            Option("MUTE", "صدای رسانه را قطع یا بی‌صدا کن. Mute the sound."),
                            Option("UNMUTE", "صدا را دوباره وصل کن و از حالت بی‌صدا خارج کن. Unmute or restore sound."),
                        )) ?: return@synchronized null
                    trace += "audio_operation=${operation.id} confidence=${"%.3f".format(Locale.US, operation.confidence)}"
                    when (operation.id) {
                        "MUTE" -> DinaCommand(DinaIntent.MUTE)
                        "UNMUTE" -> DinaCommand(DinaIntent.UNMUTE)
                        "INCREASE" -> DinaCommand(DinaIntent.INCREASE_VOLUME)
                        "DECREASE" -> DinaCommand(DinaIntent.DECREASE_VOLUME)
                        "SET" -> {
                            // The Laya model itself must interpret spoken numerals and percentages.
                            // Do not call the legacy regex/number parser here.
                            val actual = currentVolume
                            val maximum = maxVolume
                            val currentLevel = if (actual != null && maximum != null && maximum > 0) {
                                (actual * 10.0 / maximum).toInt().coerceIn(0, 10)
                            } else null
                            val state = buildString {
                                append(rawText)
                                if (currentLevel != null) append("\nCurrent media volume level: ").append(currentLevel).append(" of 10.")
                            }
                            val level = choose(
                                state,
                                "مقدار دقیق صدای درخواستی را تعیین کن. اگر درصد گفته شده، خودت آن را به مقیاس ۰ تا ۱۰ تبدیل کن؛ اگر عدد مستقیم ۰ تا ۱۰ گفته شده همان را انتخاب کن. عدد را حدس نزن و اگر نامشخص است UNKNOWN را برگردان. Interpret the original request and choose its exact target level from 0 to 10.",
                                (0..10).map { n ->
                                    Option("LEVEL_$n", "سطح صدای درخواستی برابر $n از ۱۰ است. Exact target volume level is $n out of 10.")
                                } + Option("UNKNOWN", "مقدار قابل تشخیص نیست؛ حدس نزن. Target level is unknown; do not guess.")
                            ) ?: return@synchronized null
                            trace += "volume_target=${level.id} confidence=${"%.3f".format(Locale.US, level.confidence)} current=${currentLevel ?: "unknown"}"
                            if (level.id == "UNKNOWN") return@synchronized null
                            val targetLevel = level.id.removePrefix("LEVEL_").toIntOrNull()
                                ?.takeIf { it in 0..10 } ?: return@synchronized null
                            DinaCommand(DinaIntent.SET_VOLUME, targetLevel * 10, valueIsPercent = true)
                        }
                        else -> return@synchronized null
                    }
                }
                "DISPLAY" -> {
                    val operation = choose(rawText, "کاربر برای روشنایی صفحه چه عملی می‌خواهد؟ Choose the screen-brightness operation.",
                        listOf(
                            Option("INCREASE", "روشنایی صفحه را بیشتر کن. Increase screen brightness."),
                            Option("DECREASE", "روشنایی صفحه را کمتر کن. Decrease screen brightness."),
                            Option("SET", "روشنایی صفحه را روی درصد مشخص تنظیم کن. Set brightness to a percentage."),
                        )) ?: return@synchronized null
                    trace += "display_operation=${operation.id} confidence=${"%.3f".format(Locale.US, operation.confidence)}"
                    when (operation.id) {
                        "INCREASE" -> DinaCommand(DinaIntent.INCREASE_BRIGHTNESS)
                        "DECREASE" -> DinaCommand(DinaIntent.DECREASE_BRIGHTNESS)
                        "SET" -> {
                            val target = choose(
                                rawText,
                                "مقدار دقیق روشنایی درخواستی را از ۰ تا ۱۰۰ درصد تعیین کن. اگر مقدار مشخص نیست UNKNOWN برگردان و حدس نزن. Choose the exact requested brightness percentage from 0 to 100.",
                                (0..10).map { n ->
                                    Option("LEVEL_$n", "روشنایی صفحه روی ${n * 10} درصد تنظیم شود. Set screen brightness to ${n * 10} percent.")
                                } + Option("UNKNOWN", "مقدار روشنایی مشخص نیست؛ حدس نزن. Unknown target brightness.")
                            ) ?: return@synchronized null
                            trace += "brightness_target=${target.id} confidence=${"%.3f".format(Locale.US, target.confidence)}"
                            if (target.id == "UNKNOWN") return@synchronized null
                            val level = target.id.removePrefix("LEVEL_").toIntOrNull()
                                ?.takeIf { it in 0..10 } ?: return@synchronized null
                            DinaCommand(DinaIntent.SET_BRIGHTNESS, level * 10, valueIsPercent = true)
                        }
                        else -> return@synchronized null
                    }
                }
                else -> return@synchronized null
            }
            lastTrace = trace.toList() + lastTrace
            lastError = null
            // Laya's resident memory is large. Keep it warm only after local device actions;
            // close the session before the chat model is loaded, and also on every abstention.
            retainLoadedSession = command.intent != DinaIntent.CHAT
            Decision(command, domain.id, domain.confidence, trace.toList())
        } catch (t: Throwable) {
            val causes = generateSequence(t as Throwable?) { it.cause }
                .take(8)
                .joinToString(" <- ") { cause ->
                    cause.javaClass.name + (cause.message?.takeIf { it.isNotBlank() }?.let { ": $it" } ?: "")
                }
            val origin = t.stackTrace.firstOrNull()?.let { " at $it" }.orEmpty()
            lastError = "LAYA-INFER-001: $causes$origin"
            lastTrace = trace.toList() + lastTrace
            null
        } finally {
            if (!retainLoadedSession) closeLoaded()
        }
    }

    private data class Option(val id: String, val description: String)
    private data class Ranked(val id: String, val confidence: Double)

    private fun choose(state: String, instructions: String, options: List<Option>): Ranked? {
        require(options.size in 2..16) { "LAYA-QUESTION-001: تعداد گزینه‌های تصمیم باید بین ۲ و ۱۶ باشد." }
        val env = environment ?: error("محیط ONNX آماده نیست.")
        val active = session ?: error("نشست ONNX آماده نیست.")
        val tok = tokenizer ?: error("توکن‌ساز Laya آماده نیست.")
        // Match the multilingual checkpoint's two-budget layout: the question and option
        // descriptions are bounded by head_max_len; only the remaining room is used by state.
        // Each option keeps its marker and at least a small label budget, rather than letting
        // verbose descriptions silently consume the user's actual request.
        val optionTokenBudget = minOf(48, maxOf(4, (headMaxSequence - 16) / options.size))
        val optionTokens = options.map { option ->
            tok.encode(" ${option.id}: ${option.description}", optionTokenBudget - 1)
        }
        val fixedHeadTokens = 1 + 1 + options.indices.sumOf { 1 + optionTokens[it].size } + 1
        val instructionBudget = (headMaxSequence - fixedHeadTokens).coerceAtLeast(8)
        val instructionTokens = tok.encode("choice question: $instructions", instructionBudget)
        val ids = ArrayList<Long>(maxSequence)
        val markers = ArrayList<Long>(options.size)
        ids += tok.clsId
        ids += instructionTokens
        ids += tok.sepId
        for (index in options.indices) {
            markers += ids.size.toLong()
            ids += tok.maskId
            ids += optionTokens[index]
        }
        ids += tok.sepId
        require(ids.size <= headMaxSequence) {
            "LAYA-QUESTION-002: سربرگ گزینه‌ها از بودجه head_max_len عبور کرد؛ head=${ids.size}, budget=$headMaxSequence, options=${options.size}"
        }
        val remaining = (maxSequence - ids.size - 1).coerceAtLeast(0)
        val stateIds = tok.encode(state, remaining)
        ids += stateIds
        ids += tok.sepId
        require(markers.isNotEmpty() && ids.size <= maxSequence) {
            "LAYA-QUESTION-003: طول ورودی نامعتبر است؛ total=${ids.size}, max_len=$maxSequence, head=${instructionTokens.size + fixedHeadTokens}"
        }

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
                require(logits.all { it.isFinite() }) {
                    "LAYA-INFER-007: خروجی مدل شامل امتیاز NaN یا بی‌نهایت است."
                }
                val max = logits.maxOrNull() ?: return null
                // Apply the official checkpoint's choice temperature and per-option bucket if
                // present. The published multilingual bundle has no fitted temperatures; in that
                // case this remains a raw softmax score and is explicitly reported as uncalibrated.
                val bucket = if (options.size >= 11) "choice:11+" else "choice:${options.size}"
                val bucketTemperature = optionTemperatures[bucket]
                val activeTemperature = (bucketTemperature ?: choiceTemperature).coerceIn(MIN_TEMPERATURE, MAX_TEMPERATURE)
                val activeCalibration = if (bucketTemperature != null) "rl_agent_config.json:$bucket" else temperatureSource
                val scaled = logits.map { (it.toDouble() / activeTemperature).coerceIn(-80.0, 80.0) }
                val scaledMax = scaled.maxOrNull() ?: return null
                val exps = scaled.map { exp(it - scaledMax) }
                val sum = exps.sum().takeIf { it.isFinite() && it > 0.0 } ?: return null
                val probs = exps.map { it / sum }
                val ranked = probs.indices.sortedByDescending { probs[it] }
                val best = ranked.first()
                val second = ranked.getOrNull(1)
                val margin = probs[best] - (second?.let { probs[it] } ?: 0.0)
                val top = ranked.take(3).joinToString(";") { index ->
                    "${options[index].id}:logit=${"%.4f".format(Locale.US, logits[index])},score=${"%.4f".format(Locale.US, probs[index])}"
                }
                val detail = "stage=${instructions.take(48).replace(Regex("\\s+"), " ")} options=${options.size} inputTokens=${ids.size} stateTokens=${stateIds.size} headTokens=${ids.size - stateIds.size - 1} temperature=${"%.3f".format(Locale.US, activeTemperature)} calibration=$activeCalibration top=[$top]"
                lastTrace = lastTrace + detail
                if (probs[best] < MIN_CONFIDENCE || margin < MIN_MARGIN) {
                    lastError = "LAYA-INFER-005: تصمیم کم‌اطمینان؛ score=${"%.3f".format(Locale.US, probs[best])}, margin=${"%.3f".format(Locale.US, margin)}, threshold=${MIN_CONFIDENCE}/${MIN_MARGIN}; $detail"
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
        val configFile = File(root, "rl_agent_config.json")
        val tokenizerFile = File(root, "tokenizer/tokenizer.json")
        val key = "${model.absolutePath}:${model.length()}:${model.lastModified()}:${File(root, "$MODEL_FILE.data").length()}:${configFile.lastModified()}:${tokenizerFile.lastModified()}"
        if (session != null && cachedModelKey == key) return
        closeLoaded()
        loadRuntimeConfig(configFile)
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

    private fun loadRuntimeConfig(configFile: File) {
        val cfg = JSONObject(configFile.readText(Charsets.UTF_8))
        maxSequence = cfg.optInt("max_len", DEFAULT_MAX_SEQUENCE)
            .coerceIn(128, 8192)
        headMaxSequence = cfg.optInt("head_max_len", DEFAULT_HEAD_MAX_SEQUENCE)
            .coerceIn(64, maxSequence - 16)
        val temperatures = cfg.optJSONArray("temperature")
        val rawTemperature = temperatures?.optDouble(0, Double.NaN) ?: Double.NaN
        val bucketMap = cfg.optJSONObject("temperature_by_options")
        val temperatureByOptions = mutableMapOf<String, Double>()
        if (bucketMap != null) {
            val keys = bucketMap.keys()
            while (keys.hasNext()) {
                val key = keys.next()
                val value = bucketMap.optDouble(key, Double.NaN)
                if (value.isFinite() && value > 0.0) temperatureByOptions[key] = value
            }
        }
        val selected = if (rawTemperature.isFinite() && rawTemperature > 0.0) rawTemperature else Double.NaN
        choiceTemperature = if (selected.isFinite()) selected.coerceIn(MIN_TEMPERATURE, MAX_TEMPERATURE) else 1.0
        temperatureSource = if (selected.isFinite() && kotlin.math.abs(selected - 1.0) > 1e-6) {
            "rl_agent_config.json:temperature[choice]; bucket calibration checked per option count"
        } else {
            "neutral temperature 1.0; calibration not established"
        }
        optionTemperatures.clear()
        optionTemperatures.putAll(temperatureByOptions)
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
    private val tokenScores = HashMap<String, Double>()
    private val modelType: String
    private val lowercase: Boolean
    private val byteLevel: Boolean
    private val metaspace: Boolean
    private val rootPrependNever: Boolean
    private val unkToken: String
    private var normalizerConfig: String = ""
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
                if (item.length() >= 2) {
                    val token = item.getString(0)
                    vocab[token] = i
                    tokenScores[token] = item.optDouble(1, -10.0)
                }
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
        val normalizer = root.opt("normalizer")
        normalizerConfig = normalizer?.toString().orEmpty()
        lowercase = normalizerConfig.contains("Lowercase") ||
            (normalizer as? JSONObject)?.optBoolean("lowercase", false) == true
        val pre = root.opt("pre_tokenizer")?.toString().orEmpty()
        byteLevel = pre.contains("ByteLevel")
        metaspace = pre.contains("Metaspace")
        rootPrependNever = pre.contains("prepend_scheme") && pre.contains("never")
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
        var text = raw.replace('ي', 'ی').replace('ك', 'ک')
        if (lowercase) text = text.lowercase(Locale.ROOT)
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
        for (word in splitBertPreTokens(normalizeBertText(text))) {
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

    private fun normalizeBertText(raw: String): String {
        var value = raw.replace('ي', 'ی').replace('ك', 'ک')
        if (lowercase) value = value.lowercase(Locale.ROOT)
        // BertNormalizer strips combining marks for accent-insensitive matching. This also
        // handles optional Arabic vowel marks in Persian input without altering base letters.
        val stripAccents = lowercase ||
            rootNormalizerContains("StripAccents")
        if (stripAccents) {
            value = Normalizer.normalize(value, Normalizer.Form.NFD)
                .filterNot { Character.getType(it) == Character.NON_SPACING_MARK.toInt() }
                .let { Normalizer.normalize(it, Normalizer.Form.NFC) }
        }
        return value.filterNot { it == '\u0000' || it == '\uFFFD' }
    }

    private fun rootNormalizerContains(needle: String): Boolean =
        normalizerConfig.contains(needle)

    private fun splitBertPreTokens(text: String): List<String> {
        val result = ArrayList<String>()
        val current = StringBuilder()
        fun flush() {
            if (current.isNotEmpty()) {
                result += current.toString()
                current.setLength(0)
            }
        }
        var index = 0
        while (index < text.length) {
            val cp = text.codePointAt(index)
            val ch = String(Character.toChars(cp))
            val type = Character.getType(cp)
            val punctuation = type == Character.CONNECTOR_PUNCTUATION.toInt() ||
                type == Character.DASH_PUNCTUATION.toInt() ||
                type == Character.START_PUNCTUATION.toInt() ||
                type == Character.END_PUNCTUATION.toInt() ||
                type == Character.INITIAL_QUOTE_PUNCTUATION.toInt() ||
                type == Character.FINAL_QUOTE_PUNCTUATION.toInt() ||
                type == Character.OTHER_PUNCTUATION.toInt()
            if (Character.isWhitespace(cp) || Character.isSpaceChar(cp)) {
                flush()
            } else if (punctuation) {
                flush()
                result += ch
            } else {
                current.append(ch)
            }
            index += Character.charCount(cp)
        }
        flush()
        return result
    }

    private fun encodeBpe(text: String): List<String> {
        val result = ArrayList<String>()
        val preTokens = if (metaspace) {
            text.trim().split(Regex("\\s+")).filter { it.isNotEmpty() }.mapIndexed { i, word ->
                (if (i == 0 && rootPrependNever) "" else "▁") + word
            }
        } else {
            (if (byteLevel) Regex("\\s*[\\p{L}\\p{M}]+|\\s*[\\p{N}]+|\\s*[^\\s\\p{L}\\p{M}\\p{N}]+|\\s+")
             else Regex("[\\p{L}\\p{M}]+|[\\p{N}]+|[^\\s\\p{L}\\p{M}\\p{N}]|\\s+"))
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
        // SentencePiece/Unigram tokenization uses a Viterbi path over scored vocabulary pieces.
        val normalized = if (metaspace) {
            val joined = text.replace(Regex("\\s+"), "▁")
            if (!rootPrependNever && !joined.startsWith("▁")) "▁$joined" else joined
        } else text
        val n = normalized.length
        if (n == 0) return emptyList()
        val best = DoubleArray(n + 1) { Double.NEGATIVE_INFINITY }
        val previous = IntArray(n + 1) { -1 }
        val previousToken = arrayOfNulls<String>(n + 1)
        best[0] = 0.0
        for (start in 0 until n) {
            if (best[start].isInfinite() && best[start] < 0) continue
            var matched = false
            // Tokenizer vocabularies can contain hundreds of thousands of entries. Scan a
            // bounded character window instead of iterating the entire vocabulary per position.
            for (end in minOf(n, start + 32) downTo start + 1) {
                val piece = normalized.substring(start, end)
                if (piece !in vocab) continue
                val score = tokenScores[piece] ?: -10.0
                val candidate = best[start] + score
                if (candidate > best[end]) {
                    best[end] = candidate
                    previous[end] = start
                    previousToken[end] = piece
                }
                matched = true
            }
            if (!matched) {
                val end = start + 1
                val candidate = best[start] - 100.0
                if (candidate > best[end]) {
                    best[end] = candidate
                    previous[end] = start
                    previousToken[end] = unkToken
                }
            }
        }
        if (previous[n] < 0) return listOf(unkToken)
        val output = ArrayList<String>()
        var cursor = n
        while (cursor > 0) {
            val token = previousToken[cursor] ?: return listOf(unkToken)
            output += token
            cursor = previous[cursor]
        }
        output.reverse()
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
