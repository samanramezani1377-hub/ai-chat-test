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
        // Match convaiinnovations/laya rl_common.py build_sequence:
        // [CLS] "choice question: ..." [SEP] [MASK] option ... [SEP] state [SEP].
        val optionTokens = options.map { option ->
            listOf(tok.maskId) + tok.encode(" ${option.id}: ${option.description}".replace(tok.maskToken, " "), 48)
        }.toMutableList()
        var optionBudget = headMaxSequence - optionTokens.sumOf { it.size }
        if (optionBudget < 16) {
            val perOptionBudget = maxOf(4, (headMaxSequence - 16) / optionTokens.size)
            for (index in optionTokens.indices) optionTokens[index] = optionTokens[index].take(perOptionBudget)
            optionBudget = headMaxSequence - optionTokens.sumOf { it.size }
        }
        val instructionTokens = tok.encode("choice question: ${instructions.replace(tok.maskToken, " ")}", maxOf(8, optionBudget))
        val ids = ArrayList<Long>(maxSequence)
        val markers = ArrayList<Long>(options.size)
        ids += tok.clsId
        ids += instructionTokens.take(maxOf(8, optionBudget))
        ids += tok.sepId
        for (option in optionTokens) {
            markers += ids.size.toLong()
            ids += option
        }
        ids += tok.sepId
        // Upstream reserves head_max_len for instruction + option token content, excluding
        // the structural [CLS]/[SEP] markers. It does not assert this intermediate length.
        require(markers.size == options.size && ids.size <= maxSequence) {
            "LAYA-QUESTION-002: سربرگ گزینه‌ها نامعتبر است؛ head=${ids.size}, max_len=$maxSequence, options=${options.size}"
        }
        val remaining = (maxSequence - ids.size - 1).coerceAtLeast(0)
        val stateIds = tok.encode(state.replace(tok.maskToken, " "), remaining)
        ids += stateIds
        ids += tok.sepId
        require(ids.size <= maxSequence) {
            "LAYA-QUESTION-003: طول ورودی نامعتبر است؛ total=${ids.size}, max_len=$maxSequence, head=${ids.size - stateIds.size - 1}"
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
                // Official Laya ONNX outputs are logits [1, options] and act_logits [1, 2].
                // Match rl_agent_api.py temperature bucket naming and entropy confidence.
                val bucket = when {
                    options.size <= 2 -> "choice:2"
                    options.size <= 5 -> "choice:3-5"
                    options.size <= 10 -> "choice:6-10"
                    else -> "choice:11+"
                }
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
                val entropy = -probs.sumOf { p -> if (p > 0.0) p * kotlin.math.ln(p) else 0.0 }
                val confidence = (1.0 - entropy / kotlin.math.ln(options.size.toDouble())).coerceIn(0.0, 1.0)
                val actTensor = result.toList().getOrNull(1)?.value as? OnnxTensor
                val actSummary = actTensor?.let { tensor ->
                    val values = tensor.floatBuffer.let { b -> FloatArray(b.remaining()).also(b::get) }
                    if (values.size >= 2 && values.all { it.isFinite() }) {
                        val m = maxOf(values[0].toDouble(), values[1].toDouble())
                        val e0 = kotlin.math.exp(values[0] - m)
                        val e1 = kotlin.math.exp(values[1] - m)
                        " actProbability=${"%.3f".format(Locale.US, e1 / (e0 + e1))}"
                    } else ""
                }.orEmpty()
                val top = ranked.take(3).joinToString(";") { index ->
                    "${options[index].id}:logit=${"%.4f".format(Locale.US, logits[index])},p=${"%.4f".format(Locale.US, probs[index])}"
                }
                val detail = "stage=${instructions.take(48).replace(Regex("\\s+"), " ")} options=${options.size} inputTokens=${ids.size} stateTokens=${stateIds.size} headTokens=${ids.size - stateIds.size - 1} temperature=${"%.3f".format(Locale.US, activeTemperature)} calibration=$activeCalibration confidence=${"%.3f".format(Locale.US, confidence)} margin=${"%.3f".format(Locale.US, margin)}$actSummary top=[$top]"
                lastTrace = lastTrace + detail
                if (confidence < MIN_CONFIDENCE || margin < MIN_MARGIN) {
                    lastError = "LAYA-INFER-005: تصمیم کم‌اطمینان؛ confidence=${"%.3f".format(Locale.US, confidence)}, maxProbability=${"%.3f".format(Locale.US, probs[best])}, margin=${"%.3f".format(Locale.US, margin)}, threshold=${MIN_CONFIDENCE}/${MIN_MARGIN}; $detail"
                    return null
                }
                return Ranked(options[best].id, confidence)
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
        val tokenizerConfigFile = File(root, "tokenizer/tokenizer_config.json")
        val key = "${model.absolutePath}:${model.length()}:${model.lastModified()}:${File(root, "$MODEL_FILE.data").length()}:${configFile.lastModified()}:${tokenizerFile.lastModified()}:${tokenizerConfigFile.lastModified()}"
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
        tokenizer = LayaTokenizer(File(root, "tokenizer/tokenizer.json"), tokenizerConfigFile)
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
        runCatching { tokenizer?.close() }
        tokenizer = null
        cachedModelKey = null
    }

    override fun close() = synchronized(lock) {
        closeLoaded()
        environment = null
    }
}

/**
 * Adapter over the official Hugging Face Tokenizers Rust pipeline.
 * tokenizer.json is authoritative; no custom Persian normalization or algorithm fallback.
 */
private class LayaTokenizer(tokenizerFile: File, tokenizerConfigFile: File) : AutoCloseable {
    private val native = LayaNativeTokenizer(tokenizerFile.readBytes())
    private val config = JSONObject(tokenizerConfigFile.readText(Charsets.UTF_8))

    private fun configuredToken(name: String): String {
        val raw = config.opt(name)
        val value = when (raw) {
            is String -> raw
            is JSONObject -> raw.optString("content").ifBlank { raw.optString("token") }
            else -> ""
        }
        require(value.isNotBlank()) {
            "LAYA-TOKENIZER-012: توکن ویژه '$name' در tokenizer_config.json وجود ندارد."
        }
        return value
    }

    val clsToken: String = configuredToken("cls_token")
    val sepToken: String = configuredToken("sep_token")
    val maskToken: String = configuredToken("mask_token")
    val clsId: Long = native.tokenId(clsToken)
        ?: error("LAYA-TOKENIZER-013: شناسهٔ cls_token در tokenizer.json پیدا نشد: $clsToken")
    val sepId: Long = native.tokenId(sepToken)
        ?: error("LAYA-TOKENIZER-014: شناسهٔ sep_token در tokenizer.json پیدا نشد: $sepToken")
    val maskId: Long = native.tokenId(maskToken)
        ?: error("LAYA-TOKENIZER-015: شناسهٔ mask_token در tokenizer.json پیدا نشد: $maskToken")

    fun encode(raw: String, maxTokens: Int): List<Long> {
        if (maxTokens <= 0 || raw.isEmpty()) return emptyList()
        return native.encode(raw, addSpecialTokens = false).asList().take(maxTokens)
    }

    override fun close() = native.close()
}
