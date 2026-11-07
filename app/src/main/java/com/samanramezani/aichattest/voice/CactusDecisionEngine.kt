package com.samanramezani.aichattest.voice

import org.json.JSONArray
import org.json.JSONObject
import java.util.Locale
import java.io.File

/** Needle is the only semantic router for voice actions; no regex classifier runs beside it. */
internal class CactusDecisionEngine(private val modelFile: File) : AutoCloseable {
    data class Decision(
        val command: DinaCommand,
        val route: String,
        val confidence: Double,
        val trace: List<String>,
    )

    @Volatile var lastError: String? = null
        private set
    @Volatile var lastTrace: List<String> = emptyList()
        private set
    /** True when Needle emitted at least one tool call, even if it was rejected as unsafe/invalid. */
    @Volatile var lastToolSelected: Boolean = false
        private set

    private val lock = Any()
    private var initialized = false
    private var loadedLastModified = 0L
    private var loadedLength = 0L

    companion object {
        private const val MIN_READ_ONLY_CONFIDENCE = 0.50
        private const val MIN_MUTATION_CONFIDENCE = 0.70

        // Needle treats the system turn as environment facts, not behavioural instructions.
        private val systemFacts = "locale: fa-IR; device: Android phone"

        private val toolsJson: String by lazy {
            val tools = JSONArray()
            tools.put(tool(
                "get_clock",
                "Read the current local time or today's date on this Android phone; Persian requests include ساعت الان چنده and تاریخ امروز چنده.",
                JSONObject().put("action", enumProperty("time", "date")),
                JSONArray().put("action"),
            ))
            tools.put(tool(
                "adjust_volume",
                "Increase or decrease this phone's media volume, or mute and unmute its media audio. Persian phrases include بلندتر، کم‌تر، قطع صدا، وصل صدا.",
                JSONObject().put("action", enumProperty("increase", "decrease", "mute", "unmute")),
                JSONArray().put("action"),
            ))
            tools.put(tool(
                "set_volume",
                "Set this phone's media volume to a stated 0–10 level. Persian requests include تنظیم صدا روی عدد یا درصد مشخص.",
                JSONObject().put("value", integerProperty("Target media volume level, 0 to 10. A bare 0-10 is a level; an explicit percentage or bare number above 10 is rounded to the nearest 10% step.", 0, 10)),
                JSONArray().put("value"),
            ))
            tools.put(tool(
                "adjust_brightness",
                "Increase or decrease this phone's screen brightness. Persian phrases include روشن‌تر، کم‌نورتر، نور صفحه را زیاد کن، نور صفحه را کم کن.",
                JSONObject().put("action", enumProperty("increase", "decrease")),
                JSONArray().put("action"),
            ))
            tools.put(tool(
                "set_brightness",
                "Set this phone's screen brightness to a stated 0–10 level. Persian requests include تنظیم نور صفحه روی عدد یا درصد مشخص.",
                JSONObject().put("value", integerProperty("Target screen brightness level, 0 to 10. A bare 0-10 is a level; an explicit percentage or bare number above 10 is rounded to the nearest 10% step.", 0, 10)),
                JSONArray().put("value"),
            ))
            tools.toString()
        }

        private fun tool(name: String, description: String, properties: JSONObject, required: JSONArray) =
            JSONObject()
                .put("name", name)
                .put("description", description)
                .put("parameters", JSONObject()
                    .put("type", "object")
                    .put("properties", properties)
                    .put("required", required))

        private fun enumProperty(vararg values: String) = JSONObject()
            .put("type", "string")
            .put("enum", JSONArray(values.toList()))

        private fun integerProperty(description: String, minimum: Int, maximum: Int) = JSONObject()
            .put("type", "integer")
            .put("minimum", minimum)
            .put("maximum", maximum)
            .put("description", description)
    }

    fun isReady(): Boolean = modelFile.isFile && modelFile.length() >= 1_000_000L

    fun classify(rawText: String): Decision? = synchronized(lock) {
        lastTrace = emptyList()
        lastError = null
        lastToolSelected = false
        if (rawText.isBlank()) {
            lastTrace = listOf("inputChars=0 reason=blank-user-input")
            return@synchronized null
        }
        if (!isReady()) {
            lastError = "CACTUS-MODEL-001: فایل Cactus Needle 3 نصب نشده یا ناقص است؛ از تنظیمات مکالمه صوتی دانلود یا واردش کن."
            return@synchronized null
        }

        try {
            ensureInitialized()
            CactusNeedleNative.resetNative()
            val output = JSONObject(CactusNeedleNative.completeNative(rawText, 160))
            val confidence = if (output.isNull("confidence")) 0.0 else output.optDouble("confidence", 0.0)
            val trace = mutableListOf(
                "engine=Cactus Needle 3 inputChars=${rawText.length}",
                "responseType=${output.optString("type", "missing")} confidence=${"%.3f".format(Locale.US, confidence)}",
            )
            output.optString("reasoning").takeIf { it.isNotBlank() }?.let {
                trace += "reasoning=${it.take(500)}"
            }
            if (output.has("prefill_tps")) trace += "prefill_tps=${output.optDouble("prefill_tps")}"
            if (output.has("decode_tps")) trace += "decode_tps=${output.optDouble("decode_tps")}"
            if (output.has("peak_ram_mb")) trace += "peak_ram_mb=${output.optDouble("peak_ram_mb")}"

            val calls = output.optJSONArray("function_calls") ?: JSONArray()
            // A rejected/ambiguous tool call is still a tool decision: never reinterpret it as chat.
            lastToolSelected = output.optString("type") == "call" && calls.length() > 0
            val suppressedCalls = output.optJSONArray("suppressed_calls")
            if (suppressedCalls != null && suppressedCalls.length() > 0) {
                trace += "route=CHAT reason=needle-suppressed-call"
                lastTrace = trace
                return@synchronized null
            }
            if (output.optString("type") != "call" || !output.optBoolean("success", true) || calls.length() != 1) {
                trace += "route=CHAT reason=no-single-grounded-tool-call"
                lastTrace = trace
                return@synchronized null
            }
            val call = calls.optJSONObject(0) ?: run {
                lastTrace = trace + "route=CHAT reason=malformed-tool-call"
                return@synchronized null
            }
            val name = call.optString("name")
            val args = call.optJSONObject("arguments") ?: JSONObject()
            val readOnly = name == "get_clock"
            val minimum = if (readOnly) MIN_READ_ONLY_CONFIDENCE else MIN_MUTATION_CONFIDENCE
            if (confidence < minimum) {
                trace += "route=CHAT reason=confidence-below-$minimum"
                lastTrace = trace
                return@synchronized null
            }

            val command = when (name) {
                "get_clock" -> when (args.optString("action")) {
                    "time" -> DinaCommand(DinaIntent.GET_TIME)
                    "date" -> DinaCommand(DinaIntent.GET_DATE)
                    else -> return@synchronized null
                }
                "adjust_volume" -> when (args.optString("action")) {
                    "increase" -> DinaCommand(DinaIntent.INCREASE_VOLUME)
                    "decrease" -> DinaCommand(DinaIntent.DECREASE_VOLUME)
                    "mute" -> DinaCommand(DinaIntent.MUTE)
                    "unmute" -> DinaCommand(DinaIntent.UNMUTE)
                    else -> return@synchronized null
                }
                "set_volume" -> {
                    val value = args.optInt("value", -1)
                    if (value !in 0..10) return@synchronized null
                    DinaCommand(DinaIntent.SET_VOLUME, value * 10, valueIsPercent = true)
                }
                "adjust_brightness" -> when (args.optString("action")) {
                    "increase" -> DinaCommand(DinaIntent.INCREASE_BRIGHTNESS)
                    "decrease" -> DinaCommand(DinaIntent.DECREASE_BRIGHTNESS)
                    else -> return@synchronized null
                }
                "set_brightness" -> {
                    val value = args.optInt("value", -1)
                    if (value !in 0..10) return@synchronized null
                    DinaCommand(DinaIntent.SET_BRIGHTNESS, value * 10, valueIsPercent = true)
                }
                else -> {
                    trace += "route=CHAT reason=unknown-tool:$name"
                    lastTrace = trace
                    return@synchronized null
                }
            }
            trace += "route=$name confidence=${"%.3f".format(Locale.US, confidence)}"
            lastTrace = trace
            Decision(command, name, confidence, trace)
        } catch (failure: Throwable) {
            lastError = "CACTUS-INFER-001: اجرای محلی Needle ناموفق بود: ${failure.message ?: failure.javaClass.simpleName}"
            lastTrace = listOf("engine=Cactus Needle 3", "error=$lastError")
            null
        }
    }

    private fun ensureInitialized() {
        val modified = modelFile.lastModified()
        val length = modelFile.length()
        if (initialized && loadedLastModified == modified && loadedLength == length) return
        initialized = false
        check(CactusNeedleNative.loadModelNative(modelFile.absolutePath, modified, length)) {
            "بارگذاری فایل needle3.cact ناموفق بود."
        }
        val prefixTokens = CactusNeedleNative.initNative(systemFacts, toolsJson)
        check(prefixTokens >= 0) { "ساخت زمینه ابزارهای Needle ناموفق بود." }
        loadedLastModified = modified
        loadedLength = length
        initialized = true
    }

    override fun close() {
        synchronized(lock) {
            if (initialized) runCatching { CactusNeedleNative.resetNative() }
            initialized = false
        }
    }
}

/** A tool selection is never re-routed to the conversational LLM. */
internal fun shouldForwardNeedleRequestToChat(toolSelected: Boolean): Boolean = !toolSelected

private object CactusNeedleNative {
    init {
        System.loadLibrary("cactus_needle_jni")
    }

    external fun loadModelNative(path: String, lastModified: Long, length: Long): Boolean
    external fun initNative(systemPrompt: String, toolsJson: String): Int
    external fun completeNative(input: String, maxNewTokens: Int): String
    external fun resetNative()
}
