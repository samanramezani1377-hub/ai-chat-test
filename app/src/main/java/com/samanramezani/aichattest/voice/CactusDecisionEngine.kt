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

    private val lock = Any()
    private var initialized = false
    private var loadedLastModified = 0L
    private var loadedLength = 0L

    companion object {
        private const val MIN_READ_ONLY_CONFIDENCE = 0.50
        private const val MIN_MUTATION_CONFIDENCE = 0.70

        private val systemPrompt = """
            You are the on-device Persian voice-action router for an Android phone.
            Select a tool only when the user's current utterance explicitly requests
            an action on this phone. The user's language is primarily Persian (Farsi).
            Do not answer conversational questions. For greetings, knowledge questions,
            quoted commands, negated commands, hypothetical requests, other devices,
            ambiguous intent, or anything not covered by the tools, return no tool call.
            Never infer a numeric target that the user did not say. For set_volume and
            set_brightness, "set" requires an explicit level from 0 through 10.
            An explicitly stated percentage (including 7% or 10%) must be rounded to
            the nearest 10% level (7% becomes level 1; 47% becomes level 5). Only a bare
            number without a percent marker from 0 to 10 is already a level (7 means 70%).
            A bare number above 10 through 100 is treated as a percentage (40 becomes level 4).
            Never guess a missing value. Require high confidence for every state-changing
            action; if confidence is not high enough, abstain and let the chat model handle
            the unchanged utterance.
            "Increase" and "decrease" do not require a value.
            Mute and unmute are volume actions. Do not confuse current time with work
            hours, durations, event times, or changing the clock. get_date means today's
            date only, not a historical or future date. Never call more than one tool
            for one utterance. Be conservative: abstain rather than guess.
        """.trimIndent()

        private val toolsJson: String by lazy {
            val tools = JSONArray()
            tools.put(tool(
                "get_time",
                "Return the current local time of this Android phone. Use only when the user directly asks what time it is now.",
                JSONObject(),
                JSONArray(),
            ))
            tools.put(tool(
                "get_date",
                "Return today's local date. Use only when the user directly asks for today's date.",
                JSONObject(),
                JSONArray(),
            ))
            val volumeProperties = JSONObject()
                .put("action", enumProperty("increase", "decrease", "set", "mute", "unmute"))
                .put("value", integerProperty("Explicit target media volume level, 0 to 10. A bare 0-10 is a level; any explicit percentage or bare number above 10 is rounded to the nearest 10% step.", 0, 10))
            tools.put(tool(
                "set_volume",
                "Change this phone's media volume. Persian requests include بلندتر، کم‌تر، قطع صدا، وصل صدا or تنظیم روی درصد. Never infer value.",
                volumeProperties,
                JSONArray().put("action"),
            ))
            val brightnessProperties = JSONObject()
                .put("action", enumProperty("increase", "decrease", "set"))
                .put("value", integerProperty("Explicit target screen brightness level, 0 to 10. A bare 0-10 is a level; any explicit percentage or bare number above 10 is rounded to the nearest 10% step.", 0, 10))
            tools.put(tool(
                "set_brightness",
                "Change this phone's screen brightness. Use only for an explicit command to increase, decrease or set the display brightness.",
                brightnessProperties,
                JSONArray().put("action"),
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
            val readOnly = name == "get_time" || name == "get_date"
            val minimum = if (readOnly) MIN_READ_ONLY_CONFIDENCE else MIN_MUTATION_CONFIDENCE
            if (confidence < minimum) {
                trace += "route=CHAT reason=confidence-below-$minimum"
                lastTrace = trace
                return@synchronized null
            }

            val command = when (name) {
                "get_time" -> DinaCommand(DinaIntent.GET_TIME)
                "get_date" -> DinaCommand(DinaIntent.GET_DATE)
                "set_volume" -> when (args.optString("action")) {
                    "increase" -> DinaCommand(DinaIntent.INCREASE_VOLUME)
                    "decrease" -> DinaCommand(DinaIntent.DECREASE_VOLUME)
                    "mute" -> DinaCommand(DinaIntent.MUTE)
                    "unmute" -> DinaCommand(DinaIntent.UNMUTE)
                    "set" -> {
                        val value = args.optInt("value", -1)
                        if (value !in 0..10) return@synchronized null
                        DinaCommand(DinaIntent.SET_VOLUME, value * 10, valueIsPercent = true)
                    }
                    else -> return@synchronized null
                }
                "set_brightness" -> when (args.optString("action")) {
                    "increase" -> DinaCommand(DinaIntent.INCREASE_BRIGHTNESS)
                    "decrease" -> DinaCommand(DinaIntent.DECREASE_BRIGHTNESS)
                    "set" -> {
                        val value = args.optInt("value", -1)
                        if (value !in 0..10) return@synchronized null
                        DinaCommand(DinaIntent.SET_BRIGHTNESS, value * 10, valueIsPercent = true)
                    }
                    else -> return@synchronized null
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
        val prefixTokens = CactusNeedleNative.initNative(systemPrompt, toolsJson)
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

private object CactusNeedleNative {
    init {
        System.loadLibrary("cactus_needle_jni")
    }

    external fun loadModelNative(path: String, lastModified: Long, length: Long): Boolean
    external fun initNative(systemPrompt: String, toolsJson: String): Int
    external fun completeNative(input: String, maxNewTokens: Int): String
    external fun resetNative()
}
