package com.samanramezani.aichattest.voice

import org.json.JSONArray
import org.json.JSONObject
import java.util.Locale
import java.io.File

/** Needle is the only semantic router for voice actions; triggers are part of its official tool schema. */
internal class CactusDecisionEngine(private val modelFile: File) : AutoCloseable {
    data class Decision(
        val commands: List<DinaCommand>,
        val route: String,
        val confidence: Double,
        val trace: List<String>,
    )

    @Volatile var lastError: String? = null
        private set
    @Volatile var lastTrace: List<String> = emptyList()
        private set
    @Volatile var lastToolSelected: Boolean = false
        private set
    @Volatile var lastExplicitAbstention: Boolean = false
        private set

    private val lock = Any()
    private var initialized = false
    private var loadedLastModified = 0L
    private var loadedLength = 0L

    companion object {
        private const val MIN_READ_ONLY_CONFIDENCE = 0.50
        private const val MIN_MUTATION_CONFIDENCE = 0.70
        private const val systemFacts = "locale: fa-IR; device: Android phone"

        // Needle 3 has no retrieval head: keep the complete tool catalogue at five tools or fewer.
        private val toolsJson: String by lazy {
            val tools = JSONArray()
            tools.put(tool(
                "get_clock",
                "Read the current local time or today's date on this Android phone. The action is time for ساعت چنده، الان ساعت چنده، چه ساعتیه and date for تاریخ امروز چنده، امروز چندمه.",
                JSONObject().put("action", enumProperty("time", "date")),
                JSONArray().put("action"),
                arrayOf(
                    "(?:ساعت\\s*(?:چنده|چند(?:ه| است)?|الان)|الان\\s+ساعت|چه\\s+ساعتی(?:ه| است)?|زمان\\s+فعلی)",
                    "(?:تاریخ\\s+امروز|امروز\\s*(?:چندمه|چه\\s+تاریخ(?:یه| است)?)|تاریخ\\s*(?:چنده|رو\\s+بگو))",
                ),
            ))
            tools.put(tool(
                "adjust_volume",
                "Change this phone's media volume by one device step, or mute or unmute media audio. The action is increase, decrease, mute, or unmute.",
                JSONObject().put("action", enumProperty("increase", "decrease", "mute", "unmute")),
                JSONArray().put("action"),
                arrayOf(
                    "(?:(?:صدا|ولوم|بلندی\\s+صدا).{0,20}(?:زیادتر|بیشتر|بلندتر|زیاد\\s+کن|بالا\\s+ببر)|(?:زیادتر|بیشتر|بلندتر).{0,16}(?:صدا|ولوم))",
                    "(?:(?:صدا|ولوم|بلندی\\s+صدا).{0,20}(?:کمتر|آروم\\s+تر|پایین|کم\\s+کن|کاهش)|(?:کمتر|آروم\\s+تر).{0,16}(?:صدا|ولوم))",
                    "(?:قطع\\s+صدا|صدا\\s+رو\\s+قطع|بی.?صدا\\s+کن|ساکت\\s+کن|mute)",
                    "(?:وصل\\s+صدا|صدا\\s+رو\\s+وصل|از\\s+بی.?صدا\\s+درآر|unmute)",
                ),
            ))
            tools.put(tool(
                "set_volume",
                "Set this phone's media volume to the value stated by the user. value is 0 to 100; is_percent is true only when the request explicitly says percent or درصد, otherwise false.",
                JSONObject()
                    .put("value", integerProperty("The exact target number stated in the request, from 0 to 100.", 0, 100))
                    .put("is_percent", JSONObject().put("type", "boolean").put("description", "True only when the user explicitly says percent or درصد; false for a bare device-step number.")),
                JSONArray().put("value").put("is_percent"),
                arrayOf(
                    "(?:صدا|ولوم|بلندی\\s+صدا).{0,24}(?:روی|رو|به|بذار|بزار|تنظیم).{0,18}(?:[0-9۰-۹٠-٩]+|صفر|یک|یه|دو|سه|چهار|پنج|شش|هفت|هشت|نه|ده|بیست|سی|چهل|پنجاه|شصت|هفتاد|هشتاد|نود|صد)",
                ),
            ))
            tools.put(tool(
                "adjust_brightness",
                "Increase or decrease this phone's screen brightness by one step. The action is increase or decrease.",
                JSONObject().put("action", enumProperty("increase", "decrease")),
                JSONArray().put("action"),
                arrayOf(
                    "(?:(?:نور\\s+صفحه|روشنایی\\s+صفحه|نور\\s+نمایشگر).{0,20}(?:زیادتر|بیشتر|روشن.?تر|زیاد\\s+کن|بالا\\s+ببر)|(?:زیادتر|بیشتر|روشن.?تر).{0,18}(?:نور\\s+صفحه|روشنایی))",
                    "(?:(?:نور\\s+صفحه|روشنایی\\s+صفحه|نور\\s+نمایشگر).{0,20}(?:کمتر|تاریک.?تر|کم\\s+کن|پایین\\s+بیار)|(?:کمتر|تاریک.?تر).{0,18}(?:نور\\s+صفحه|روشنایی))",
                ),
            ))
            tools.put(tool(
                "set_brightness",
                "Set this phone's screen brightness to the percentage stated by the user, from 0 to 100.",
                JSONObject().put("value", integerProperty("The exact target brightness percentage stated in the request, from 0 to 100.", 0, 100)),
                JSONArray().put("value"),
                arrayOf(
                    "(?:نور\\s+صفحه|روشنایی\\s+صفحه|نور\\s+نمایشگر).{0,24}(?:روی|رو|به|بذار|بزار|تنظیم).{0,18}(?:[0-9۰-۹٠-٩]+|صفر|یک|یه|دو|سه|چهار|پنج|شش|هفت|هشت|نه|ده|بیست|سی|چهل|پنجاه|شصت|هفتاد|هشتاد|نود|صد)",
                ),
            ))
            tools.toString()
        }

        private fun tool(name: String, description: String, properties: JSONObject, required: JSONArray, triggers: Array<String>) =
            JSONObject()
                .put("name", name)
                .put("description", description)
                .put("triggers", JSONArray(triggers.toList()))
                .put("parameters", JSONObject().put("type", "object").put("properties", properties).put("required", required))

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
        lastExplicitAbstention = false
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
            val rawOutput = CactusNeedleNative.completeNative(rawText, 192)
            val output = JSONObject(rawOutput)
            val confidence = if (output.isNull("confidence")) 0.0 else output.optDouble("confidence", 0.0)
            val calls = output.optJSONArray("function_calls") ?: JSONArray()
            val suppressedCalls = output.optJSONArray("suppressed_calls") ?: JSONArray()
            lastToolSelected = calls.length() > 0 || suppressedCalls.length() > 0

            val trace = mutableListOf(
                "engine=Cactus Needle 3 inputChars=${rawText.length}",
                "responseType=${output.optString("type", "missing")} success=${output.optBoolean("success", false)} confidence=${"%.3f".format(Locale.US, confidence)}",
                "functionCalls=${calls.toString().take(1200)}",
                "suppressedCalls=${suppressedCalls.toString().take(1200)}",
                "responseJson=${output.toString().take(2400)}",
            )
            output.optString("reasoning").takeIf { it.isNotBlank() }?.let { trace += "reasoning=${it.take(500)}" }
            if (output.has("prefill_tps")) trace += "prefill_tps=${output.optDouble("prefill_tps")}"
            if (output.has("decode_tps")) trace += "decode_tps=${output.optDouble("decode_tps")}"
            if (output.has("peak_ram_mb")) trace += "peak_ram_mb=${output.optDouble("peak_ram_mb")}"

            val responseType = output.optString("type")
            val success = output.optBoolean("success", false)
            if (suppressedCalls.length() > 0) {
                lastToolSelected = true
                trace += "route=CONFIRM reason=needle-suppressed-call"
                lastTrace = trace
                return@synchronized null
            }
            if (calls.length() == 0) {
                if (responseType == "call" && success) {
                    lastExplicitAbstention = true
                    trace += "route=CHAT reason=explicit-empty-function-calls"
                } else {
                    lastError = "CACTUS-PROTOCOL-001: پاسخ Needle با قرارداد رسمی سازگار نبود؛ درخواست به مدل اصلی ارسال نشد."
                    trace += "route=ERROR reason=invalid-empty-response"
                }
                lastTrace = trace
                return@synchronized null
            }
            if (responseType != "call" || !success) {
                lastError = "CACTUS-PROTOCOL-002: Needle ابزار انتخاب کرد اما پاسخ call موفق و معتبر نبود."
                trace += "route=CONFIRM reason=invalid-call-envelope"
                lastTrace = trace
                return@synchronized null
            }
            if (calls.length() > 5) {
                lastError = "CACTUS-PROTOCOL-003: Needle بیش از پنج فراخوانی ابزار در یک نوبت برگرداند."
                trace += "route=CONFIRM reason=too-many-tool-calls"
                lastTrace = trace
                return@synchronized null
            }

            val commands = mutableListOf<DinaCommand>()
            val routes = mutableListOf<String>()
            for (index in 0 until calls.length()) {
                val call = calls.optJSONObject(index) ?: run {
                    lastError = "CACTUS-PROTOCOL-004: یکی از فراخوانی‌های ابزار ساختار JSON معتبری ندارد."
                    trace += "route=CONFIRM reason=malformed-tool-call index=$index"
                    lastTrace = trace
                    return@synchronized null
                }
                val name = call.optString("name")
                val args = call.optJSONObject("arguments") ?: JSONObject()
                val command = when (name) {
                    "get_clock" -> when (args.optString("action")) {
                        "time" -> DinaCommand(DinaIntent.GET_TIME)
                        "date" -> DinaCommand(DinaIntent.GET_DATE)
                        else -> null
                    }
                    "adjust_volume" -> when (args.optString("action")) {
                        "increase" -> DinaCommand(DinaIntent.INCREASE_VOLUME)
                        "decrease" -> DinaCommand(DinaIntent.DECREASE_VOLUME)
                        "mute" -> DinaCommand(DinaIntent.MUTE)
                        "unmute" -> DinaCommand(DinaIntent.UNMUTE)
                        else -> null
                    }
                    "set_volume" -> {
                        val value = args.optInt("value", -1)
                        if (value in 0..100 && args.has("is_percent") && !args.isNull("is_percent")) {
                            DinaCommand(DinaIntent.SET_VOLUME, value, args.optBoolean("is_percent"))
                        } else null
                    }
                    "adjust_brightness" -> when (args.optString("action")) {
                        "increase" -> DinaCommand(DinaIntent.INCREASE_BRIGHTNESS)
                        "decrease" -> DinaCommand(DinaIntent.DECREASE_BRIGHTNESS)
                        else -> null
                    }
                    "set_brightness" -> {
                        val value = args.optInt("value", -1)
                        if (value in 0..100) DinaCommand(DinaIntent.SET_BRIGHTNESS, value, valueIsPercent = true) else null
                    }
                    else -> null
                }
                if (command == null) {
                    lastError = "CACTUS-TOOL-VALIDATION-001: ابزار یا پارامتر برگشتی معتبر نیست: $name"
                    trace += "route=CONFIRM reason=invalid-tool-or-arguments tool=$name"
                    lastTrace = trace
                    return@synchronized null
                }
                commands += command
                routes += name
            }

            val readOnly = commands.all { it.intent == DinaIntent.GET_TIME || it.intent == DinaIntent.GET_DATE }
            val minimum = if (readOnly) MIN_READ_ONLY_CONFIDENCE else MIN_MUTATION_CONFIDENCE
            if (confidence < minimum) {
                trace += "route=CONFIRM reason=confidence-below-$minimum"
                lastTrace = trace
                return@synchronized null
            }

            trace += "route=${routes.joinToString(",")} confidence=${"%.3f".format(Locale.US, confidence)} commands=${commands.size}"
            lastTrace = trace
            Decision(commands, routes.joinToString(","), confidence, trace)
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

internal fun shouldForwardNeedleRequestToChat(explicitAbstention: Boolean, toolSelectedOrSuppressed: Boolean): Boolean =
    explicitAbstention && !toolSelectedOrSuppressed

private object CactusNeedleNative {
    init { System.loadLibrary("cactus_needle_jni") }
    external fun loadModelNative(path: String, lastModified: Long, length: Long): Boolean
    external fun initNative(systemPrompt: String, toolsJson: String): Int
    external fun completeNative(input: String, maxNewTokens: Int): String
    external fun resetNative()
}
