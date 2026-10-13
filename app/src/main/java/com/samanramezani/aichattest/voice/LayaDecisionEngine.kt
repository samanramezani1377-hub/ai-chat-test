package com.samanramezani.aichattest.voice

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale
import kotlin.math.exp


/**
 * Laya must classify the latest user turn, never a serialized conversation transcript.
 * The normal voice path passes one utterance; this boundary defensively strips old turns
 * if a transcript was accidentally supplied upstream. It does not infer user intent.
 */
internal class LayaDecisionEngine(private val root: File) : AutoCloseable {
    data class Decision(val command: DinaCommand, val route: String, val confidence: Double, val trace: List<String>)

    companion object {
        private const val MODEL_FILE = LayaDecisionModelStore.MODEL_FILE
        private const val DEFAULT_MAX_SEQUENCE = 1024
        private const val DEFAULT_HEAD_MAX_SEQUENCE = 256
        // Entropy confidence and top-class probability are distinct. Both must pass,
        // so diffuse predictions such as "سلام" -> GET_TIME (p=0.62, confidence=0.297)
        // abstain instead of triggering a local device action.
        private const val MIN_CONFIDENCE = 0.43
        private const val MIN_MAX_PROBABILITY = 0.43
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
    private var maxSequence = DEFAULT_MAX_SEQUENCE
    private var headMaxSequence = DEFAULT_HEAD_MAX_SEQUENCE
    private var choiceTemperature = 1.0
    private var temperatureSource = "missing; neutral fallback 1.0"
    private val optionTemperatures = mutableMapOf<String, Double>()

    fun missingModelFiles(): List<String> {
        val required = listOf(
            MODEL_FILE,
            "$MODEL_FILE.data",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
            "encoder/config.json",
            "rl_agent_config.json",
        )
        val missing = required.filter { relative ->
            val file = File(root, relative)
            !file.isFile || file.length() <= 0L
        }.toMutableList()
        val graph = File(root, MODEL_FILE)
        if (graph.isFile && graph.length() <= 16_000L) missing += "$MODEL_FILE (گراف ناقص)"
        val weights = File(root, "$MODEL_FILE.data")
        if (weights.isFile && weights.length() <= 100_000_000L) missing += "$MODEL_FILE.data (وزن‌ها ناقص)"
        return missing.distinct()
    }

    fun isReady(): Boolean = missingModelFiles().isEmpty()

    fun classify(rawText: String, currentVolume: Int? = null, maxVolume: Int? = null): Decision? = synchronized(lock) {
        lastTrace = emptyList()
        lastError = null
        // Preserve the complete user-authored text exactly for Laya inference.
        val decisionText = rawText
        if (decisionText.isBlank()) {
            lastTrace = listOf("inputChars=${rawText.length} modelInputChars=0 reason=blank-user-input")
            return@synchronized null
        }
        if (!isReady()) {
            closeLoaded()
            val missing = missingModelFiles()
            lastTrace = listOf(
                "modelRoot=${root.absolutePath}",
                "missingModelFiles=${missing.joinToString(",").ifBlank { "bundle validation failed" }}",
            )
            lastError = "LAYA-MODEL-001: بستهٔ Laya نصب یا کامل نیست؛ فایل‌های ناقص/مفقود: ${missing.joinToString(", ").ifBlank { "اعتبارسنجی بسته ناموفق بود" }}. از تنظیمات مکالمه صوتی «دانلود و نصب خودکار Laya INT4» را اجرا کن."
            return@synchronized null
        }
        val trace = mutableListOf<String>()
        var retainLoadedSession = false
        try {
            ensureLoaded()
            // Use actionable top-level choices instead of a broad INFORMATION/SYSTEM
            // split. The previous taxonomy confidently misrouted short Persian requests
            // such as "ساعت چنده؟" to SYSTEM, preventing the local clock/date path from running.
            // Laya still makes the decision; this is not a keyword/regex fallback.
            trace += "input originalChars=${rawText.length} modelInputChars=${decisionText.length} userTextPreserved=true"
            val domain = choose(
                decisionText,
                "وظیفه: فقط نیت اصلی و صریح کاربر را از روی کل پیام فعلی انتخاب کن؛ متن را خلاصه، اصلاح یا بازنویسی نکن و هیچ شماره‌گذاری یا بخشی از متن را حذف نکن. انتخاب باید بر اساس کاری باشد که کاربر می‌خواهد انجام شود، نه صرفاً واژه‌های موجود در پیام. اگر درخواست روشن و مستقیم برای ساعت فعلی، تاریخ امروز یا تغییر تنظیم دستگاه وجود ندارد، CHAT را انتخاب کن. نقل‌قول، مثال، نفی، فرضیه و پرسش دربارهٔ موضوعی با فرمان انجام آن موضوع یکی نیستند. فقط یک گزینه را انتخاب کن؛ اگر تردید داری CHAT امن‌تر است. Classify intent from the complete current user message, not keywords alone.",
                listOf(
                    Option("GET_TIME", "کاربر واقعاً ساعت همین لحظه را می‌پرسد: «ساعت چنده؟»، «الان ساعت رو بگو»، «می‌دونی الان چه ساعتیه؟». نه ساعت کاری، مدت‌زمان، زمان یک رویداد، تنظیم ساعت گوشی، یا صرفاً ذکر/نقل‌قول واژهٔ ساعت. Explicit request for the current clock time now."),
                    Option("GET_DATE", "کاربر تاریخ امروز را می‌خواهد: «امروز چندمه؟»، «تاریخ امروز رو بگو»، «امروز چه روزیه؟» وقتی منظور تاریخ/روز فعلی است. نه تاریخ یک رویداد، تاریخ تاریخی، روز هفتهٔ یک تاریخ دیگر یا برنامه‌ریزی آینده. Explicit request for today's current date."),
                    Option("AUDIO", "درخواست اجرایی برای تغییر صدای رسانهٔ همین گوشی: زیاد/کم کردن، تنظیم سطح، بی‌صدا کردن یا برگرداندن صدا؛ مثل «صدا رو کم کن». نه سؤال یا گفت‌وگو دربارهٔ صدا، صدای یک فایل، یا نقل‌قول فرمان. Explicit media-volume action on this device."),
                    Option("DISPLAY", "درخواست اجرایی برای تغییر روشنایی نمایشگر همین گوشی: زیاد/کم کردن یا تنظیم سطح؛ مثل «نور صفحه رو کم کن». نه سؤال دربارهٔ صفحه/روشنایی، تنظیمات دستگاهی دیگر، یا نقل‌قول فرمان. Explicit screen-brightness action on this device."),
                    Option("CHAT", "پیش‌فرض برای سلام و گفت‌وگوی عادی، دانش عمومی، راهنمایی/مسیر/آدرس/سفر، ساعت کاری، مدت‌زمان، تاریخ‌های غیرامروز، درخواست مبهم، نقل‌قول یا نفی فرمان؛ همچنین هر پیام که صریحاً یکی از عملیات محلی بالا را درخواست نمی‌کند. All other requests and uncertain cases."),
                ),
                minConfidence = 0.48,
                minMaxProbability = 0.52,
                minMargin = 0.12,
            ) ?: run {
                lastTrace = trace.toList() + lastTrace
                return@synchronized null
            }
            trace += "domain=${domain.id} confidence=${"%.3f".format(Locale.US, domain.confidence)}"

            // Independent second-stage confirmation is required for every local route,
            // including read-only time/date answers. The confirmation prompt is conditioned
            // on the exact first-stage route and contrasts it with realistic near-misses;
            // this avoids the old generic yes/no question that rejected valid time requests
            // while still guarding against false positives.
            if (domain.id != "CHAT") {
                val confirmation = when (domain.id) {
                    "GET_TIME" -> Triple(
                        "تأیید مستقل مسیر پیشنهادی GET_TIME: فقط بررسی کن آیا کل پیام فعلی واقعاً درخواست دانستن ساعت همین لحظه است. به قصد ارتباطی پیام توجه کن، نه حضور کلمهٔ «ساعت». «ساعت چنده؟» و «الان ساعت رو می‌گی؟» تأیید می‌شوند؛ «ساعت کاری چنده؟»، «دو ساعت طول می‌کشه؟»، «ساعت گوشی رو تنظیم کن»، نقل‌قول فرمان یا ذکر ساعت در مثال تأیید نمی‌شوند. اگر متن مبهم، نفی‌شده یا صرفاً نقل‌قول است، رد کن. This is a binary verification of the proposed route, not a fresh broad classification.",
                        Option("CONFIRM_ROUTE", "درخواست واقعی و مستقیم برای دانستن زمان فعلی همین لحظه وجود دارد؛ مثلاً «ساعت چنده؟»، «الان ساعت چند است؟»، «می‌تونی ساعت فعلی رو بگی؟». The current clock time is the requested answer."),
                        Option("REJECT_ROUTE", "درخواست ساعت فعلی وجود ندارد یا از متن با اطمینان قابل استنباط نیست؛ از جمله ساعت کاری، مدت‌زمان، زمان رویداد، تنظیم ساعت، نقل‌قول، نفی یا ابهام. Reject keyword-only matches.")
                    )
                    "GET_DATE" -> Triple(
                        "تأیید مستقل مسیر پیشنهادی GET_DATE: فقط بررسی کن آیا کل پیام فعلی واقعاً تاریخ امروز/تاریخ فعلی را می‌خواهد. «امروز چندمه؟» و «تاریخ امروز رو بگو» تأیید می‌شوند؛ ساعت فعلی، تاریخ یک رویداد یا مناسبت، تاریخ تاریخی، پرسش دربارهٔ روزی در آینده یا برنامه‌ریزی زمان‌بندی تأیید نمی‌شوند. اگر مبهم، نفی‌شده یا نقل‌قول است، رد کن. Verify the proposed intent, not keywords.",
                        Option("CONFIRM_ROUTE", "پیام به‌وضوح تاریخ امروز/تاریخ فعلی را می‌خواهد؛ مانند «امروز چندمه؟»، «تاریخ امروز رو بگو»، «امروز چه روزیه؟» با معنای روز فعلی. The requested answer is today's date."),
                        Option("REJECT_ROUTE", "تاریخ امروز درخواست نشده یا متن دربارهٔ تاریخ دیگری، یک رویداد، برنامهٔ آینده، نقل‌قول، نفی یا مورد مبهم است. Reject keyword-only matches.")
                    )
                    "AUDIO" -> Triple(
                        "تأیید مستقل مسیر پیشنهادی AUDIO: فقط بررسی کن آیا کاربر از دستیار می‌خواهد صدای رسانهٔ همین گوشی را تغییر دهد. درخواست‌های «صدا رو زیاد/کم کن»، «گوشی رو بی‌صدا کن» و «صدا رو برگردون» تأیید می‌شوند. سؤال دربارهٔ صدا، کیفیت صدای فایل، صحبت دربارهٔ موسیقی، فرمانی که نقل یا نفی شده، یا درخواست مبهم تأیید نمی‌شود. این مرحله فقط مسیر پیشنهادی را بررسی می‌کند.",
                        Option("CONFIRM_ROUTE", "کاربر از دستیار یک عمل روشن روی صدای رسانهٔ همین گوشی می‌خواهد: افزایش، کاهش، تنظیم سطح، قطع یا وصل صدا. Explicit action on this device's media volume."),
                        Option("REJECT_ROUTE", "درخواست اجرایی روشن برای صدای رسانهٔ همین گوشی وجود ندارد؛ موضوع صرفاً اطلاعاتی، نقل‌قولی، نفی‌شده یا مبهم است. No explicit device-volume action.")
                    )
                    "DISPLAY" -> Triple(
                        "تأیید مستقل مسیر پیشنهادی DISPLAY: فقط بررسی کن آیا کاربر از دستیار می‌خواهد روشنایی نمایشگر همین گوشی را تغییر دهد. «نور صفحه رو کم/زیاد کن» یا «روشنایی رو روی ۵۰ درصد بذار» تأیید می‌شوند. سؤال دربارهٔ فناوری صفحه، روشنایی یک عکس/اتاق، تنظیم دستگاهی دیگر، فرمان نقل‌شده یا نفی‌شده تأیید نمی‌شود. این مرحله فقط مسیر پیشنهادی را بررسی می‌کند.",
                        Option("CONFIRM_ROUTE", "کاربر یک عمل روشن برای کم/زیاد یا تنظیم درصد روشنایی نمایشگر همین گوشی درخواست می‌کند. Explicit action on this device's screen brightness."),
                        Option("REJECT_ROUTE", "درخواست اجرایی روشن برای تغییر روشنایی نمایشگر همین گوشی وجود ندارد؛ موضوع اطلاعاتی، مربوط به چیز دیگری، نقل‌قولی، نفی‌شده یا مبهم است. No explicit brightness action.")
                    )
                    else -> null
                }
                if (confirmation == null) {
                    lastTrace = trace.toList() + lastTrace
                    lastError = "LAYA-INFER-009: مسیر محلی ناشناخته در مرحلهٔ تأیید؛ درخواست به مدل گفت‌وگو سپرده شد."
                    return@synchronized null
                }
                val (instructions, confirmOption, rejectOption) = confirmation
                val validation = choose(
                    decisionText,
                    "$instructions\nمسیر پیشنهادی مرحلهٔ اول: ${domain.id}. تمام متن اصلی کاربر را بررسی کن؛ آن را حذف، خلاصه یا بازنویسی نکن. فقط تأیید کن که همین پیام، در معنای واقعی و نه صرفاً با داشتن واژهٔ مشابه، همین عملیات را درخواست می‌کند. از مکالمهٔ قبلی یا حدس دربارهٔ نیت استفاده نکن. اگر درخواست صریح نیست، REJECT_ROUTE را انتخاب کن. Output only the best matching option.",
                    listOf(confirmOption, rejectOption),
                    minConfidence = 0.50,
                    minMaxProbability = 0.58,
                    minMargin = 0.16,
                )
                if (validation?.id != "CONFIRM_ROUTE") {
                    trace += "route_confirmation=${validation?.id ?: "ABSTAIN"} proposed=${domain.id} result=CHAT"
                    lastTrace = trace.toList() + lastTrace
                    lastError = "LAYA-INFER-008: مسیر ${domain.id} در تأیید مستقل تأیید نشد؛ متن اصلی به مدل گفت‌وگو فرستاده می‌شود."
                    return@synchronized null
                }
                trace += "route_confirmation=CONFIRM_ROUTE proposed=${domain.id} confidence=${"%.3f".format(Locale.US, validation.confidence)}"
            }

            val command = when (domain.id) {
                "GET_TIME" -> DinaCommand(DinaIntent.GET_TIME)
                "GET_DATE" -> DinaCommand(DinaIntent.GET_DATE)
                "CHAT" -> DinaCommand(DinaIntent.CHAT)
                "AUDIO" -> {
                    val operation = choose(decisionText, "از متن کامل کاربر، عمل مشخصی را که برای صدای رسانهٔ همین گوشی درخواست شده انتخاب کن. منظور صدای رسانه است، نه صدای زنگ/تماس یا صدای یک فایل به‌عنوان موضوع گفتگو. اگر درخواست تغییر صدا روشن نیست، هیچ فرمانی را حدس نزن. Select the requested operation, not the topic.",
                        listOf(
                            Option("INCREASE", "کاربر صریحاً می‌خواهد صدای رسانهٔ گوشی بلندتر شود؛ «صدا رو زیاد کن»، «بلندترش کن». Increase media volume."),
                            Option("DECREASE", "کاربر صریحاً می‌خواهد صدای رسانهٔ گوشی پایین‌تر بیاید؛ «صدا رو کم کن»، «آروم‌ترش کن». Decrease media volume."),
                            Option("SET", "کاربر سطح هدف مشخصی می‌خواهد؛ «صدا رو روی ۴۰ درصد بذار»، «بلندی رو روی ۵ از ۱۰ تنظیم کن». Set an explicit target level."),
                            Option("MUTE", "کاربر می‌خواهد صدای رسانه قطع/بی‌صدا شود؛ «صدا رو قطع کن»، «بی‌صداش کن». Mute media audio."),
                            Option("UNMUTE", "کاربر می‌خواهد صدای رسانه دوباره فعال شود؛ «صدا رو وصل کن»، «از بی‌صدا درش بیار». Restore media audio."),
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
                                append(decisionText)
                                if (currentLevel != null) append("\nCurrent media volume level: ").append(currentLevel).append(" of 10.")
                            }
                            val level = choose(
                                state,
                                "فقط مقدار هدفی را که کاربر درخواست کرده استخراج کن؛ سطح فعلی فقط اطلاعات زمینه‌ای است و هدف محسوب نمی‌شود. درصد را به نزدیک‌ترین گام ۱۰درصدی تبدیل کن (مثلاً ۴۷٪ → سطح ۵ از ۱۰؛ ۸۰٪ → سطح ۸). اگر عدد روی مقیاس ۰ تا ۱۰ گفته شده همان سطح را بگیر. از «بیشتر/کمتر» به‌تنهایی مقدار هدف نساز؛ اگر مقدار دقیق گفته نشده یا قابل تشخیص نیست UNKNOWN را انتخاب کن. Determine the requested target, never mistake the current level for the target.",
                                (0..10).map { n ->
                                    Option("LEVEL_$n", "هدف صدای درخواستی $n از ۱۰، معادل حدود ${n * 10}٪ است. فقط وقتی انتخاب کن که متن کاربر همین مقدار هدف را بیان کند. Target media volume: $n/10 (about ${n * 10}%).")
                                } + Option("UNKNOWN", "کاربر سطح عددی مشخصی درخواست نکرده یا مقدار هدف از متن قابل تشخیص نیست؛ حدس نزن. No explicit numeric target.")
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
                    val operation = choose(decisionText, "از متن کامل کاربر، عمل مشخصی را که برای روشنایی نمایشگر همین گوشی درخواست شده انتخاب کن. اگر فقط دربارهٔ روشنایی صحبت می‌کند و تغییر نمی‌خواهد، هیچ عملی را حدس نزن. Select the requested operation, not merely the topic.",
                        listOf(
                            Option("INCREASE", "کاربر صریحاً می‌خواهد نمایشگر همین گوشی روشن‌تر شود؛ «نور صفحه رو زیاد کن». Increase screen brightness."),
                            Option("DECREASE", "کاربر صریحاً می‌خواهد نمایشگر همین گوشی کم‌نورتر شود؛ «نور صفحه رو کم کن». Decrease screen brightness."),
                            Option("SET", "کاربر درصد/سطح هدف مشخصی برای روشنایی می‌دهد؛ «روشنایی رو روی ۵۰ درصد بذار». Set an explicit target brightness."),
                        )) ?: return@synchronized null
                    trace += "display_operation=${operation.id} confidence=${"%.3f".format(Locale.US, operation.confidence)}"
                    when (operation.id) {
                        "INCREASE" -> DinaCommand(DinaIntent.INCREASE_BRIGHTNESS)
                        "DECREASE" -> DinaCommand(DinaIntent.DECREASE_BRIGHTNESS)
                        "SET" -> {
                            val target = choose(
                                decisionText,
                                "فقط مقدار هدف روشنایی را از متن کاربر استخراج کن؛ سطح فعلی یا درصدی که در مثال/نقل‌قول آمده هدف نیست. این پیاده‌سازی مقدار را به نزدیک‌ترین گام ۱۰درصدی تبدیل می‌کند (مثلاً ۴۷٪ → ۵۰٪). اگر کاربر فقط «بیشتر/کمتر» می‌گوید، SET را انتخاب نکن؛ آن درخواست باید در مرحلهٔ عملیات تشخیص داده شود. اگر درصد هدف روشن نیست، UNKNOWN را انتخاب کن. Extract the requested target, not current state; do not guess.",
                                (0..10).map { n ->
                                    Option("LEVEL_$n", "هدف روشنایی ${n * 10}٪ است. فقط وقتی انتخاب کن که متن کاربر این مقدار یا مقداری نزدیک به آن را درخواست کند. Target screen brightness: ${n * 10}%.")
                                } + Option("UNKNOWN", "درخواست درصدی مشخصی وجود ندارد یا هدف روشنایی از متن قابل تشخیص نیست؛ حدس نزن. No explicit numeric brightness target.")
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
            val fullTrace = trace.toList() + lastTrace
            lastTrace = fullTrace
            lastError = null
            // Laya's resident memory is large. Keep it warm only after local device actions;
            // close the session before the chat model is loaded, and also on every abstention.
            retainLoadedSession = command.intent != DinaIntent.CHAT
            Decision(command, domain.id, domain.confidence, fullTrace)
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

    private fun choose(
        state: String,
        instructions: String,
        options: List<Option>,
        minConfidence: Double = MIN_CONFIDENCE,
        minMaxProbability: Double = MIN_MAX_PROBABILITY,
        minMargin: Double = MIN_MARGIN,
    ): Ranked? {
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
            // Official build_sequence inserts a separator after every option, not just after the list.
            ids += tok.sepId
        }
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
                val logitsEntry = result.firstOrNull { it.key.equals("logits", ignoreCase = true) }
                    ?: result.firstOrNull()
                val output = logitsEntry?.value as? OnnxTensor
                    ?: error("LAYA-INFER-003: خروجی logits مدل پیدا نشد.")
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
                val activeTemperature = bucketTemperature?.takeIf { it.isFinite() && it > 0.0 } ?: choiceTemperature
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
                val actTensor = result.firstOrNull { it.key.equals("act_logits", ignoreCase = true) }?.value as? OnnxTensor
                val actSummary = actTensor?.let { tensor ->
                    val values = tensor.floatBuffer.let { b -> FloatArray(b.remaining()).also(b::get) }
                    if (values.size >= 2 && values.all { it.isFinite() }) {
                        val m = maxOf(values[0].toDouble(), values[1].toDouble())
                        val e0 = kotlin.math.exp(values[0] - m)
                        val e1 = kotlin.math.exp(values[1] - m)
                        " actProbability=${"%.3f".format(Locale.US, e0 / (e0 + e1))}"
                    } else ""
                }.orEmpty()
                val top = ranked.take(3).joinToString(";") { index ->
                    "${options[index].id}:logit=${"%.4f".format(Locale.US, logits[index])},p=${"%.4f".format(Locale.US, probs[index])}"
                }
                val detail = "stage=${instructions.take(48).replace(Regex("\\s+"), " ")} tokenizer=hf-tokenizers-rust clsId=${tok.clsId} sepId=${tok.sepId} maskId=${tok.maskId} qtype=choice(0) options=${options.size} inputTokens=${ids.size} stateTokens=${stateIds.size} headTokens=${ids.size - stateIds.size - 1} outputs=${active.outputNames.joinToString(",")} temperature=${"%.3f".format(Locale.US, activeTemperature)} calibration=$activeCalibration confidence=${"%.3f".format(Locale.US, confidence)} margin=${"%.3f".format(Locale.US, margin)}$actSummary top=[$top]"
                lastTrace = lastTrace + detail
                // Keep the app's conservative action gate on the top probability and margin.
                // Entropy confidence is the official Laya confidence metric, but it is not
                // interchangeable with the old max-probability threshold.
                if (confidence < minConfidence ||
                    probs[best] < minMaxProbability ||
                    margin < minMargin
                ) {
                    lastError = "LAYA-INFER-005: تصمیم کم‌اطمینان؛ confidence=${"%.3f".format(Locale.US, confidence)}, maxProbability=${"%.3f".format(Locale.US, probs[best])}, margin=${"%.3f".format(Locale.US, margin)}, confidenceThreshold=$minConfidence, maxProbabilityThreshold=$minMaxProbability, marginThreshold=$minMargin; $detail"
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
        lastTrace = lastTrace + "tokenizerBackend=HuggingFace-Tokenizers-Rust-0.22.1 clsId=${tokenizer!!.clsId} sepId=${tokenizer!!.sepId} maskId=${tokenizer!!.maskId} max_len=$maxSequence head_max_len=$headMaxSequence"
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
        choiceTemperature = if (selected.isFinite() && selected > 0.0) selected else 1.0
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
