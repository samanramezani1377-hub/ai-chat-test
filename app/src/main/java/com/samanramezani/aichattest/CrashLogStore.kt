package com.samanramezani.aichattest

import android.app.ActivityManager
import android.app.ApplicationExitInfo
import android.content.Context
import android.os.Build
import android.os.Process
import com.woogit.aicore.runtime.RuntimeDiagnosticsStore
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.io.File
import java.io.PrintWriter
import java.io.StringWriter
import java.security.MessageDigest
import java.text.DateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Persists fatal Java crashes and native fatal-signal records so they survive
 * process death and can be copied from the in-app Crash Log screen.
 *
 * A signal/last checkpoint is evidence, not a guaranteed root cause. The report
 * deliberately separates confirmed facts from likely explanations.
 */
object CrashLogStore {
    private const val PREFS = "crash_log_store"
    private const val KEY_NATIVE_FINGERPRINT = "last_native_fingerprint"
    private const val KEY_PLATFORM_EXIT_TIMESTAMP = "last_platform_exit_timestamp"
    private const val KEY_PENDING_PLATFORM_EXIT = "pending_platform_exit"
    private const val KEY_PENDING_PLATFORM_TRACE = "pending_platform_trace"
    private const val FILE_NAME = "crash-log.txt"
    private const val MAX_REPORT_CHARS = 40_000
    private val installed = AtomicBoolean(false)
    private val mutableReport = MutableStateFlow<String?>(null)
    val report: StateFlow<String?> = mutableReport.asStateFlow()

    fun initialize(context: Context) {
        val app = context.applicationContext
        capturePlatformExitInfo(app)
        capturePendingNativeCrash(app)
        mutableReport.value = readReport(app)
        if (installed.compareAndSet(false, true)) installJavaCrashHandler(app)
    }

    fun refresh(context: Context) {
        capturePlatformExitInfo(context.applicationContext)
        capturePendingNativeCrash(context.applicationContext)
        mutableReport.value = readReport(context.applicationContext)
    }

    fun recordRuntimeContext(context: Context, modelName: String?, runtimeStatus: String, inferenceRunning: Boolean) {
        runCatching {
            val snapshot = RuntimeDiagnosticsStore.snapshot.value
            val prefs = context.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            prefs.edit().putString("runtime_context", buildString {
                appendLine("===== LAST RUNTIME CONTEXT =====")
                appendLine("Captured: ${DateFormat.getDateTimeInstance().format(Date())}")
                appendLine("Model: ${modelName ?: snapshot.model?.displayName ?: "N/A"}")
                appendLine("Quantization: ${snapshot.model?.quantization ?: "N/A"}")
                appendLine("Runtime: ${snapshot.runtime.name} ${snapshot.runtime.version}")
                appendLine("Backend: ${snapshot.runtime.backend ?: "N/A"}")
                appendLine("Threads: ${snapshot.runtime.threads ?: "N/A"}")
                appendLine("GPU Layers: ${snapshot.runtime.gpuLayers ?: "N/A"}")
                appendLine("Context: ${snapshot.runtime.contextLength ?: "N/A"}")
                appendLine("App status: $runtimeStatus")
                appendLine("Inference running: $inferenceRunning")
                appendLine("Last native event: ${snapshot.lastNativeEvent ?: "N/A"}")
                appendLine("GPU device: ${snapshot.gpuDevice?.name ?: "N/A"}")
                appendLine("GPU weight residency MiB: ${snapshot.weightResidency?.gpuTensorMiB ?: "N/A"}")
                appendLine("Host tensor memory MiB: ${snapshot.weightResidency?.hostTensorMiB ?: "N/A"}")
                appendLine("Context setting: ${snapshot.settings?.contextLength ?: "N/A"}")
            }).commit()
        }
    }

    private fun installJavaCrashHandler(context: Context) {
        val previous = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { thread, throwable ->
            runCatching { persistJavaCrash(context, thread, throwable) }
            if (previous != null) previous.uncaughtException(thread, throwable)
            else {
                Process.killProcess(Process.myPid())
                kotlin.system.exitProcess(10)
            }
        }
    }

    private fun persistJavaCrash(context: Context, thread: Thread, throwable: Throwable) {
        val stack = StringWriter().also { writer -> throwable.printStackTrace(PrintWriter(writer)) }.toString()
        val runtimeContext = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .getString("runtime_context", "Runtime context: unavailable") ?: "Runtime context: unavailable"
        val report = buildString {
            appendLine("AI Chat Test — Java Crash Report")
            appendLine("Detected: ${DateFormat.getDateTimeInstance().format(Date())}")
            appendLine("Crash type: Uncaught Java/Kotlin exception")
            appendLine("Thread: ${thread.name} (id=${thread.id})")
            appendLine("Exception: ${throwable.javaClass.name}")
            appendLine("Message: ${throwable.message ?: "N/A"}")
            appendLine()
            appendLine(explainJavaFailure(throwable))
            appendLine()
            appendLine(deviceInfo(context))
            appendLine(runtimeContext)
            appendLine("===== JAVA STACK TRACE =====")
            appendLine(stack)
        }
        appendReport(context, report)
        mutableReport.value = readReport(context)
    }

    private fun capturePlatformExitInfo(context: Context) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.R) return
        runCatching {
            val manager = context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager ?: return
            val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            val lastHandled = prefs.getLong(KEY_PLATFORM_EXIT_TIMESTAMP, 0L)
            val exit = manager.getHistoricalProcessExitReasons(context.packageName, 0, 12)
                .filter { it.timestamp > lastHandled }
                .filter { it.reason in setOf(ApplicationExitInfo.REASON_CRASH, ApplicationExitInfo.REASON_CRASH_NATIVE,
                    ApplicationExitInfo.REASON_ANR, ApplicationExitInfo.REASON_LOW_MEMORY,
                    ApplicationExitInfo.REASON_SIGNALED, ApplicationExitInfo.REASON_EXCESSIVE_RESOURCE_USAGE,
                    ApplicationExitInfo.REASON_INITIALIZATION_FAILURE) }
                .maxByOrNull { it.timestamp } ?: return
            val reason = when (exit.reason) {
                ApplicationExitInfo.REASON_CRASH -> "Java/Kotlin process crash"
                ApplicationExitInfo.REASON_CRASH_NATIVE -> "Native process crash"
                ApplicationExitInfo.REASON_ANR -> "Application Not Responding (ANR)"
                ApplicationExitInfo.REASON_LOW_MEMORY -> "Process terminated due to low memory"
                ApplicationExitInfo.REASON_SIGNALED -> "Process terminated by signal"
                ApplicationExitInfo.REASON_EXCESSIVE_RESOURCE_USAGE -> "Excessive resource usage"
                ApplicationExitInfo.REASON_INITIALIZATION_FAILURE -> "Process initialization failure"
                else -> "Abnormal process exit"
            }
            val platformTrace = runCatching { exit.traceInputStream?.bufferedReader()?.use { it.readText().takeLast(220_000) } }.getOrNull().orEmpty()
            val nativeTrace = readNativeTrace(context)
            val hasNativeFatal = nativeTrace.lastIndexOf("NATIVE_FATAL_SIGNAL=") >= 0
            val isNativeExit = exit.reason == ApplicationExitInfo.REASON_CRASH_NATIVE || exit.reason == ApplicationExitInfo.REASON_SIGNALED
            if (isNativeExit && hasNativeFatal) {
                prefs.edit()
                    .putString(KEY_PENDING_PLATFORM_EXIT, "$reason | code=${exit.reason} | status=${exit.status} | process=${exit.processName}")
                    .putString(KEY_PENDING_PLATFORM_TRACE, extractTombstoneEvidence(platformTrace))
                    .putLong(KEY_PLATFORM_EXIT_TIMESTAMP, exit.timestamp)
                    .commit()
                return@runCatching
            }
            val alreadyCapturedJava = exit.reason == ApplicationExitInfo.REASON_CRASH && readReport(context).orEmpty().contains("Java Crash Report")
            if (!alreadyCapturedJava) {
                val report = buildString {
                    appendLine("AI Chat Test — Android Process Exit")
                    appendLine("Time: ${DateFormat.getDateTimeInstance().format(Date(exit.timestamp))}")
                    appendLine("Reason: $reason (code=${exit.reason})")
                    appendLine("Status: ${exit.status}")
                    appendLine("Description: ${exit.description ?: "N/A"}")
                    appendLine()
                    appendLine(when (exit.reason) {
                        ApplicationExitInfo.REASON_ANR -> "Android ثبت کرده برنامه پاسخ‌گو نبوده است."
                        ApplicationExitInfo.REASON_LOW_MEMORY -> "Android ثبت کرده پردازش به علت فشار حافظه خاتمه یافته است."
                        else -> "Android خروج غیرعادی پردازش را ثبت کرده است."
                    })
                    appendLine()
                    appendLine(deviceInfo(context))
                    appendLine(context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getString("runtime_context", "Runtime context: unavailable"))
                    appendLine("===== ANDROID TRACE / TOMBSTONE =====")
                    appendLine(extractTombstoneEvidence(platformTrace).ifBlank { "Android برای این خروج traceInputStream ارائه نکرد." })
                }
                appendReport(context, report)
            }
            prefs.edit().putLong(KEY_PLATFORM_EXIT_TIMESTAMP, exit.timestamp).commit()
            mutableReport.value = readReport(context)
        }
    }

    private fun capturePendingNativeCrash(context: Context) {
        val trace = readNativeTrace(context)
        val signalIndex = trace.lastIndexOf("NATIVE_FATAL_SIGNAL=")
        if (signalIndex < 0) return
        val fatalTail = trace.substring(signalIndex).lineSequence()
            .takeWhile { it.startsWith("NATIVE_FATAL_") || it.isBlank() }
            .joinToString("\n").take(4_000)
        val fingerprint = sha256(fatalTail)
        val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        if (prefs.getString(KEY_NATIVE_FINGERPRINT, null) == fingerprint) return
        fun fatalValue(key: String): String = Regex("^${Regex.escape(key)}=(.*)$", RegexOption.MULTILINE)
            .find(fatalTail)?.groupValues?.get(1)?.trim().orEmpty().ifBlank { "UNKNOWN" }
        val signalNumber = fatalValue("NATIVE_FATAL_SIGNAL")
        val recordedSignalName = fatalValue("NATIVE_FATAL_SIGNAL_NAME")
        val signalName = if (recordedSignalName != "UNKNOWN") recordedSignalName else when (signalNumber) {
            "6" -> "SIGABRT"
            "11" -> "SIGSEGV"
            "7" -> "SIGBUS"
            "4" -> "SIGILL"
            "8" -> "SIGFPE"
            else -> "UNKNOWN"
        }
        val phase = fatalValue("NATIVE_FATAL_SPEC_PHASE")
        val runtimePhase = fatalValue("NATIVE_FATAL_RUNTIME_PHASE")
        val recentLines = trace.lineSequence().toList().takeLast(500)
        val lastCheckpoint = recentLines.lastOrNull { line ->
            line.contains("graph_reserve") || line.contains("sched_reserve") ||
                line.contains("llama_graph_n_input_tensors") || line.contains("CONTEXT_INIT") ||
                line.contains("WEBGPU_CONTEXT_INIT") || line.contains("MODEL_LOAD")
        }
        val platformExit = prefs.getString(KEY_PENDING_PLATFORM_EXIT, null)
        val platformTrace = prefs.getString(KEY_PENDING_PLATFORM_TRACE, null).orEmpty()
        val report = buildString {
            appendLine("AI Chat Test — Native Crash")
            appendLine("Detected: ${DateFormat.getDateTimeInstance().format(Date())}")
            appendLine("Signal: $signalName ($signalNumber)")
            appendLine("Signal code: ${fatalValue("NATIVE_FATAL_SI_CODE")}")
            appendLine("Speculative phase: $phase")
            appendLine("Last native phase: $runtimePhase")
            appendLine("Last graph checkpoint: ${lastCheckpoint ?: "N/A"}")
            appendLine("PC: ${fatalValue("NATIVE_FATAL_PC")}")
            appendLine("LR: ${fatalValue("NATIVE_FATAL_LR")}")
            val faultAddress = fatalValue("NATIVE_FATAL_FAULT_ADDR")
            if (faultAddress != "UNKNOWN") appendLine("Fault address: $faultAddress")
            if (platformExit != null) appendLine("Android exit: $platformExit")
            appendLine()
            appendLine(explainNativeFailure(signalName, phase, runtimePhase, lastCheckpoint, trace))
            appendLine()
            appendLine(deviceInfo(context))
            appendLine(context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getString("runtime_context", "Runtime context: unavailable"))
            appendLine("===== IMPORTANT NATIVE RESULTS =====")
            appendLine(summarizeNativeTrace(trace).ifBlank { "رویداد تشخیصی تکمیلی ثبت نشده است." })
            appendLine("===== ANDROID TOMBSTONE / BACKTRACE =====")
            appendLine(platformTrace.ifBlank { "Android در این اجرا backtrace قابل‌دسترسی ارائه نکرد. برای تعیین خط Native، tombstone و Build ID کتابخانه‌های همین APK لازم است." })
            appendLine("===== FATAL SIGNAL RECORD =====")
            appendLine(fatalTail)
        }
        appendReport(context, report)
        prefs.edit().putString(KEY_NATIVE_FINGERPRINT, fingerprint)
            .remove(KEY_PENDING_PLATFORM_EXIT).remove(KEY_PENDING_PLATFORM_TRACE).commit()
        mutableReport.value = readReport(context)
    }

    private fun summarizeNativeTrace(trace: String): String {
        val lines = trace.lineSequence().toList()
        val markers = listOf("WEBGPU_MODEL_LOAD_RETURNED_SUCCESS", "NATIVE_BACKEND_DEVICE ", "NATIVE_GPU_DEVICE ",
            "NATIVE_WEIGHT_RESIDENCY ", "WEBGPU_MODEL_RESIDENCY ", "WEBGPU_MODEL_RESIDENCY_POLICY ",
            "NATIVE_ATTENTION_POLICY ", "NATIVE_CONTEXT_THREADS ", "NATIVE_GRAPH_REUSE_POLICY ",
            "WEBGPU_CONTEXT_INIT_STARTED", "WEBGPU_MODEL_RESIDENCY_VALIDATED")
        val successful = lines.filter { line -> markers.any(line::contains) }.takeLast(12)
        val failures = lines.filter { line -> line.contains("FAILED", true) || line.contains("FATAL", true) ||
            line.contains("ERROR", true) || line.contains("GGML_STATUS_FAILED", true) ||
            line.contains("out of memory", true) || line.contains("assert", true) }.takeLast(12)
        return (successful + failures).distinct().takeLast(20).joinToString("\n")
    }

    private fun extractTombstoneEvidence(trace: String): String {
        if (trace.isBlank()) return ""
        val lines = trace.lineSequence().toList()
        val relevant = lines.filter { line -> line.contains("signal ", true) || line.contains("Abort message:", true) ||
            line.contains("backtrace:", true) || Regex("^\\s*#\\d+").containsMatchIn(line) ||
            line.contains("BuildId", true) || line.contains("Build ID", true) ||
            line.contains("libai_chat_runtime", true) || line.contains("libggml", true) ||
            line.contains("libllama", true) || line.contains("libc.so", true) ||
            line.contains("Cause:", true) || line.contains("pid:", true) || line.contains("tid:", true) }
        val selected = if (relevant.isNotEmpty()) relevant.takeLast(100).joinToString("\n")
            else lines.takeLast(100).joinToString("\n")
        return selected.takeLast(18_000)
    }

    private fun explainJavaFailure(error: Throwable): String {
        val root = generateSequence(error) { it.cause }.last()
        return buildString {
            appendLine("تشخیص:")
            appendLine("این مورد یک استثنای Java/Kotlin ثبت‌نشده است؛ بنابراین کلاس استثنا و Stack Trace شواهد مستقیم خطا هستند.")
            appendLine("ریشه زنجیره استثنا: ${root.javaClass.name}: ${root.message ?: "بدون پیام"}")
            appendLine("علت دقیق باید از اولین فریم Stack Trace متعلق به کد برنامه و زنجیره cause بررسی شود؛ صرف نام استثنا برای تعیین قطعی علت کافی نیست.")
        }
    }

    private fun explainNativeFailure(signal: String, phase: String, runtimePhase: String, checkpoint: String?, trace: String): String {
        val signalMeaning = when (signal) {
            "SIGABRT" -> "پردازش Native با abort متوقف شده است؛ معمولاً assert داخلی، خطای غیرقابل‌بازیابی کتابخانه یا فراخوانی صریح abort می‌تواند آن را ایجاد کند."
            "SIGSEGV" -> "دسترسی نامعتبر به حافظه رخ داده است (segmentation fault)."
            "SIGBUS" -> "خطای دسترسی حافظه/نگاشت فایل یا هم‌ترازی رخ داده است."
            "SIGILL" -> "دستورالعمل پردازنده نامعتبر یا پشتیبانی‌نشده اجرا شده است."
            "SIGFPE" -> "خطای محاسباتی سطح Native رخ داده است."
            else -> "سیگنال Fatal در لایه Native ثبت شده، اما نوع آن در گزارش قابل تشخیص نیست."
        }
        val clue = when {
            runtimePhase.contains("sched_reserve", ignoreCase = true) || runtimePhase.contains("graph_reserve", ignoreCase = true) ->
                "آخرین رویداد Native ثبت‌شده مربوط به رزرو Scheduler/ساخت گراف است: $runtimePhase. این محل توقف را مشخص می‌کند، نه علت ریشه‌ای قطعی را."
            runtimePhase.contains("FLASH_ATTN_EXT", ignoreCase = true) ->
                "آخرین رویداد Native ثبت‌شده به FLASH_ATTN_EXT اشاره می‌کند: $runtimePhase. این قرینهٔ مستقیم مرحلهٔ توقف است، اما به‌تنهایی علت ریشه‌ای را اثبات نمی‌کند."
            trace.contains("FLASH_ATTN_EXT") -> "آخرین گراف ثبت‌شده شامل FLASH_ATTN_EXT بوده است. این فقط قرینه‌ای از مرحله اجرای گراف است و به‌تنهایی اثبات نمی‌کند همان عملیات علت Crash بوده است."
            trace.contains("sched_reserve") || trace.contains("graph_reserve") -> "ردپا وارد مرحله رزرو Scheduler/ساخت گراف شده است. این موضوع محل تقریبی توقف را نشان می‌دهد، نه علت ریشه‌ای قطعی را."
            trace.contains("WEBGPU_MODEL_LOAD") || trace.contains("MODEL_LOAD_STARTED_WEBGPU") -> "ردپا در مسیر بارگذاری مدل با اولویت WebGPU قرار داشته است."
            trace.contains("CPU_FALLBACK_STARTED") -> "ردپا در مسیر بارگذاری یا آماده‌سازی CPU fallback قرار داشته است."
            else -> "از روی آخرین checkpoint فقط می‌توان مرحله تقریبی اجرای برنامه را محدود کرد."
        }
        return buildString {
            appendLine("تشخیص فنی:")
            appendLine(signalMeaning)
            appendLine("نشانگر فاز speculative: ${if (phase == "UNKNOWN") "نامشخص" else phase}")
            appendLine("آخرین رویداد Native: ${if (runtimePhase == "UNKNOWN") "نامشخص" else runtimePhase}")
            appendLine("آخرین checkpoint: ${checkpoint ?: "ثبت نشده"}")
            appendLine(clue)
            appendLine("محدودیت: برای تعیین علت قطعی باید backtrace و Build ID همین APK با نمادهای همان نسخه تطبیق داده شوند.")
        }
    }

    private fun deviceInfo(context: Context): String = buildString {
        appendLine("===== DEVICE / APP =====")
        appendLine("Manufacturer: ${Build.MANUFACTURER}")
        appendLine("Device: ${Build.DEVICE}")
        appendLine("Model: ${Build.MODEL}")
        appendLine("Android: ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})")
        appendLine("ABI: ${Build.SUPPORTED_ABIS.joinToString()}")
        val packageInfo = runCatching { context.packageManager.getPackageInfo(context.packageName, 0) }.getOrNull()
        appendLine("App: ${context.packageName}")
        appendLine("Version: ${packageInfo?.versionName ?: "N/A"} (${packageInfo?.longVersionCode ?: packageInfo?.versionCode ?: "N/A"})")
    }

    private fun readNativeTrace(context: Context): String {
        val tempDir = System.getProperty("java.io.tmpdir")?.takeIf { it.isNotBlank() }
        val candidates = listOfNotNull(
            tempDir?.let { File(it, "ai-chat-native-trace.txt") },
            File(context.cacheDir, "ai-chat-native-trace.txt"),
            File(context.filesDir, "ai-chat-native-trace.txt"),
        ).distinctBy { it.absolutePath }
        return candidates.firstOrNull { it.isFile }?.let { file ->
            runCatching {
                java.io.RandomAccessFile(file, "r").use { input ->
                    val start = (input.length() - 256_000L).coerceAtLeast(0L)
                    input.seek(start)
                    if (start > 0L) input.readLine()
                    val remaining = (input.length() - input.filePointer).coerceAtLeast(0L).coerceAtMost(256_000L).toInt()
                    val bytes = ByteArray(remaining)
                    input.readFully(bytes)
                    String(bytes, Charsets.UTF_8)
                }
            }.getOrNull()
        }.orEmpty()
    }

    private fun appendReport(context: Context, report: String) {
        val file = File(context.filesDir, FILE_NAME)
        file.parentFile?.mkdirs()
        val old = if (file.isFile) file.readText().takeLast(12_000) else ""
        val updated = (old + "\n\n================ CRASH ================\n" + report.takeLast(24_000)).takeLast(MAX_REPORT_CHARS)
        file.writeText(updated)
    }

    private fun readReport(context: Context): String? = runCatching {
        val file = File(context.filesDir, FILE_NAME)
        if (!file.isFile) return@runCatching null
        val original = file.readText()
        val compact = compactLegacyReport(original).takeLast(MAX_REPORT_CHARS)
        if (compact != original) file.writeText(compact)
        compact.takeIf { it.isNotBlank() }
    }.getOrNull()

    private fun compactLegacyReport(value: String): String {
        var compact = value
        compact = compact.replace(
            Regex("(?ms)^===== NATIVE TRACE BEFORE CRASH.*?(?=^={8,}.*CRASH|\\\\z)"),
            "===== NATIVE TRACE =====\\n(ردپای تکراری و طولانی حذف شد؛ خلاصهٔ رویدادهای مهم در گزارش نگه داشته شده است.)\\n"
        )
        compact = compact.replace(
            Regex("(?ms)^===== NATIVE TRACE TAIL =====.*?(?=^=====|^={8,}.*CRASH|\\\\z)"),
            "===== NATIVE TRACE =====\\n(ردپای خام حذف شد.)\\n"
        )
        compact = compact.replace(
            Regex("(?ms)^===== RECENT RUNTIME EVENTS =====.*?(?=^=====|^={8,}.*CRASH|\\\\z)"),
            "===== RUNTIME EVENTS =====\\n(رویدادهای تکراری حذف شدند.)\\n"
        )
        return compact
    }

    private fun sha256(value: String): String =
        MessageDigest.getInstance("SHA-256").digest(value.toByteArray()).joinToString("") { "%02x".format(Locale.US, it) }
}
