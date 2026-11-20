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
    private const val FILE_NAME = "crash-log.txt"
    private const val MAX_REPORT_CHARS = 480_000
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
            prefs.edit()
                .putString("runtime_context", buildString {
                    appendLine("===== LAST RUNTIME CONTEXT =====")
                    appendLine("Captured: ${DateFormat.getDateTimeInstance().format(Date())}")
                    appendLine("Model: ${modelName ?: snapshot.model?.displayName ?: "N/A"}")
                    appendLine("Quantization: ${snapshot.model?.quantization ?: "N/A"}")
                    appendLine("Runtime: ${snapshot.runtime.name} ${snapshot.runtime.version}")
                    appendLine("Backend: ${snapshot.runtime.backend ?: "N/A"}")
                    appendLine("Threads: ${snapshot.runtime.threads ?: "N/A"}")
                    appendLine("GPU Layers: ${snapshot.runtime.gpuLayers ?: "N/A"}")
                    appendLine("Context: ${snapshot.runtime.contextLength ?: "N/A"}")
                    appendLine("App runtime status: $runtimeStatus")
                    appendLine("Inference running: $inferenceRunning")
                    appendLine("Last native event: ${snapshot.lastNativeEvent ?: "N/A"}")
                    appendLine("GPU device: ${snapshot.gpuDevice?.name ?: "N/A"}")
                    appendLine("GPU residency MiB: ${snapshot.weightResidency?.gpuTensorMiB ?: "N/A"}")
                    appendLine("Host tensor MiB: ${snapshot.weightResidency?.hostTensorMiB ?: "N/A"}")
                    appendLine("Temperature: ${snapshot.settings?.temperature ?: "N/A"}")
                    appendLine("Top-P: ${snapshot.settings?.topP ?: "N/A"}")
                    appendLine("Top-K: ${snapshot.settings?.topK ?: "N/A"}")
                    appendLine("Min-P: ${snapshot.settings?.minP ?: "N/A"}")
                    appendLine("Repeat penalty: ${snapshot.settings?.repeatPenalty ?: "N/A"}")
                    appendLine("Max new tokens: ${snapshot.settings?.maxNewTokens ?: "N/A"}")
                    appendLine("Context setting: ${snapshot.settings?.contextLength ?: "N/A"}")
                    appendLine("===== RECENT RUNTIME EVENTS =====")
                    snapshot.trace.takeLast(30).forEach { event ->
                        appendLine("${event.timestampMs} | ${event.type} | ${event.message ?: ""}")
                    }
                })
                .commit()
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
            appendLine("===== NATIVE TRACE TAIL =====")
            appendLine(readNativeTrace(context).takeLast(120_000).ifBlank { "No native trace available." })
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
                .filter { it.reason in setOf(
                    ApplicationExitInfo.REASON_CRASH,
                    ApplicationExitInfo.REASON_CRASH_NATIVE,
                    ApplicationExitInfo.REASON_ANR,
                    ApplicationExitInfo.REASON_LOW_MEMORY,
                    ApplicationExitInfo.REASON_SIGNALED,
                    ApplicationExitInfo.REASON_EXCESSIVE_RESOURCE_USAGE,
                    ApplicationExitInfo.REASON_INITIALIZATION_FAILURE,
                ) }
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
            val platformTrace = runCatching {
                exit.traceInputStream?.bufferedReader()?.use { it.readText().takeLast(220_000) }
            }.getOrNull().orEmpty()
            val existing = readReport(context).orEmpty()
            val alreadyCapturedJava = exit.reason == ApplicationExitInfo.REASON_CRASH &&
                existing.contains("Java Crash Report")
            if (!alreadyCapturedJava) {
                val report = buildString {
                    appendLine("AI Chat Test — Android Process Exit Report")
                    appendLine("Exit timestamp: ${DateFormat.getDateTimeInstance().format(Date(exit.timestamp))}")
                    appendLine("Process: ${exit.processName}")
                    appendLine("Reason: $reason (code=${exit.reason})")
                    appendLine("Status: ${exit.status}")
                    appendLine("Importance: ${exit.importance}")
                    appendLine("Description: ${exit.description ?: "N/A"}")
                    appendLine()
                    appendLine(when (exit.reason) {
                        ApplicationExitInfo.REASON_ANR -> "تشخیص: سیستم ثبت کرده که برنامه در مهلت پاسخ‌گویی متوقف مانده است. Trace زیر برای یافتن Thread مسدودشده و Stack آن اهمیت دارد."
                        ApplicationExitInfo.REASON_LOW_MEMORY -> "تشخیص: سیستم پردازش را به دلیل فشار حافظه متوقف کرده است. این یک علت سیستمی ثبت‌شده است؛ برای تعیین تخصیص حافظه مشکل‌ساز، گزارش حافظه و لاگ پیش از خروج لازم است."
                        ApplicationExitInfo.REASON_CRASH_NATIVE -> "تشخیص: Android خروج غیرعادی در لایه Native را ثبت کرده است. اگر traceInputStream موجود باشد، tombstone و backtrace واقعی در ادامه آمده است."
                        ApplicationExitInfo.REASON_CRASH -> "تشخیص: Android خروج ناشی از Crash در پردازش را ثبت کرده است. جزئیات استثنا و Stack Trace در گزارش Java Crash قبلی قرار دارد، اگر ثبت شده باشد."
                        else -> "تشخیص: Android خروج غیرعادی پردازش را ثبت کرده است. کد reason و توضیحات سیستم شواهد قطعی در دسترس هستند."
                    })
                    appendLine()
                    appendLine(deviceInfo(context))
                    appendLine(context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getString("runtime_context", "Runtime context: unavailable"))
                    appendLine("===== ANDROID EXIT TRACE / TOMBSTONE =====")
                    appendLine(platformTrace.ifBlank { "Android did not provide a traceInputStream for this exit." })
                    appendLine("===== NATIVE TRACE TAIL =====")
                    appendLine(readNativeTrace(context).takeLast(100_000).ifBlank { "No native trace available." })
                }
                appendReport(context, report)
            }
            val nativeTrace = readNativeTrace(context)
            val fatalIndex = nativeTrace.lastIndexOf("NATIVE_FATAL_SIGNAL=")
            if (exit.reason == ApplicationExitInfo.REASON_CRASH_NATIVE && fatalIndex >= 0) {
                val fatalLines = nativeTrace.substring(fatalIndex).lineSequence()
                    .takeWhile { it.startsWith("NATIVE_FATAL_") || it.isBlank() }
                    .joinToString("\\n")
                prefs.edit().putString(KEY_NATIVE_FINGERPRINT, sha256(fatalLines)).commit()
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
            .joinToString("\\n")
            .take(12_000)
        val fingerprint = sha256(fatalTail)
        val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        if (prefs.getString(KEY_NATIVE_FINGERPRINT, null) == fingerprint) return

        val signalName = Regex("""NATIVE_FATAL_SIGNAL_NAME=([^\\s]+)""").find(fatalTail)?.groupValues?.get(1) ?: "UNKNOWN"
        val phase = Regex("""NATIVE_FATAL_SPEC_PHASE=([^\\s]+)""").find(fatalTail)?.groupValues?.get(1) ?: "UNKNOWN"
        val lastCheckpoint = trace.lineSequence().filter { it.contains("NATIVE_CHECKPOINT") || it.contains("MODEL_READY") || it.contains("CONTEXT_INIT") || it.contains("MODEL_LOAD") }.lastOrNull()
        val report = buildString {
            appendLine("AI Chat Test — Native Crash Report")
            appendLine("Detected on app restart: ${DateFormat.getDateTimeInstance().format(Date())}")
            appendLine("Crash type: Native fatal signal")
            appendLine("Signal: $signalName (${Regex("""NATIVE_FATAL_SIGNAL=([^\\s]+)""").find(fatalTail)?.groupValues?.get(1) ?: "N/A"})")
            appendLine("Native phase marker: $phase")
            appendLine("Last relevant checkpoint: ${lastCheckpoint ?: "N/A"}")
            appendLine()
            appendLine(explainNativeFailure(signalName, phase, lastCheckpoint, trace))
            appendLine()
            appendLine(deviceInfo(context))
            appendLine(context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getString("runtime_context", "Runtime context: unavailable"))
            appendLine("===== FATAL SIGNAL RECORD =====")
            appendLine(fatalTail)
            appendLine("===== NATIVE TRACE BEFORE CRASH (last 350 lines) =====")
            appendLine(trace.lineSequence().toList().takeLast(350).joinToString("\\n"))
        }
        appendReport(context, report)
        prefs.edit().putString(KEY_NATIVE_FINGERPRINT, fingerprint).commit()
        mutableReport.value = readReport(context)
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

    private fun explainNativeFailure(signal: String, phase: String, checkpoint: String?, trace: String): String {
        val signalMeaning = when (signal) {
            "SIGABRT" -> "پردازش Native با abort متوقف شده است؛ معمولاً assert داخلی، خطای غیرقابل‌بازیابی کتابخانه یا فراخوانی صریح abort می‌تواند آن را ایجاد کند."
            "SIGSEGV" -> "دسترسی نامعتبر به حافظه رخ داده است (segmentation fault)."
            "SIGBUS" -> "خطای دسترسی حافظه/نگاشت فایل یا هم‌ترازی رخ داده است."
            "SIGILL" -> "دستورالعمل پردازنده نامعتبر یا پشتیبانی‌نشده اجرا شده است."
            "SIGFPE" -> "خطای محاسباتی سطح Native رخ داده است."
            else -> "سیگنال Fatal در لایه Native ثبت شده، اما نوع آن در گزارش قابل تشخیص نیست."
        }
        val clue = when {
            trace.contains("FLASH_ATTN_EXT") -> "آخرین گراف ثبت‌شده شامل FLASH_ATTN_EXT بوده است. این فقط قرینه‌ای از مرحله اجرای گراف است و به‌تنهایی اثبات نمی‌کند همان عملیات علت Crash بوده است."
            trace.contains("sched_reserve") || trace.contains("graph_reserve") -> "ردپا وارد مرحله رزرو Scheduler/ساخت گراف شده است. این موضوع محل تقریبی توقف را نشان می‌دهد، نه علت ریشه‌ای قطعی را."
            trace.contains("WEBGPU_MODEL_LOAD") || trace.contains("MODEL_LOAD_STARTED_WEBGPU") -> "ردپا در مسیر بارگذاری مدل با اولویت WebGPU قرار داشته است."
            trace.contains("CPU_FALLBACK_STARTED") -> "ردپا در مسیر بارگذاری یا آماده‌سازی CPU fallback قرار داشته است."
            else -> "از روی آخرین checkpoint فقط می‌توان مرحله تقریبی اجرای برنامه را محدود کرد."
        }
        return buildString {
            appendLine("تشخیص فنی:")
            appendLine(signalMeaning)
            appendLine("مرحله ثبت‌شده: ${if (phase == "UNKNOWN") "نامشخص" else phase}")
            appendLine("آخرین checkpoint: ${checkpoint ?: "ثبت نشده"}")
            appendLine(clue)
            appendLine("محدودیت تشخیص: سیگنال و PC/LR به‌تنهایی علت کامل را مشخص نمی‌کنند. برای تعیین خط دقیق Native به tombstone/backtrace و Build ID همان APK نیاز است. این گزارش شواهد خام را نگه می‌دارد و از تبدیل حدس به علت قطعی خودداری می‌کند.")
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
        return candidates.firstOrNull { it.isFile }?.let { runCatching { it.readText() }.getOrNull() }.orEmpty()
    }

    private fun appendReport(context: Context, report: String) {
        val file = File(context.filesDir, FILE_NAME)
        file.parentFile?.mkdirs()
        val old = if (file.isFile) file.readText() else ""
        val updated = (old + "\\n\\n================ NEW CRASH ================\\n" + report).takeLast(MAX_REPORT_CHARS)
        file.writeText(updated)
    }

    private fun readReport(context: Context): String? =
        runCatching { File(context.filesDir, FILE_NAME).takeIf { it.isFile }?.readText()?.takeIf { it.isNotBlank() } }.getOrNull()

    private fun sha256(value: String): String =
        MessageDigest.getInstance("SHA-256").digest(value.toByteArray()).joinToString("") { "%02x".format(Locale.US, it) }
}
