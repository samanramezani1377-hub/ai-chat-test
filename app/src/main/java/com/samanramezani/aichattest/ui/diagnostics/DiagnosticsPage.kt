package com.samanramezani.aichattest.ui.diagnostics

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.ErrorOutline
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Speed
import androidx.compose.material.icons.filled.Terminal
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import com.samanramezani.aichattest.ui.components.SimplePage
import com.samanramezani.aichattest.ui.errors.ErrorCenter
import com.samanramezani.aichattest.ui.state.ExecutionState
import com.woogit.aicore.actions.ActionTraceStore
import com.woogit.aicore.runtime.RuntimeDiagnosticsStore
import java.text.DateFormat
import java.util.Date
import java.util.Locale

private enum class DiagnosticsSection(val title: String) {
    SUMMARY("نمای کلی"), PERFORMANCE("عملکرد"), LOGS("لاگ‌ها"), CONFIG("مشخصات")
}

@Composable
internal fun DiagnosticsPage(error: String?, execution: ExecutionState?) {
    val context = LocalContext.current
    val runtime by RuntimeDiagnosticsStore.snapshot.collectAsState()
    val actionTraces by ActionTraceStore.events.collectAsState()
    var selectedSection by remember { mutableStateOf(DiagnosticsSection.SUMMARY) }
    var logQuery by remember { mutableStateOf("") }

    LaunchedEffect(Unit) { RuntimeDiagnosticsStore.refreshNativeEvent() }

    val selectedActions = actionTraces.filter { execution == null || it.executionId == execution.id }
    val diagnostic = RuntimeDiagnostic(
        model = runtime.model,
        runtime = runtime.runtime,
        loadTimeMs = runtime.loadTimeMs,
        generation = runtime.generation,
        settings = runtime.settings,
        status = execution?.status ?: if (!error.isNullOrBlank()) "FAILED" else if (runtime.generation != null) "SUCCESS" else "READY",
        error = error ?: execution?.error,
        rawError = execution?.error,
        executionId = execution?.id,
        actionTrace = selectedActions,
        runtimeTrace = runtime.trace,
        nativeDiagnostics = runtime.lastNativeEvent,
        openClProfile = runtime.openClProfile,
        gpuDevice = runtime.gpuDevice,
        weightResidency = runtime.weightResidency,
    )
    val hasError = !diagnostic.error.isNullOrBlank() || diagnostic.status.equals("FAILED", true) || diagnostic.status.equals("ERROR", true)
    val resolvedError = if (hasError) ErrorCenter.resolve(
        listOf(diagnostic.error, runtime.lastNativeEvent).filterNotNull().joinToString("\n")
    ) else null
    val generation = runtime.generation
    val perf = diagnostic.nativePerformance
    // Successful load: concise lifecycle sample. Failed step: retain its full logs.
    val nativeLines = remember(runtime.lastNativeEvent, hasError, diagnostic.status) {
        diagnostic.logLines()
    }
    val filteredLines = remember(nativeLines, logQuery) {
        if (logQuery.isBlank()) nativeLines else nativeLines.filter { it.contains(logQuery, ignoreCase = true) }
    }
    val speed = generation?.let {
        val ms = it.generationTimeMs
        val tokens = it.outputTokens
        if (ms != null && ms > 0 && tokens != null) String.format(Locale.US, "%.1f tok/s", tokens.toDouble() * 1000.0 / ms) else "—"
    } ?: "—"

    SimplePage("عیب‌یابی") {
        Surface(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(24.dp),
            color = if (hasError) MaterialTheme.colorScheme.errorContainer else MaterialTheme.colorScheme.surface,
            tonalElevation = 2.dp,
        ) {
            Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    Icon(
                        if (hasError) Icons.Default.ErrorOutline else Icons.Default.CheckCircle,
                        contentDescription = null,
                        tint = if (hasError) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.primary,
                        modifier = Modifier.size(32.dp),
                    )
                    Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(3.dp)) {
                        Text(if (hasError) "خطا در اجرای اخیر" else "وضعیت اجرای اخیر", style = MaterialTheme.typography.titleLarge)
                        Text(statusLabel(diagnostic.status), style = MaterialTheme.typography.bodyMedium)
                        Text(runtime.model?.displayName ?: "مدلی ثبت نشده است", style = MaterialTheme.typography.bodySmall)
                    }
                }
                if (hasError && resolvedError != null) {
                    HorizontalDivider()
                    Text("${resolvedError.code} · ${resolvedError.title}", style = MaterialTheme.typography.titleMedium)
                    Text(resolvedError.message, style = MaterialTheme.typography.bodyMedium)
                    Text("پیشنهاد: ${resolvedError.action}", style = MaterialTheme.typography.bodySmall)
                }
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    FilledTonalButton(
                        onClick = { copyToClipboard(context, "گزارش خلاصه عیب‌یابی", if (hasError) diagnostic.errorReport() else diagnostic.report()) },
                        modifier = Modifier.weight(1f),
                    ) {
                        Icon(Icons.Default.ContentCopy, null)
                        Spacer(Modifier.width(6.dp))
                        Text(if (hasError) "کپی گزارش + لاگ کامل خطا" else "کپی خلاصهٔ گزارش")
                    }
                    OutlinedButton(
                        onClick = { copyToClipboard(context, if (hasError) "لاگ کامل مرحله ناموفق" else "نمونه لاگ بارگذاری", nativeLines.joinToString("\n")) },
                        modifier = Modifier.weight(1f),
                    ) {
                        Icon(Icons.Default.Terminal, null)
                        Spacer(Modifier.width(6.dp))
                        Text(if (hasError) "کپی لاگ کامل مرحله" else "کپی نمونه لاگ بارگذاری")
                    }
                }
            }
        }

        TabRow(selectedTabIndex = DiagnosticsSection.values().indexOf(selectedSection)) {
            DiagnosticsSection.values().forEachIndexed { index, section ->
                Tab(
                    selected = selectedSection == section,
                    onClick = { selectedSection = section },
                    text = { Text(section.title, maxLines = 1) },
                )
            }
        }

        when (selectedSection) {
            DiagnosticsSection.SUMMARY -> {
                SectionCard("خلاصه") {
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        MetricCard("بارگذاری", runtime.loadTimeMs?.let { "$it ms" } ?: "—", Modifier.weight(1f), Icons.Default.Memory)
                        MetricCard("اولین توکن", generation?.firstTokenTimeMs?.let { "$it ms" } ?: "—", Modifier.weight(1f), Icons.Default.Speed)
                        MetricCard("سرعت", speed, Modifier.weight(1f), Icons.Default.Speed)
                    }
                    InfoLine("مدل", runtime.model?.displayName ?: "—")
                    InfoLine("وضعیت", diagnostic.status)
                    InfoLine("Backend", runtime.runtime.backend ?: "گزارش نشده")
                    InfoLine("GPU", runtime.gpuDevice?.name ?: "گزارش نشده")
                    InfoLine("شناسه اجرا", execution?.id ?: "—")
                }
                SectionCard("حافظه و محل قرارگیری وزن‌ها") {
                    val device = runtime.gpuDevice
                    val residency = runtime.weightResidency
                    InfoLine("حافظه کل GPU", device?.memoryTotalMiB?.let { "%.1f MiB".format(Locale.US, it) } ?: "گزارش نشده")
                    InfoLine("حافظه آزاد GPU", device?.memoryFreeMiB?.let { "%.1f MiB".format(Locale.US, it) } ?: "گزارش نشده")
                    InfoLine("وزن‌های GPU", residency?.gpuTensorMiB?.let { "%.1f MiB".format(Locale.US, it) } ?: "—")
                    InfoLine("تعداد Tensorهای GPU", residency?.gpuTensors?.toString() ?: "—")
                    InfoLine("وزن‌های Host", residency?.hostTensorMiB?.let { "%.1f MiB".format(Locale.US, it) } ?: "—")
                    InfoLine("وزن‌های CPU", residency?.cpuTensorMiB?.let { "%.1f MiB".format(Locale.US, it) } ?: "—")
                    InfoLine("وزن‌های دیگر", residency?.otherTensorMiB?.let { "%.1f MiB".format(Locale.US, it) } ?: "—")
                }
                SectionCard("رویدادهای اخیر Runtime") {
                    val events = runtime.trace.takeLast(12).reversed()
                    if (events.isEmpty()) EmptyCard("هنوز رویدادی ثبت نشده است.")
                    else events.forEach { event ->
                        EventRow(localizedLogTitle(event.type.toString()), event.message, event.timestampMs)
                    }
                }
                if (selectedActions.isNotEmpty()) {
                    SectionCard("رویدادهای Agent") {
                        selectedActions.takeLast(12).reversed().forEach { event ->
                            EventRow(localizedLogTitle(event.type.toString()), event.message, event.timestampMs)
                        }
                    }
                }
            }

            DiagnosticsSection.PERFORMANCE -> {
                SectionCard("زمان و سرعت تولید") {
                    InfoLine("زمان بارگذاری مدل", runtime.loadTimeMs?.let { "$it ms" } ?: "—")
                    InfoLine("زمان اولین توکن (TTFT)", generation?.firstTokenTimeMs?.let { "$it ms" } ?: "—")
                    InfoLine("زمان تولید", perf.generationMs?.let { "$it ms" } ?: generation?.generationTimeMs?.let { "$it ms" } ?: "—")
                    InfoLine("زمان Prefill", perf.prefillMs?.let { "$it ms" } ?: "—")
                    InfoLine("زمان Decode", perf.decodeMs?.let { "$it ms" } ?: "—")
                    InfoLine("سرعت Decode", perf.decodeTokensPerSec?.let { String.format(Locale.US, "%.2f tok/s", it) } ?: "—")
                    InfoLine("توکن‌های ورودی", perf.promptTokens?.toString() ?: generation?.inputTokens?.toString() ?: "—")
                    InfoLine("توکن‌های خروجی", perf.generatedTokens?.toString() ?: generation?.outputTokens?.toString() ?: "—")
                }
                SectionCard("KV Cache") {
                    InfoLine("وضعیت", perf.cacheStatus ?: "گزارش نشده")
                    InfoLine("توکن‌های ذخیره‌شده", perf.cachedTokens?.toString() ?: "—")
                    InfoLine("توکن‌های استفاده‌شده مجدد", perf.reusedTokens?.toString() ?: "—")
                    InfoLine("توکن‌های جدید", perf.newTokens?.toString() ?: "—")
                    InfoLine("نسبت Cache Hit", perf.cacheHitRatio?.let { String.format(Locale.US, "%.1f%%", it) } ?: "—")
                }
                SectionCard("پروفایل GPU / OpenCL") {
                    val gpu = runtime.openClProfile
                    if (gpu == null) {
                        EmptyCard("دادهٔ پروفایل OpenCL موجود نیست؛ ممکن است Runtime آن را ثبت نکرده باشد.")
                    } else {
                        InfoLine("تعداد Kernel", gpu.kernelCount.toString())
                        InfoLine("اجرای Kernelها", gpu.totalKernelMs?.let { "%.3f ms".format(Locale.US, it) } ?: "—")
                        InfoLine("MUL_MAT Q6_K", gpu.q6KMulMatMs?.let { "%.3f ms".format(Locale.US, it) } ?: "—")
                        InfoLine("Attention", gpu.attentionMs?.let { "%.3f ms".format(Locale.US, it) } ?: "—")
                        InfoLine("RoPE", gpu.ropeMs?.let { "%.3f ms".format(Locale.US, it) } ?: "—")
                        InfoLine("RMSNorm", gpu.rmsNormMs?.let { "%.3f ms".format(Locale.US, it) } ?: "—")
                        InfoLine("FFN", gpu.ffnMs?.let { "%.3f ms".format(Locale.US, it) } ?: "—")
                        InfoLine("Softmax", gpu.softmaxMs?.let { "%.3f ms".format(Locale.US, it) } ?: "—")
                        InfoLine("ارسال Kernel", gpu.kernelSubmitMs?.let { "%.3f ms".format(Locale.US, it) } ?: "—")
                        InfoLine("همگام‌سازی", gpu.syncMs?.let { "%.3f ms".format(Locale.US, it) } ?: "—")
                        InfoLine("انتقال حافظه", gpu.memoryTransferMs?.let { "%.3f ms".format(Locale.US, it) } ?: "اندازه‌گیری نشده")
                        gpu.topKernels.forEachIndexed { i, kernel ->
                            InfoLine("Kernel برتر ${i + 1}", "${kernel.kernelName} · %.3f ms".format(Locale.US, kernel.executionMs))
                        }
                    }
                }
            }

            DiagnosticsSection.LOGS -> {
                SectionCard("لاگ خام Native") {
                    Text(if (hasError) "مرحلهٔ ناموفق است؛ لاگ‌های کامل همان مرحله نمایش داده می‌شوند." else "بارگذاری موفق است؛ خلاصه و حداکثر ۵۰ خط نمونه از لاگ بارگذاری نمایش داده می‌شود.", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    OutlinedTextField(
                        value = logQuery,
                        onValueChange = { logQuery = it },
                        modifier = Modifier.fillMaxWidth(),
                        singleLine = true,
                        label = { Text("جست‌وجو در لاگ") },
                        placeholder = { Text("مثلاً CONTEXT_INIT_FAILED یا OpenGL") },
                        trailingIcon = {
                            IconButton(onClick = { logQuery = "" }) {
                                Icon(Icons.Default.Refresh, contentDescription = "پاک‌کردن جست‌وجو")
                            }
                        },
                    )
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                        Text(if (hasError) "${filteredLines.size} خط کامل از مرحلهٔ ناموفق" else "${filteredLines.size} خط نمونه (حداکثر ۵۰)", style = MaterialTheme.typography.labelMedium)
                        TextButton(onClick = { copyToClipboard(context, "لاگ فیلترشده", filteredLines.joinToString("\n")) }) {
                            Icon(Icons.Default.ContentCopy, null)
                            Spacer(Modifier.width(4.dp))
                            Text("کپی خطوط نمایش‌داده‌شده")
                        }
                    }
                    if (filteredLines.isEmpty()) {
                        EmptyCard(if (nativeLines.isEmpty()) "لاگ خامی در دسترس نیست." else "خطی با این عبارت پیدا نشد.")
                    } else {
                        Column(
                            Modifier.fillMaxWidth()
                                .heightIn(max = 560.dp)
                                .verticalScroll(rememberScrollState())
                                .background(MaterialTheme.colorScheme.surfaceVariant, RoundedCornerShape(12.dp))
                                .padding(10.dp),
                            verticalArrangement = Arrangement.spacedBy(5.dp),
                        ) {
                            filteredLines.forEachIndexed { index, line ->
                                Text(
                                    text = line,
                                    style = MaterialTheme.typography.bodySmall.copy(fontFamily = FontFamily.Monospace),
                                    color = if (line.contains("error", true) || line.contains("failed", true) || line.contains("exception", true))
                                        MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurfaceVariant,
                                )
                                if (index < filteredLines.lastIndex) HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.35f))
                            }
                        }
                    }
                }
                SectionCard("رویدادهای مهم") {
                    val events = diagnostic.runtimeTrace.takeLast(30).reversed()
                    if (events.isEmpty()) EmptyCard("رویداد ساختاریافته‌ای ثبت نشده است.")
                    else events.forEach { EventRow(localizedLogTitle(it.type.toString()), it.message, it.timestampMs) }
                }
            }

            DiagnosticsSection.CONFIG -> {
                SectionCard("مشخصات Runtime") {
                    InfoLine("مدل", runtime.model?.displayName ?: "—")
                    InfoLine("فرمت", runtime.model?.format?.toString() ?: "—")
                    InfoLine("Quantization", runtime.model?.quantization?.toString() ?: "—")
                    InfoLine("Runtime", runtime.runtime.name)
                    InfoLine("نسخه Runtime", runtime.runtime.version)
                    InfoLine("Backend", runtime.runtime.backend ?: "—")
                    InfoLine("GPU Layers", runtime.runtime.gpuLayers?.toString() ?: "—")
                    InfoLine("Threads", runtime.runtime.threads?.toString() ?: "—")
                    InfoLine("Context", runtime.runtime.contextLength?.toString() ?: "—")
                }
                SectionCard("تنظیمات تولید") {
                    val s = runtime.settings
                    InfoLine("Temperature", s?.temperature?.toString() ?: "—")
                    InfoLine("Top-P", s?.topP?.toString() ?: "—")
                    InfoLine("Top-K", s?.topK?.toString() ?: "—")
                    InfoLine("Min-P", s?.minP?.toString() ?: "—")
                    InfoLine("Repeat Penalty", s?.repeatPenalty?.toString() ?: "—")
                    InfoLine("حداکثر توکن خروجی", s?.maxNewTokens?.toString() ?: "—")
                    InfoLine("Context تنظیم‌شده", s?.contextLength?.toString() ?: "—")
                    InfoLine("تعداد Stop Sequence", s?.stopSequences?.size?.toString() ?: "—")
                    InfoLine("Seed", s?.seed?.toString() ?: "—")
                }
                SectionCard("جزئیات Tensorهای غیر-GPU") {
                    val lines = diagnostic.nonGpuTensorLines
                    if (lines.isEmpty()) EmptyCard("Tensor غیر-GPU در لاگ ذخیره‌شده گزارش نشده است.")
                    else lines.forEach { Text(it, style = MaterialTheme.typography.bodySmall.copy(fontFamily = FontFamily.Monospace)) }
                    diagnostic.residencySummaryLines.distinct().forEach { Text(it, style = MaterialTheme.typography.bodySmall.copy(fontFamily = FontFamily.Monospace)) }
                }
                SectionCard("آخرین رویداد Native") {
                    Text(runtime.lastNativeEvent?.lineSequence()?.toList()?.takeLast(8)?.joinToString("\n") ?: "رویدادی در دسترس نیست.", style = MaterialTheme.typography.bodySmall.copy(fontFamily = FontFamily.Monospace))
                }
            }
        }
    }
}

@Composable
private fun SectionCard(title: String, content: @Composable ColumnScope.() -> Unit) {
    Surface(
        Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(20.dp),
        color = MaterialTheme.colorScheme.surface,
        tonalElevation = 1.dp,
    ) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text(title, style = MaterialTheme.typography.titleMedium)
            content()
        }
    }
}

@Composable
private fun MetricCard(label: String, value: String, modifier: Modifier, icon: androidx.compose.ui.graphics.vector.ImageVector) {
    Surface(modifier, shape = RoundedCornerShape(16.dp), color = MaterialTheme.colorScheme.surfaceVariant) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(5.dp)) {
            Icon(icon, null, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(19.dp))
            Text(label, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            Text(value, style = MaterialTheme.typography.titleMedium)
        }
    }
}

@Composable
private fun InfoLine(label: String, value: String) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        Text(label, Modifier.weight(1f), color = MaterialTheme.colorScheme.onSurfaceVariant, style = MaterialTheme.typography.bodySmall)
        Text(value, Modifier.weight(1.15f), style = MaterialTheme.typography.bodyMedium)
    }
}

@Composable
private fun EmptyCard(text: String) {
    Surface(Modifier.fillMaxWidth(), shape = RoundedCornerShape(14.dp), color = MaterialTheme.colorScheme.surfaceVariant) {
        Text(text, Modifier.padding(14.dp), color = MaterialTheme.colorScheme.onSurfaceVariant, style = MaterialTheme.typography.bodySmall)
    }
}

@Composable
private fun EventRow(title: String, message: String?, timestampMs: Long) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
        Surface(shape = RoundedCornerShape(8.dp), color = MaterialTheme.colorScheme.primaryContainer) {
            Text("•", Modifier.padding(horizontal = 9.dp, vertical = 4.dp), color = MaterialTheme.colorScheme.primary)
        }
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(3.dp)) {
            Text(title, style = MaterialTheme.typography.labelLarge)
            Text(DateFormat.getDateTimeInstance().format(Date(timestampMs)), style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            message?.takeIf { it.isNotBlank() }?.let { Text(it, style = MaterialTheme.typography.bodySmall) }
        }
    }
}

private fun statusLabel(status: String): String = when (status.uppercase(Locale.ROOT)) {
    "FAILED" -> "ناموفق"
    "SUCCESS" -> "موفق"
    "READY" -> "آماده"
    "RUNNING" -> "در حال اجرا"
    "STOPPED" -> "متوقف‌شده"
    else -> status
}

private fun localizedLogTitle(raw: String): String = when {
    raw.contains("GENERATION_STARTED") -> "شروع تولید پاسخ"
    raw.contains("GENERATION_STOPPED") -> "توقف تولید پاسخ"
    raw.contains("MODEL_LOAD") -> "بارگذاری مدل"
    raw.contains("MODEL") && raw.contains("FAILED") -> "خطای مدل"
    raw.contains("KV_CACHE") -> "وضعیت حافظه KV"
    raw.contains("TOKEN") -> "دریافت توکن"
    raw.contains("ACTION") -> "رویداد عامل"
    raw.contains("ERROR") || raw.contains("FAILED") -> "خطا"
    raw.contains("READY") -> "آماده"
    else -> raw.replace("_", " ").lowercase(Locale.ROOT).replaceFirstChar { it.titlecase(Locale.ROOT) }
}

private fun copyToClipboard(context: Context, label: String, text: String) {
    val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as? ClipboardManager ?: return
    clipboard.setPrimaryClip(ClipData.newPlainText(label, text))
}
