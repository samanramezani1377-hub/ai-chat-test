package com.samanramezani.aichattest.ui.diagnostics

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.ErrorOutline
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.Speed
import androidx.compose.material.icons.filled.Terminal
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.samanramezani.aichattest.ui.components.SimplePage
import com.samanramezani.aichattest.ui.state.ExecutionState
import com.woogit.aicore.actions.ActionTraceStore
import com.woogit.aicore.runtime.RuntimeDiagnosticsStore
import java.text.DateFormat
import java.util.Date

@Composable
internal fun DiagnosticsPage(error: String?, execution: ExecutionState?) {
    val context = LocalContext.current
    val runtime by RuntimeDiagnosticsStore.snapshot.collectAsState()
    val actionTraces by ActionTraceStore.events.collectAsState()
    LaunchedEffect(Unit) { RuntimeDiagnosticsStore.refreshNativeEvent() }

    val selectedActions = actionTraces.filter { execution == null || it.executionId == execution.id }
    val diagnostic = RuntimeDiagnostic(
        model = runtime.model,
        runtime = runtime.runtime,
        loadTimeMs = runtime.loadTimeMs,
        generation = runtime.generation,
        settings = runtime.settings,
        status = execution?.status ?: if (runtime.generation != null) "SUCCESS" else "READY",
        error = error ?: execution?.error,
        rawError = execution?.error,
        executionId = execution?.id,
        actionTrace = selectedActions,
        runtimeTrace = runtime.trace,
        nativeDiagnostics = runtime.lastNativeEvent,
    )
    val report = buildString {
        append(diagnostic.report())
        append("\n\n===== LAST NATIVE EVENT =====\n")
        append(runtime.lastNativeEvent ?: "N/A")
    }
    val errorReport = buildString {
        append(diagnostic.errorReport())
        append("\n\n===== LAST NATIVE EVENT =====\n")
        append(runtime.lastNativeEvent ?: "N/A")
    }
    val hasError = !diagnostic.error.isNullOrBlank()
    val generation = runtime.generation
    val nativePerf = diagnostic.nativePerformance
    val speed = generation?.let {
        val ms = it.generationTimeMs
        val tokens = it.outputTokens
        if (ms != null && ms > 0 && tokens != null) "%.1f tok/s".format(tokens.toDouble() * 1000.0 / ms) else "—"
    } ?: "—"

    SimplePage("عیب‌یابی") {
        StatusCard(
            hasError = hasError,
            status = diagnostic.status,
            model = runtime.model?.displayName ?: "مدلی فعال نیست",
            onCopyError = { copyToClipboard(context, "گزارش خطا", errorReport) },
            onCopyFull = { copyToClipboard(context, "گزارش کامل", report) },
        )

        Text("خلاصه اجرای اخیر", style = MaterialTheme.typography.titleLarge)
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            MetricCard("بارگذاری", runtime.loadTimeMs?.let { "$it ms" } ?: "—", Modifier.weight(1f), Icons.Default.Memory)
            MetricCard("اولین توکن", generation?.firstTokenTimeMs?.let { "$it ms" } ?: "—", Modifier.weight(1f), Icons.Default.Speed)
            MetricCard("سرعت", speed, Modifier.weight(1f), Icons.Default.Speed)
        }

        SectionCard("وضعیت Runtime") {
            InfoLine("مدل", runtime.model?.displayName ?: "—")
            InfoLine("فرمت", runtime.model?.format?.toString() ?: "—")
            InfoLine("Quantization", runtime.model?.quantization?.toString() ?: "—")
            InfoLine("Runtime", runtime.runtime.name)
            InfoLine("نسخه", runtime.runtime.version)
            InfoLine("Backend", runtime.runtime.backend ?: "OpenCL")
            InfoLine("GPU Layers", runtime.runtime.gpuLayers?.toString() ?: "—")
            InfoLine("Threads", runtime.runtime.threads?.toString() ?: "—")
            InfoLine("Context", runtime.runtime.contextLength?.toString() ?: "—")
        }

        SectionCard("ورودی و خروجی") {
            InfoLine("Prompt tokens", nativePerf.promptTokens?.toString() ?: "—")
            InfoLine("Output tokens", generation?.outputTokens?.toString() ?: nativePerf.generatedTokens?.toString() ?: "—")
            InfoLine("Generation time", generation?.generationTimeMs?.let { "$it ms" } ?: "—")
            InfoLine("Prefill", nativePerf.prefillMs?.let { "$it ms" } ?: "—")
            InfoLine("Decode", nativePerf.decodeMs?.let { "$it ms" } ?: "—")
            InfoLine("KV reused", nativePerf.reusedTokens?.toString() ?: "0")
        }

        if (hasError) {
            SectionCard("خطای ثبت‌شده", MaterialTheme.colorScheme.errorContainer) {
                Text(diagnostic.error ?: "خطای نامشخص", color = MaterialTheme.colorScheme.onErrorContainer)
                Spacer(Modifier.height(4.dp))
                Text("گزارش بالا متن کامل قابل ارسال است؛ این بخش خودش علت را حدس نمی‌زند.", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onErrorContainer)
            }
        }

        Text("مراحل آخر", style = MaterialTheme.typography.titleLarge)
        val events = runtime.trace.takeLast(10).reversed()
        if (events.isEmpty()) {
            EmptyCard("هنوز رویداد Runtime ثبت نشده است.")
        } else {
            events.forEachIndexed { index, event ->
                TimelineItem(index + 1, event.type.toString(), event.message, event.timestampMs)
            }
        }

        if (selectedActions.isNotEmpty()) {
            Text("مراحل Agent", style = MaterialTheme.typography.titleLarge)
            selectedActions.takeLast(8).reversed().forEachIndexed { index, event ->
                TimelineItem(index + 1, event.type.toString(), event.message, event.timestampMs)
            }
        }

        SectionCard("آخرین رویداد Native") {
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                Icon(Icons.Default.Terminal, null, tint = MaterialTheme.colorScheme.primary)
                Text(runtime.lastNativeEvent ?: "رویدادی در دسترس نیست.", style = MaterialTheme.typography.bodySmall, modifier = Modifier.weight(1f))
            }
        }

        Text("برای بررسی فنی، «گزارش کامل» را کپی کن. اطلاعات خام عملکرد نگه داشته می‌شود و تشخیص گلوگاه به بررسی گزارش سپرده می‌شود.", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@Composable
private fun StatusCard(
    hasError: Boolean,
    status: String,
    model: String,
    onCopyError: () -> Unit,
    onCopyFull: () -> Unit,
) {
    Surface(
        Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(24.dp),
        color = if (hasError) MaterialTheme.colorScheme.errorContainer else MaterialTheme.colorScheme.primaryContainer,
    ) {
        Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Icon(
                    if (hasError) Icons.Default.ErrorOutline else Icons.Default.CheckCircle,
                    null,
                    tint = if (hasError) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.primary,
                    modifier = Modifier.size(32.dp),
                )
                Column(Modifier.weight(1f)) {
                    Text(if (hasError) "این اجرا خطا داشته" else "Runtime آماده است", style = MaterialTheme.typography.titleLarge)
                    Text("وضعیت: $status", style = MaterialTheme.typography.labelLarge)
                    Text("مدل: $model", style = MaterialTheme.typography.bodySmall)
                }
            }
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                if (hasError) {
                    Button(onClick = onCopyError, Modifier.weight(1f).height(48.dp)) {
                        Icon(Icons.Default.ContentCopy, null)
                        Spacer(Modifier.width(6.dp))
                        Text("کپی خطا")
                    }
                }
                OutlinedButton(onClick = onCopyFull, Modifier.weight(1f).height(48.dp)) {
                    Text("کپی گزارش کامل")
                }
            }
        }
    }
}

@Composable
private fun MetricCard(label: String, value: String, modifier: Modifier, icon: androidx.compose.ui.graphics.vector.ImageVector) {
    Surface(modifier, shape = RoundedCornerShape(18.dp), color = MaterialTheme.colorScheme.surfaceVariant) {
        Column(Modifier.padding(13.dp), verticalArrangement = Arrangement.spacedBy(5.dp)) {
            Icon(icon, null, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(19.dp))
            Text(label, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            Text(value, style = MaterialTheme.typography.titleMedium)
        }
    }
}

@Composable
private fun SectionCard(
    title: String,
    color: androidx.compose.ui.graphics.Color = MaterialTheme.colorScheme.surface,
    content: @Composable ColumnScope.() -> Unit,
) {
    Surface(Modifier.fillMaxWidth(), shape = RoundedCornerShape(20.dp), color = color, tonalElevation = 1.dp) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(9.dp)) {
            Text(title, style = MaterialTheme.typography.titleMedium)
            content()
        }
    }
}

@Composable
private fun InfoLine(label: String, value: String) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        Text(label, Modifier.weight(1f), color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text(value, Modifier.weight(1f), style = MaterialTheme.typography.bodyMedium)
    }
}

@Composable
private fun EmptyCard(text: String) {
    Surface(Modifier.fillMaxWidth(), shape = RoundedCornerShape(18.dp), color = MaterialTheme.colorScheme.surfaceVariant) {
        Text(text, Modifier.padding(16.dp), color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@Composable
private fun TimelineItem(number: Int, title: String, message: String?, timestampMs: Long) {
    Surface(Modifier.fillMaxWidth(), shape = RoundedCornerShape(16.dp), color = MaterialTheme.colorScheme.surfaceVariant) {
        Row(Modifier.padding(12.dp), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            Surface(shape = RoundedCornerShape(10.dp), color = MaterialTheme.colorScheme.primaryContainer) {
                Text(number.toString(), Modifier.padding(horizontal = 9.dp, vertical = 6.dp), color = MaterialTheme.colorScheme.primary, style = MaterialTheme.typography.labelLarge)
            }
            Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(3.dp)) {
                Text(title, style = MaterialTheme.typography.labelLarge)
                Text(DateFormat.getDateTimeInstance().format(Date(timestampMs)), style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                message?.takeIf { it.isNotBlank() }?.let { Text(it, style = MaterialTheme.typography.bodySmall) }
            }
        }
    }
}

private fun copyToClipboard(context: Context, label: String, text: String) {
    val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as? ClipboardManager ?: return
    clipboard.setPrimaryClip(ClipData.newPlainText(label, text))
}
