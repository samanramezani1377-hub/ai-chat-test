package com.samanramezani1377.aichattest.ui.diagnostics

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.ErrorOutline
import androidx.compose.material.icons.filled.Speed
import androidx.compose.material.icons.filled.Terminal
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.platform.LocalContext
import com.samanramezani.aichattest.ui.components.SimplePage
import com.samanramezani.aichattest.ui.state.ExecutionState
import com.woogit.aicore.actions.ActionTraceStore
import com.woogit.aicore.runtime.RuntimeDiagnosticsStore
import java.text.DateFormat
import java.util.Date

private const val DISPLAY_RUNTIME_EVENTS = 12
private const val DISPLAY_ACTION_EVENTS = 8

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
        status = execution?.status ?: if (runtime.generation != null) "SUCCESS" else "N/A",
        error = error ?: execution?.error,
        rawError = execution?.error,
        executionId = execution?.id,
        actionTrace = selectedActions,
        runtimeTrace = runtime.trace,
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
    val recentRuntime = runtime.trace.takeLast(DISPLAY_RUNTIME_EVENTS)
    val recentActions = selectedActions.takeLast(DISPLAY_ACTION_EVENTS)
    val hasError = !diagnostic.error.isNullOrBlank()
    val speed = runtime.generation?.let {
        val seconds = it.generationTimeMs?.toDouble()?.div(1000.0)
        if (seconds != null && seconds > 0 && it.outputTokens != null) "%.1f".format(it.outputTokens / seconds) else "—"
    } ?: "—"

    SimplePage("عیب‌یابی") {
        Surface(
            Modifier.fillMaxWidth(),
            shape = MaterialTheme.shapes.large,
            color = if (hasError) MaterialTheme.colorScheme.errorContainer else MaterialTheme.colorScheme.secondaryContainer,
        ) {
            Row(Modifier.padding(18.dp), horizontalArrangement = Arrangement.spacedBy(14.dp)) {
                Icon(if (hasError) Icons.Default.ErrorOutline else Icons.Default.CheckCircle, null, modifier = Modifier.size(30.dp))
                Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text(if (hasError) "یک اجرای ناموفق ثبت شده" else "همه‌چیز عادی به نظر می‌رسد", style = MaterialTheme.typography.titleLarge)
                    Text(
                        if (hasError) "گزارش خطا را کپی کن تا بتوانیم دقیقاً مسیر شکست را بررسی کنیم."
                        else "آخرین Runtime بدون خطای ثبت‌شده تمام شده است.",
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
            }
        }

        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = { copyToClipboard(context, "گزارش خطا", errorReport) }, modifier = Modifier.weight(1f).height(50.dp)) {
                Icon(Icons.Default.ContentCopy, null); Spacer(Modifier.width(7.dp)); Text("کپی خطا")
            }
            OutlinedButton(onClick = { copyToClipboard(context, "گزارش کامل", report) }, modifier = Modifier.weight(1f).height(50.dp)) {
                Text("گزارش کامل")
            }
        }

        Text("آنچه اخیراً اتفاق افتاد", style = MaterialTheme.typography.titleLarge)
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            MetricCard("TTFT", runtime.generation?.firstTokenTimeMs?.let { "$it ms" } ?: "—", Modifier.weight(1f))
            MetricCard("سرعت", if (speed == "—") "—" else "$speed tok/s", Modifier.weight(1f))
            MetricCard("خروجی", runtime.generation?.outputTokens?.toString() ?: "—", Modifier.weight(1f))
        }

        Surface(Modifier.fillMaxWidth(), shape = MaterialTheme.shapes.medium, color = MaterialTheme.colorScheme.surface) {
            Column(Modifier.padding(15.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                InfoLine("وضعیت", diagnostic.status)
                InfoLine("مدل", diagnostic.model?.displayName ?: "—")
                InfoLine("Runtime", diagnostic.runtime.name + " " + diagnostic.runtime.version)
                InfoLine("Backend", diagnostic.runtime.backend ?: "OpenCL")
                InfoLine("GPU Layers", diagnostic.runtime.gpuLayers?.toString() ?: "—")
                if (hasError) Text(diagnostic.error ?: "", color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
            }
        }

        if (runtime.lastNativeEvent != null) {
            Text("آخرین رویداد Native", style = MaterialTheme.typography.titleMedium)
            Surface(Modifier.fillMaxWidth(), shape = MaterialTheme.shapes.medium, color = MaterialTheme.colorScheme.surfaceVariant) {
                Row(Modifier.padding(13.dp), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    Icon(Icons.Default.Terminal, null, tint = MaterialTheme.colorScheme.secondary)
                    Text(runtime.lastNativeEvent ?: "", style = MaterialTheme.typography.bodySmall, modifier = Modifier.weight(1f))
                }
            }
        }

        Text("Timeline", style = MaterialTheme.typography.titleMedium)
        if (recentRuntime.isEmpty()) {
            Text("رویدادی ثبت نشده است.", color = MaterialTheme.colorScheme.onSurfaceVariant)
        } else {
            recentRuntime.reversed().forEach { event ->
                Surface(Modifier.fillMaxWidth(), shape = MaterialTheme.shapes.small, color = MaterialTheme.colorScheme.surfaceVariant) {
                    Column(Modifier.padding(11.dp)) {
                        Text(event.type.toString(), style = MaterialTheme.typography.labelLarge)
                        Text(DateFormat.getDateTimeInstance().format(Date(event.timestampMs)), style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        event.message?.let { Text(it, style = MaterialTheme.typography.bodySmall) }
                    }
                }
            }
        }

        if (recentActions.isNotEmpty()) {
            Text("Action timeline", style = MaterialTheme.typography.titleMedium)
            recentActions.reversed().forEach { event ->
                Surface(Modifier.fillMaxWidth(), shape = MaterialTheme.shapes.small, color = MaterialTheme.colorScheme.surfaceVariant) {
                    Column(Modifier.padding(11.dp)) {
                        Text(event.type.toString(), style = MaterialTheme.typography.labelLarge)
                        Text(DateFormat.getDateTimeInstance().format(Date(event.timestampMs)), style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        event.message?.let { Text(it, style = MaterialTheme.typography.bodySmall) }
                    }
                }
            }
        }

        Text("مشخصات فنی", style = MaterialTheme.typography.titleMedium)
        Surface(Modifier.fillMaxWidth(), shape = MaterialTheme.shapes.medium, color = MaterialTheme.colorScheme.surface) {
            Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                InfoLine("فرمت", runtime.model?.format?.toString() ?: "—")
                InfoLine("Quantization", runtime.model?.quantization?.toString() ?: "—")
                InfoLine("Threads", runtime.runtime.threads?.toString() ?: "—")
                InfoLine("Context", runtime.runtime.contextLength?.toString() ?: "—")
                InfoLine("Load time", runtime.loadTimeMs?.let { "$it ms" } ?: "—")
                InfoLine("Generation", runtime.generation?.generationTimeMs?.let { "$it ms" } ?: "—")
            }
        }
    }
}

@Composable private fun MetricCard(label: String, value: String, modifier: Modifier) {
    Surface(modifier, shape = MaterialTheme.shapes.medium, color = MaterialTheme.colorScheme.surfaceVariant) {
        Column(Modifier.padding(13.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(label, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            Text(value, style = MaterialTheme.typography.titleMedium)
        }
    }
}

@Composable private fun InfoLine(label: String, value: String) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        Text(label, Modifier.weight(1f), color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text(value, Modifier.weight(1f), style = MaterialTheme.typography.bodyMedium)
    }
}

private fun copyToClipboard(context: Context, label: String, text: String) {
    val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as? ClipboardManager ?: return
    clipboard.setPrimaryClip(ClipData.newPlainText(label, text))
}
