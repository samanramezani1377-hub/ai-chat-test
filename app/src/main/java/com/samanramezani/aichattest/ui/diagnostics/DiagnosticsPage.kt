package com.samanramezani.aichattest.ui.diagnostics

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.samanramezani.aichattest.ui.state.ExecutionState
import java.text.DateFormat
import java.util.Date

@Composable
internal fun DiagnosticsPage(
    diagnostic: RuntimeDiagnostic?,
    error: String?,
    execution: ExecutionState?,
) {
    val context = LocalContext.current
    val report = diagnostic?.report() ?: fallbackReport(error, execution)
    val errorReport = diagnostic?.errorReport() ?: fallbackErrorReport(error, execution)

    SimplePage("عیب‌یابی") {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = { copyToClipboard(context, "گزارش عیب‌یابی", report) }, Modifier.weight(1f)) { Text("کپی گزارش کامل") }
            OutlinedButton(onClick = { copyToClipboard(context, "گزارش خطا", errorReport) }, Modifier.weight(1f)) { Text("کپی خطا") }
        }

        Text("عملکرد Runtime", style = MaterialTheme.typography.titleLarge)
        if (diagnostic == null) {
            Text("هنوز گزارش Runtime ثبت نشده است.", color = MaterialTheme.colorScheme.onSurfaceVariant)
        } else {
            MetricRow("زمان بارگذاری مدل", diagnostic.loadTimeMs?.let { "$it ms" } ?: "N/A")
            MetricRow("زمان تا اولین توکن (TTFT)", diagnostic.generation?.firstTokenTimeMs?.let { "$it ms" } ?: "N/A")
            MetricRow("زمان تولید", diagnostic.generation?.generationTimeMs?.let { "$it ms" } ?: "N/A")
            MetricRow("توکن ورودی", diagnostic.generation?.inputTokens?.toString() ?: "N/A")
            MetricRow("توکن خروجی", diagnostic.generation?.outputTokens?.toString() ?: "N/A")
            MetricRow("سرعت", diagnostic.generation?.let { speed(it) } ?: "N/A")
        }

        Text("Runtime", style = MaterialTheme.typography.titleLarge)
        diagnostic?.let {
            MetricRow("مدل", it.model?.displayName ?: "N/A")
            MetricRow("فرمت", it.model?.format?.toString() ?: "N/A")
            MetricRow("Quantization", it.model?.quantization ?: "N/A")
            MetricRow("Runtime", it.runtime.name)
            MetricRow("نسخه", it.runtime.version)
            MetricRow("Backend", it.runtime.backend ?: "N/A")
            MetricRow("Threads", it.runtime.threads?.toString() ?: "N/A")
            MetricRow("GPU Layers", it.runtime.gpuLayers?.toString() ?: "N/A")
            MetricRow("Context", it.runtime.contextLength?.toString() ?: "N/A")
        }

        Text("پارامترهای Generation", style = MaterialTheme.typography.titleLarge)
        diagnostic?.settings?.let { settings ->
            MetricRow("Temperature", settings.temperature.toString())
            MetricRow("Top-P", settings.topP?.toString() ?: "N/A")
            MetricRow("Top-K", settings.topK?.toString() ?: "N/A")
            MetricRow("Min-P", settings.minP?.toString() ?: "N/A")
            MetricRow("Repeat Penalty", settings.repeatPenalty?.toString() ?: "N/A")
            MetricRow("Max New Tokens", settings.maxNewTokens.toString())
            MetricRow("Context Length", settings.contextLength?.toString() ?: "N/A")
            MetricRow("Seed", settings.seed?.toString() ?: "N/A")
            MetricRow("Stop Sequences", settings.stopSequences.size.toString())
        }

        Text("Execution", style = MaterialTheme.typography.titleLarge)
        if (execution == null) {
            Text("Execution ثبت‌شده‌ای وجود ندارد.", color = MaterialTheme.colorScheme.onSurfaceVariant)
        } else {
            MetricRow("Execution", execution.id)
            MetricRow("Action", execution.action)
            MetricRow("وضعیت", execution.status)
            MetricRow("شروع", DateFormat.getDateTimeInstance().format(Date(execution.startedAt)))
            MetricRow("پایان", execution.finishedAt?.let { DateFormat.getDateTimeInstance().format(Date(it)) } ?: "N/A")
            MetricRow("Approval", if (execution.approvalRequired) "لازم است" else "لازم نیست")
            execution.error?.let { Text("Error: $it", color = MaterialTheme.colorScheme.error) }
        }

        Text("Trace", style = MaterialTheme.typography.titleLarge)
        if (diagnostic?.trace.isNullOrEmpty()) {
            Text("Trace برای این Execution ثبت نشده است.", color = MaterialTheme.colorScheme.onSurfaceVariant)
        } else {
            diagnostic!!.trace.forEach { event ->
                UiSurface {
                    Column(Modifier.padding(10.dp)) {
                        Text(event.type.toString(), style = MaterialTheme.typography.labelLarge)
                        Text(DateFormat.getDateTimeInstance().format(Date(event.timestampMs)), style = MaterialTheme.typography.bodySmall)
                        event.message?.let { Text(it) }
                    }
                }
            }
        }

        Text("گزارش خام قابل کپی", style = MaterialTheme.typography.titleLarge)
        Surface(Modifier.fillMaxWidth(), color = MaterialTheme.colorScheme.surfaceVariant, shape = MaterialTheme.shapes.medium) {
            Text(report, Modifier.padding(12.dp), style = MaterialTheme.typography.bodySmall)
        }
    }
}

@Composable
private fun MetricRow(label: String, value: String) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        Text(label, Modifier.weight(1f), color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text(value, Modifier.weight(1f))
    }
}

private fun speed(result: com.woogit.aicore.domain.GenerationResult): String {
    val tokens = result.outputTokens ?: return "N/A"
    val ms = result.generationTimeMs ?: return "N/A"
    if (tokens <= 0 || ms <= 0) return "N/A"
    return "%.2f tok/s".format(tokens.toDouble() / (ms / 1000.0))
}

private fun fallbackReport(error: String?, execution: ExecutionState?): String = buildString {
    appendLine("AI Chat Test — Runtime Diagnostic Report")
    appendLine("Status: ${execution?.status ?: "N/A"}")
    appendLine("Execution: ${execution?.id ?: "N/A"}")
    appendLine("Error: ${error ?: execution?.error ?: "N/A"}")
}

private fun fallbackErrorReport(error: String?, execution: ExecutionState?): String = buildString {
    appendLine("AI Chat Test — Error Report")
    appendLine("Execution: ${execution?.id ?: "N/A"}")
    appendLine("Error: ${error ?: execution?.error ?: "N/A"}")
}

private fun copyToClipboard(context: Context, label: String, text: String) {
    val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
    clipboard.setPrimaryClip(ClipData.newPlainText(label, text))
}

@Composable
private fun SimplePage(title: String, content: @Composable ColumnScope.() -> Unit) {
    LazyColumn(Modifier.fillMaxSize().padding(20.dp), contentPadding = PaddingValues(bottom = 32.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        item { Text(title, style = MaterialTheme.typography.headlineMedium) }
        item { Column(verticalArrangement = Arrangement.spacedBy(10.dp), content = content) }
    }
}
