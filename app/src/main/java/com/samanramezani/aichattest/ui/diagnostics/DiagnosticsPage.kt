package com.samanramezani.aichattest.ui.diagnostics

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
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
    val selectedActionTraces = actionTraces.filter { execution == null || it.executionId == execution.id }
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
        actionTrace = selectedActionTraces,
        runtimeTrace = runtime.trace,
    )
    val report = diagnostic.report()
    val errorReport = diagnostic.errorReport()

    SimplePage("عیب‌یابی") {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = { copyToClipboard(context, "گزارش عیب‌یابی", report) }, modifier = Modifier.weight(1f)) { Text("کپی گزارش کامل") }
            OutlinedButton(onClick = { copyToClipboard(context, "گزارش خطا", errorReport) }, modifier = Modifier.weight(1f)) { Text("کپی خطا") }
        }

        Text("عملکرد Runtime", style = MaterialTheme.typography.titleLarge)
        MetricRow("زمان بارگذاری مدل", runtime.loadTimeMs?.let { "$it ms" } ?: "N/A")
        MetricRow("زمان تا اولین توکن (TTFT)", runtime.generation?.firstTokenTimeMs?.let { "$it ms" } ?: "N/A")
        MetricRow("زمان تولید", runtime.generation?.generationTimeMs?.let { "$it ms" } ?: "N/A")
        MetricRow("توکن ورودی", runtime.generation?.inputTokens?.toString() ?: "N/A")
        MetricRow("توکن خروجی", runtime.generation?.outputTokens?.toString() ?: "N/A")
        MetricRow("سرعت", runtime.generation?.let { generation ->
            val seconds = generation.generationTimeMs?.toDouble()?.div(1000.0)
            if (seconds == null || seconds <= 0.0) "N/A" else "%.2f tok/s".format(generation.outputTokens / seconds)
        } ?: "N/A")

        Text("Runtime", style = MaterialTheme.typography.titleLarge)
        MetricRow("مدل", runtime.model?.displayName ?: "N/A")
        MetricRow("فرمت", runtime.model?.format?.toString() ?: "N/A")
        MetricRow("Quantization", runtime.model?.quantization?.toString() ?: "N/A")
        MetricRow("Runtime", runtime.runtime.name)
        MetricRow("نسخه", runtime.runtime.version)
        MetricRow("Backend", runtime.runtime.backend ?: "N/A")
        MetricRow("Threads", runtime.runtime.threads?.toString() ?: "N/A")
        MetricRow("GPU Layers", runtime.runtime.gpuLayers?.toString() ?: "N/A")
        MetricRow("Context", runtime.runtime.contextLength?.toString() ?: "N/A")

        Text("پارامترهای Generation", style = MaterialTheme.typography.titleLarge)
        runtime.settings?.let { settings ->
            MetricRow("Temperature", settings.temperature.toString())
            MetricRow("Top-P", settings.topP?.toString() ?: "N/A")
            MetricRow("Top-K", settings.topK?.toString() ?: "N/A")
            MetricRow("Min-P", settings.minP?.toString() ?: "N/A")
            MetricRow("Repeat Penalty", settings.repeatPenalty?.toString() ?: "N/A")
            MetricRow("Max New Tokens", settings.maxNewTokens.toString())
            MetricRow("Context Length", settings.contextLength?.toString() ?: "N/A")
            MetricRow("Seed", settings.seed?.toString() ?: "N/A")
            MetricRow("Stop Sequences", settings.stopSequences.size.toString())
        } ?: Text("برای آخرین Generation تنظیماتی ثبت نشده است.", color = MaterialTheme.colorScheme.onSurfaceVariant)

        Text("Execution", style = MaterialTheme.typography.titleLarge)
        if (execution == null) Text("Execution ثبت‌شده‌ای وجود ندارد.", color = MaterialTheme.colorScheme.onSurfaceVariant)
        else {
            MetricRow("Execution", execution.id)
            MetricRow("Action", execution.action)
            MetricRow("وضعیت", execution.status)
            MetricRow("شروع", DateFormat.getDateTimeInstance().format(Date(execution.startedAt)))
            MetricRow("پایان", execution.finishedAt?.let { DateFormat.getDateTimeInstance().format(Date(it)) } ?: "N/A")
            MetricRow("Approval", if (execution.approvalRequired) "لازم است" else "لازم نیست")
            execution.error?.let { Text("Error: $it", color = MaterialTheme.colorScheme.error) }
        }

        Text("Runtime Trace", style = MaterialTheme.typography.titleLarge)
        if (runtime.trace.isEmpty()) Text("Trace Runtime ثبت نشده است.", color = MaterialTheme.colorScheme.onSurfaceVariant)
        else runtime.trace.forEach { event ->
            Surface(Modifier.fillMaxWidth(), shape = MaterialTheme.shapes.medium, tonalElevation = 1.dp) {
                Column(Modifier.padding(10.dp)) {
                    Text(event.type.toString(), style = MaterialTheme.typography.labelLarge)
                    Text(DateFormat.getDateTimeInstance().format(Date(event.timestampMs)), style = MaterialTheme.typography.bodySmall)
                    event.message?.let { Text(it) }
                }
            }
        }

        Text("Action Trace", style = MaterialTheme.typography.titleLarge)
        if (selectedActionTraces.isEmpty()) Text("Trace عملیات برای این Execution ثبت نشده است.", color = MaterialTheme.colorScheme.onSurfaceVariant)
        else selectedActionTraces.forEach { event ->
            Surface(Modifier.fillMaxWidth(), shape = MaterialTheme.shapes.medium, tonalElevation = 1.dp) {
                Column(Modifier.padding(10.dp)) {
                    Text(event.type.toString(), style = MaterialTheme.typography.labelLarge)
                    Text(DateFormat.getDateTimeInstance().format(Date(event.timestampMs)), style = MaterialTheme.typography.bodySmall)
                    event.message?.let { Text(it) }
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

private fun copyToClipboard(context: Context, label: String, text: String) {
    val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as? ClipboardManager ?: return
    clipboard.setPrimaryClip(ClipData.newPlainText(label, text))
}
