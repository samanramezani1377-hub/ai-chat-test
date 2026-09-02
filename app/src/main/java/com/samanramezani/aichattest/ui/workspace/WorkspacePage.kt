package com.samanramezani.aichattest.ui.workspace

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import com.samanramezani.aichattest.ui.state.ExecutionState
import com.woogit.aicore.actions.ActionErrorLog
import com.woogit.aicore.actions.ActionTraceEvent
import com.woogit.aicore.domain.ModelDescriptor

@Composable
internal fun WorkspacePage(
    execution: ExecutionState?,
    model: ModelDescriptor?,
    traces: List<ActionTraceEvent> = emptyList(),
    errors: List<ActionErrorLog> = emptyList(),
    pendingApproval: Boolean = false,
    onApprove: () -> Unit = {},
    onReject: () -> Unit = {},
    onRetry: () -> Unit = {},
    onReconcile: () -> Unit = {},
    onDiagnostics: () -> Unit,
) {
    var expanded by remember(execution?.id) { mutableStateOf(false) }
    var traceOpen by remember(execution?.id) { mutableStateOf(true) }
    var errorsOpen by remember { mutableStateOf(false) }
    var recoveryOpen by remember { mutableStateOf(false) }
    val executionTraces = remember(execution?.id, traces) {
        if (execution == null) emptyList() else traces.filter { it.executionId == execution.id }.sortedBy { it.timestampMs }
    }

    SimplePage("فضای کار") {
        SectionCard("وضعیت Workspace") {
            StatusRow("مدل فعال", model?.displayName ?: "مدلی فعال نیست")
            StatusRow("Execution", execution?.id ?: "هیچ اجرای فعالی ثبت نشده است")
            StatusRow("وضعیت", execution?.status ?: "آماده")
            if (execution != null) StatusRow("مدت", durationText(execution))
        }

        SectionCard("Execution جاری") {
            if (execution == null) {
                Text("هنوز Executionای برای این گفتگو ثبت نشده است.", color = MaterialTheme.colorScheme.onSurfaceVariant)
            } else {
                Text(execution.action, style = MaterialTheme.typography.titleMedium)
                Text(execution.status, style = MaterialTheme.typography.bodyLarge)
                execution.error?.let { Text(it, color = MaterialTheme.colorScheme.error) }
                Text("شناسه: ${execution.id}", style = MaterialTheme.typography.bodySmall)
                TextButton(onClick = { expanded = !expanded }, Modifier.fillMaxWidth().heightIn(min = 48.dp)) {
                    Text(if (expanded) "▲ بستن جزئیات" else "▼ نمایش جزئیات")
                }
                if (expanded) {
                    DetailBlock("درخواست") { Text(execution.requestPreview.ifBlank { "N/A" }) }
                    DetailBlock("نتیجه") { Text(execution.resultPreview ?: "نتیجه‌ای ثبت نشده است.") }
                    DetailBlock("Approval") {
                        Text(if (execution.approvalRequired) "این عملیات نیازمند تأیید کاربر است." else "تأیید اضافی لازم نبود.")
                    }
                }
            }
        }

        if (pendingApproval || execution?.approvalRequired == true) {
            SectionCard("تأیید عملیات") {
                Text("این عملیات هنوز اجرا نشده و منتظر تصمیم شماست.")
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(onClick = onApprove, Modifier.weight(1f).heightIn(min = 48.dp)) { Text("تأیید و اجرا") }
                    OutlinedButton(onClick = onReject, Modifier.weight(1f).heightIn(min = 48.dp)) { Text("رد") }
                }
            }
        }

        if (execution?.status?.contains("ناموفق") == true) {
            SectionCard("بازیابی Execution") {
                Text("این Execution ناموفق بوده است. Retry فقط از مسیر کنترل‌شده Action انجام می‌شود.")
                Button(onClick = onRetry, Modifier.fillMaxWidth().heightIn(min = 48.dp)) { Text("تلاش مجدد") }
            }
        }

        SectionCard("Timeline واقعی Action") {
            if (executionTraces.isEmpty()) {
                Text("هنوز رویداد Lifecycle برای این Execution دریافت نشده است.", color = MaterialTheme.colorScheme.onSurfaceVariant)
            } else {
                TextButton(onClick = { traceOpen = !traceOpen }, Modifier.fillMaxWidth().heightIn(min = 48.dp)) {
                    Text(if (traceOpen) "▲ بستن Timeline" else "▼ نمایش Timeline")
                }
                if (traceOpen) {
                    executionTraces.forEachIndexed { index, event ->
                        TimelineItem(index + 1, event)
                    }
                }
            }
        }

        SectionCard("Recovery") {
            Text("Executionهای نیمه‌تمام به Unknown تبدیل می‌شوند تا بدون بررسی دوباره اجرا نشوند.")
            TextButton(onClick = { recoveryOpen = !recoveryOpen }, Modifier.fillMaxWidth().heightIn(min = 48.dp)) {
                Text(if (recoveryOpen) "▲ بستن Recovery" else "▼ مدیریت Recovery")
            }
            if (recoveryOpen) {
                Text("اگر برنامه هنگام اجرای Action متوقف شده باشد، وضعیت آن باید ابتدا reconcile شود.", style = MaterialTheme.typography.bodySmall)
                Button(onClick = onReconcile, Modifier.fillMaxWidth().heightIn(min = 48.dp)) { Text("بررسی Executionهای Interrupted") }
            }
        }

        SectionCard("خطاها") {
            TextButton(onClick = { errorsOpen = !errorsOpen }, Modifier.fillMaxWidth().heightIn(min = 48.dp)) {
                Text(if (errorsOpen) "▲ بستن خطاها (${errors.size})" else "▼ نمایش خطاها (${errors.size})")
            }
            if (errorsOpen) {
                val relevant = if (execution == null) errors else errors.filter { it.executionId == execution.id }
                if (relevant.isEmpty()) Text("خطای ثبت‌شده‌ای برای این Execution وجود ندارد.", color = MaterialTheme.colorScheme.onSurfaceVariant)
                relevant.takeLast(10).reversed().forEach { error ->
                    Surface(Modifier.fillMaxWidth().padding(top = 6.dp), shape = RoundedCornerShape(12.dp), color = MaterialTheme.colorScheme.errorContainer) {
                        Column(Modifier.padding(12.dp)) {
                            Text(error.userMessage, style = MaterialTheme.typography.bodyMedium)
                            Text(error.rawError, style = MaterialTheme.typography.bodySmall)
                        }
                    }
                }
            }
        }

        SectionCard("نتیجه و Diagnostics") {
            Text("برای اطلاعات Runtime، Performance و گزارش خام از صفحه عیب‌یابی استفاده کنید.", color = MaterialTheme.colorScheme.onSurfaceVariant)
            OutlinedButton(onClick = onDiagnostics, Modifier.fillMaxWidth().heightIn(min = 48.dp)) { Text("مشاهده Diagnostics →") }
        }
    }
}

@Composable
private fun SectionCard(title: String, content: @Composable ColumnScope.() -> Unit) {
    Card(Modifier.fillMaxWidth(), shape = RoundedCornerShape(20.dp)) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp), content = {
            Text(title, style = MaterialTheme.typography.titleLarge)
            content()
        })
    }
}

@Composable
private fun StatusRow(label: String, value: String) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        Text(label, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text(value, style = MaterialTheme.typography.bodyMedium)
    }
}

@Composable
private fun DetailBlock(title: String, content: @Composable () -> Unit) {
    Column(Modifier.fillMaxWidth().padding(top = 6.dp)) {
        Text(title, style = MaterialTheme.typography.labelLarge)
        Surface(Modifier.fillMaxWidth().padding(top = 4.dp), shape = RoundedCornerShape(12.dp), color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = .45f)) {
            Column(Modifier.padding(10.dp), content = { content() })
        }
    }
}

@Composable
private fun TimelineItem(index: Int, event: ActionTraceEvent) {
    Surface(
        Modifier.fillMaxWidth().padding(top = 6.dp).semantics { contentDescription = "مرحله $index: ${event.type}" },
        shape = RoundedCornerShape(12.dp),
        color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = .45f),
    ) {
        Column(Modifier.padding(12.dp)) {
            Text("$index. ${event.type}", style = MaterialTheme.typography.titleSmall)
            event.message?.let { Text(it, style = MaterialTheme.typography.bodySmall) }
            Text(event.timestampMs.toString(), style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}

private fun durationText(execution: ExecutionState): String {
    val end = execution.finishedAt ?: System.currentTimeMillis()
    return "${(end - execution.startedAt).coerceAtLeast(0L)} ms"
}

@Composable
private fun SimplePage(title: String, content: @Composable ColumnScope.() -> Unit) {
    LazyColumn(
        Modifier.fillMaxSize().padding(20.dp),
        contentPadding = PaddingValues(bottom = 32.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        item { Text(title, style = MaterialTheme.typography.headlineMedium) }
        item { Column(verticalArrangement = Arrangement.spacedBy(12.dp), content = content) }
    }
}
