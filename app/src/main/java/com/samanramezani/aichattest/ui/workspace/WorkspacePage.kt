package com.samanramezani.aichattest.ui.workspace

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import com.samanramezani.aichattest.AppContainer
import com.samanramezani1377.aichattest.ui.components.SimplePage
import com.samanramezani.aichattest.ui.state.ExecutionState
import com.woogit.aicore.actions.ActionErrorLog
import com.woogit.aicore.actions.ActionExecutionState
import com.woogit.aicore.actions.ActionTraceEvent
import com.woogit.aicore.actions.ActionTraceStore
import com.woogit.aicore.actions.PreparedAction
import com.woogit.aicore.domain.ModelDescriptor
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

@Composable
internal fun WorkspacePage(execution: ExecutionState?, model: ModelDescriptor?, onDiagnostics: () -> Unit) {
    val container = AppContainer.latest
    val scope = rememberCoroutineScope()
    val liveTraces by ActionTraceStore.events.collectAsState()
    var checkpoint by remember(execution?.id) { mutableStateOf<PreparedAction?>(null) }
    var reconcileResult by remember { mutableStateOf<List<PreparedAction>>(emptyList()) }
    var actionBusy by remember { mutableStateOf(false) }
    var actionMessage by remember { mutableStateOf<String?>(null) }
    var expanded by remember(execution?.id) { mutableStateOf(false) }
    var traceOpen by remember(execution?.id) { mutableStateOf(true) }
    var errorsOpen by remember { mutableStateOf(false) }
    var recoveryOpen by remember { mutableStateOf(false) }

    LaunchedEffect(execution?.id, liveTraces) {
        val id = execution?.id ?: return@LaunchedEffect
        checkpoint = container?.let { withContext(Dispatchers.Default) { it.executionCheckpoint(id) } }
    }

    val executionTraces = remember(execution?.id, liveTraces) {
        if (execution == null) emptyList() else liveTraces.filter { it.executionId == execution.id }.sortedBy { it.timestampMs }
    }
    val errors: List<ActionErrorLog> = remember(container, liveTraces) { container?.actionErrors().orEmpty() }
    val isAwaitingApproval = checkpoint?.state == ActionExecutionState.AwaitingApproval || execution?.approvalRequired == true
    val isFailed = checkpoint?.state is ActionExecutionState.Failed || execution?.status?.contains("ناموفق") == true

    SimplePage("فضای کار") {
        SectionCard("وضعیت Workspace") {
            StatusRow("مدل فعال", model?.displayName ?: "مدلی فعال نیست")
            StatusRow("Execution", execution?.id ?: "هیچ اجرای فعالی ثبت نشده است")
            StatusRow("Lifecycle", checkpoint?.state?.toString() ?: execution?.status ?: "آماده")
            if (execution != null) StatusRow("مدت", durationText(execution))
            actionMessage?.let { Text(it, color = MaterialTheme.colorScheme.primary) }
        }

        SectionCard("Execution جاری") {
            if (execution == null) Text("هنوز Executionای برای این گفتگو ثبت نشده است.", color = MaterialTheme.colorScheme.onSurfaceVariant)
            else {
                Text(execution.action, style = MaterialTheme.typography.titleMedium)
                Text(execution.status, style = MaterialTheme.typography.bodyLarge)
                execution.error?.let { Text(it, color = MaterialTheme.colorScheme.error) }
                Text("شناسه: ${execution.id}", style = MaterialTheme.typography.bodySmall)
                TextButton(onClick = { expanded = !expanded }, modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp)) { Text(if (expanded) "▲ بستن جزئیات" else "▼ نمایش جزئیات") }
                if (expanded) {
                    DetailBlock("درخواست") { Text(execution.requestPreview.ifBlank { "N/A" }) }
                    DetailBlock("نتیجه") { Text(execution.resultPreview ?: "نتیجه‌ای ثبت نشده است.") }
                    DetailBlock("Approval") { Text(if (isAwaitingApproval) "در انتظار تأیید کاربر" else "نیاز به تأیید ندارد") }
                }
            }
        }

        if (isAwaitingApproval && checkpoint != null) {
            SectionCard("تأیید عملیات") {
                Text("این Action در Checkpoint ذخیره شده و هنوز اجرا نشده است.")
                Text("Action: ${checkpoint!!.actionId}", style = MaterialTheme.typography.titleSmall)
                Text("Risk: ${checkpoint!!.risk}", style = MaterialTheme.typography.bodySmall)
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(enabled = !actionBusy, onClick = {
                        val id = checkpoint!!.executionId
                        actionBusy = true
                        actionMessage = null
                        scope.launch {
                            val result = withContext(Dispatchers.Default) { container?.approveAndExecute(id) }
                            actionMessage = result?.message ?: "Execution انجام شد."
                            checkpoint = container?.executionCheckpoint(id)
                            actionBusy = false
                        }
                    }, modifier = Modifier.weight(1f).heightIn(min = 48.dp)) { Text(if (actionBusy) "در حال اجرا…" else "تأیید و اجرا") }
                    OutlinedButton(enabled = !actionBusy, onClick = {
                        val id = checkpoint!!.executionId
                        actionBusy = true
                        scope.launch {
                            val ok = withContext(Dispatchers.Default) { container?.reject(id) == true }
                            actionMessage = if (ok) "Action رد شد." else "رد Action ناموفق بود."
                            checkpoint = container?.executionCheckpoint(id)
                            actionBusy = false
                        }
                    }, modifier = Modifier.weight(1f).heightIn(min = 48.dp)) { Text("رد") }
                }
            }
        }

        if (isFailed && checkpoint != null) {
            SectionCard("بازیابی و Retry") {
                Text("Retry فقط از مسیر ActionLifecycle و با محدودیت retry و بررسی idempotency انجام می‌شود.")
                Button(enabled = !actionBusy, onClick = {
                    val id = checkpoint!!.executionId
                    actionBusy = true
                    scope.launch {
                        val result = withContext(Dispatchers.Default) { container?.retryAction(id) }
                        actionMessage = result?.message ?: "Retry انجام شد."
                        checkpoint = container?.executionCheckpoint(id)
                        actionBusy = false
                    }
                }, modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp)) { Text(if (actionBusy) "در حال Retry…" else "تلاش مجدد") }
            }
        }

        SectionCard("Timeline واقعی Action") {
            if (executionTraces.isEmpty()) Text("هنوز رویداد Lifecycle برای این Execution دریافت نشده است.", color = MaterialTheme.colorScheme.onSurfaceVariant)
            else {
                TextButton(onClick = { traceOpen = !traceOpen }, modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp)) { Text(if (traceOpen) "▲ بستن Timeline" else "▼ نمایش Timeline") }
                if (traceOpen) executionTraces.forEachIndexed { index, event -> TimelineItem(index + 1, event) }
            }
        }

        SectionCard("Recovery") {
            Text("Executionهای باقی‌مانده در حالت Executing پس از قطع ناگهانی به Unknown تبدیل می‌شوند و خودکار دوباره اجرا نمی‌شوند.")
            TextButton(onClick = { recoveryOpen = !recoveryOpen }, modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp)) { Text(if (recoveryOpen) "▲ بستن Recovery" else "▼ مدیریت Recovery") }
            if (recoveryOpen) {
                Button(enabled = !actionBusy, onClick = {
                    actionBusy = true
                    scope.launch {
                        reconcileResult = withContext(Dispatchers.Default) { container?.reconcileInterruptedActions().orEmpty() }
                        actionMessage = if (reconcileResult.isEmpty()) "Execution نیمه‌تمامی برای reconcile پیدا نشد." else "${reconcileResult.size} Execution به Unknown منتقل شد."
                        actionBusy = false
                    }
                }, modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp)) { Text("بررسی Executionهای Interrupted") }
                reconcileResult.takeLast(10).forEach { Text("• ${it.executionId}: Unknown") }
            }
        }

        SectionCard("خطاها") {
            TextButton(onClick = { errorsOpen = !errorsOpen }, modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp)) { Text(if (errorsOpen) "▲ بستن خطاها (${errors.size})" else "▼ نمایش خطاها (${errors.size})") }
            if (errorsOpen) {
                val relevant = if (execution == null) errors else errors.filter { it.executionId == execution.id }
                if (relevant.isEmpty()) Text("خطای ثبت‌شده‌ای برای این Execution وجود ندارد.", color = MaterialTheme.colorScheme.onSurfaceVariant)
                relevant.takeLast(10).reversed().forEach { error ->
                    Surface(Modifier.fillMaxWidth().padding(top = 6.dp), shape = RoundedCornerShape(12.dp), color = MaterialTheme.colorScheme.errorContainer) {
                        Column(Modifier.padding(12.dp)) { Text(error.userMessageFa); Text(error.rawMachineError, style = MaterialTheme.typography.bodySmall) }
                    }
                }
            }
        }

        SectionCard("نتیجه و Diagnostics") {
            Text("برای اطلاعات Runtime، Performance و گزارش خام از صفحه عیب‌یابی استفاده کنید.", color = MaterialTheme.colorScheme.onSurfaceVariant)
            OutlinedButton(onClick = onDiagnostics, modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp)) { Text("مشاهده Diagnostics →") }
        }
    }
}

@Composable private fun SectionCard(title: String, content: @Composable ColumnScope.() -> Unit) {
    Card(Modifier.fillMaxWidth(), shape = RoundedCornerShape(20.dp)) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) { Text(title, style = MaterialTheme.typography.titleLarge); content() }
    }
}

@Composable private fun StatusRow(label: String, value: String) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) { Text(label, color = MaterialTheme.colorScheme.onSurfaceVariant); Text(value, style = MaterialTheme.typography.bodyMedium) }
}

@Composable private fun DetailBlock(title: String, content: @Composable () -> Unit) {
    Column(Modifier.fillMaxWidth().padding(top = 6.dp)) {
        Text(title, style = MaterialTheme.typography.labelLarge)
        Surface(Modifier.fillMaxWidth().padding(top = 4.dp), shape = RoundedCornerShape(12.dp), color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = .45f)) { Column(Modifier.padding(10.dp)) { content() } }
    }
}

@Composable private fun TimelineItem(index: Int, event: ActionTraceEvent) {
    Surface(Modifier.fillMaxWidth().padding(top = 6.dp).semantics { contentDescription = "مرحله $index: ${event.type}" }, shape = RoundedCornerShape(12.dp), color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = .45f)) {
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
