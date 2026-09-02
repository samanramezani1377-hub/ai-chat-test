package com.samanramezani.aichattest.ui.workspace

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import com.samanramezani.aichattest.ui.state.ExecutionState
import com.woogit.aicore.domain.ModelDescriptor

@Composable
internal fun WorkspacePage(execution: ExecutionState?, model: ModelDescriptor?, onDiagnostics: () -> Unit) {
    var expanded by remember(execution?.id) { mutableStateOf(false) }
    SimplePage("فضای کار") {
        Text("خلاصه اجرای جاری", style = MaterialTheme.typography.titleLarge)
        if (execution == null) Text("اجرای فعالی ثبت نشده است.", color = MaterialTheme.colorScheme.onSurfaceVariant) else {
            Text(execution.action, style = MaterialTheme.typography.titleMedium)
            Text(execution.status)
            Text("مدل: ${model?.displayName ?: "مدل محلی"}", style = MaterialTheme.typography.bodySmall)
            Text("شناسه Execution: ${execution.id}", style = MaterialTheme.typography.bodySmall)
            Text("Timeline مرکزی", style = MaterialTheme.typography.titleLarge)
            Surface(Modifier.fillMaxWidth().clickable { expanded = !expanded }.semantics { contentDescription = "جزئیات اجرای ${execution.action}" }, shape = RoundedCornerShape(18.dp), color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = .5f)) {
                Column(Modifier.padding(14.dp)) {
                    Text("● ${execution.action}", style = MaterialTheme.typography.titleMedium)
                    Text("${execution.status} · ${durationText(execution)}", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    execution.error?.let { Text(it, color = MaterialTheme.colorScheme.error) }
                    Text(if (expanded) "▲ بستن جزئیات" else "▼ نمایش جزئیات", style = MaterialTheme.typography.labelMedium, modifier = Modifier.padding(top = 8.dp))
                    if (expanded) {
                        ExpandLayer("اطلاعات درخواست") { Text(execution.requestPreview, style = MaterialTheme.typography.bodySmall) }
                        ExpandLayer("Preview / Result") { Text(execution.resultPreview ?: "نتیجه‌ای ثبت نشده است.", style = MaterialTheme.typography.bodySmall) }
                        ExpandLayer("Approval") { Text(if (execution.approvalRequired) "این عملیات برای ادامه نیازمند تأیید کاربر است." else "تأیید اضافی لازم نبود.") }
                        TextButton(onClick = onDiagnostics, Modifier.fillMaxWidth().heightIn(min = 48.dp)) { Text("مشاهده جزئیات → عیب‌یابی") }
                    }
                }
            }
        }
        Text("تاریخچه عملکرد", style = MaterialTheme.typography.titleLarge)
        Text("آمار فقط در صورت دریافت داده واقعی Runtime نمایش داده می‌شود.", color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@Composable private fun ExpandLayer(title: String, content: @Composable ColumnScope.() -> Unit) {
    var open by remember { mutableStateOf(false) }
    Column(Modifier.fillMaxWidth().padding(top = 8.dp)) {
        TextButton(onClick = { open = !open }, Modifier.fillMaxWidth().heightIn(min = 48.dp), contentPadding = PaddingValues(8.dp)) { Text(if (open) "▲ $title" else "▼ $title", Modifier.fillMaxWidth()) }
        if (open) Column(Modifier.fillMaxWidth().padding(horizontal = 8.dp), content = content)
    }
}

private fun durationText(execution: ExecutionState): String { val end = execution.finishedAt ?: System.currentTimeMillis(); return "${(end - execution.startedAt).coerceAtLeast(0L)} ms" }

@Composable
private fun SimplePage(title: String, content: @Composable ColumnScope.() -> Unit) {
    LazyColumn(Modifier.fillMaxSize().padding(20.dp), contentPadding = PaddingValues(bottom = 32.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        item { Text(title, style = MaterialTheme.typography.headlineMedium) }
        item { Column(verticalArrangement = Arrangement.spacedBy(10.dp), content = content) }
    }
}
