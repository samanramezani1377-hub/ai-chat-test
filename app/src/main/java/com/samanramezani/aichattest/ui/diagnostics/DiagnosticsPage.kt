package com.samanramezani.aichattest.ui.diagnostics

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.samanramezani.aichattest.ui.state.ExecutionState
import java.text.DateFormat
import java.util.Date

@Composable
internal fun DiagnosticsPage(error: String?, execution: ExecutionState?) {
    SimplePage("عیب‌یابی") {
        Text("خطاها", style = MaterialTheme.typography.titleLarge)
        if (error == null) Text("خطایی برای نمایش ثبت نشده است.", color = MaterialTheme.colorScheme.onSurfaceVariant) else Text(error, color = MaterialTheme.colorScheme.error)
        Text("گزارش Execution / Trace", style = MaterialTheme.typography.titleLarge)
        if (execution == null) Text("Execution ثبت‌شده‌ای وجود ندارد.", color = MaterialTheme.colorScheme.onSurfaceVariant) else {
            Text("Execution: ${execution.id}", style = MaterialTheme.typography.bodySmall)
            Text("Action: ${execution.action}")
            Text("وضعیت: ${execution.status}")
            Text("شروع: ${DateFormat.getDateTimeInstance().format(Date(execution.startedAt))}", style = MaterialTheme.typography.bodySmall)
            execution.finishedAt?.let { Text("پایان: ${DateFormat.getDateTimeInstance().format(Date(it))}", style = MaterialTheme.typography.bodySmall) }
            execution.error?.let { Text("Error: $it", color = MaterialTheme.colorScheme.error) }
            Text("Trace", style = MaterialTheme.typography.titleMedium)
            Text("جزئیات Trace فقط از Execution واقعی نمایش داده می‌شود.", color = MaterialTheme.colorScheme.onSurfaceVariant)
            Text("Approval: ${if (execution.approvalRequired) "لازم است" else "لازم نیست"}")
        }
    }
}

@Composable
private fun SimplePage(title: String, content: @Composable ColumnScope.() -> Unit) {
    LazyColumn(Modifier.fillMaxSize().padding(20.dp), contentPadding = PaddingValues(bottom = 32.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        item { Text(title, style = MaterialTheme.typography.headlineMedium) }
        item { Column(verticalArrangement = Arrangement.spacedBy(10.dp), content = content) }
    }
}
