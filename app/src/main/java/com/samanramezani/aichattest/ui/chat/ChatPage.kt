package com.samanramezani.aichattest.ui.chat

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowUpward
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import com.samanramezani.aichattest.ui.state.ExecutionState
import com.samanramezani.aichattest.ui.state.UiMessage
import com.woogit.aicore.domain.ChatMessage

@Composable
internal fun ChatPage(messages: List<UiMessage>, composer: String, generating: Boolean, approvalBusy: Boolean, onComposer: (String) -> Unit, onSend: () -> Unit, onStop: () -> Unit, execution: ExecutionState?, onApprove: () -> Unit, onReject: () -> Unit, onWorkspace: () -> Unit) {
    Column(Modifier.fillMaxSize().padding(horizontal = 16.dp)) {
        LazyColumn(Modifier.weight(1f).fillMaxWidth(), contentPadding = PaddingValues(vertical = 20.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
            if (messages.isEmpty()) item { Column(Modifier.fillMaxWidth().padding(top = 64.dp), horizontalAlignment = Alignment.CenterHorizontally) { Text("گفت‌وگو", style = MaterialTheme.typography.headlineMedium); Spacer(Modifier.height(6.dp)); Text("پیام خود را بنویسید و گفتگو را شروع کنید.", color = MaterialTheme.colorScheme.onSurfaceVariant) } }
            items(messages, key = { it.id }) { MessageRow(it) }
            if (generating) item { Text(if (approvalBusy) "در حال پردازش تأیید…" else "در حال اجرای Agent…", color = MaterialTheme.colorScheme.onSurfaceVariant) }
            if (!generating && execution != null) item { ActionSummary(execution, onWorkspace, onApprove, onReject, approvalBusy) }
        }
        Surface(Modifier.fillMaxWidth().padding(bottom = 12.dp), shape = RoundedCornerShape(50), color = MaterialTheme.colorScheme.surface.copy(alpha = .96f), tonalElevation = 2.dp) {
            Row(Modifier.padding(horizontal = 8.dp, vertical = 6.dp), verticalAlignment = Alignment.Bottom) {
                TextField(value = composer, onValueChange = onComposer, modifier = Modifier.weight(1f), placeholder = { Text("پیام خود را بنویسید…") }, maxLines = 6, shape = RoundedCornerShape(50))
                Spacer(Modifier.width(8.dp))
                FilledIconButton(onClick = if (generating) onStop else onSend, enabled = if (approvalBusy) false else generating || composer.isNotBlank(), modifier = Modifier.size(48.dp).semantics { contentDescription = if (generating) "توقف تولید" else "ارسال پیام" }) {
                    Icon(if (generating) Icons.Default.Stop else Icons.Default.ArrowUpward, if (generating) "توقف تولید" else "ارسال پیام", modifier = Modifier.size(20.dp))
                }
            }
        }
    }
}

@Composable private fun MessageRow(message: UiMessage) {
    val user = message.role == ChatMessage.Role.USER
    Column(Modifier.fillMaxWidth(), horizontalAlignment = if (user) Alignment.Start else Alignment.End) {
        Text(if (user) "شما" else "مدل", style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        if (user) Surface(color = MaterialTheme.colorScheme.primaryContainer.copy(alpha = .70f), shape = RoundedCornerShape(18.dp)) { Text(message.text, Modifier.padding(13.dp)) } else Text(message.text, Modifier.fillMaxWidth(.94f))
    }
}

@Composable private fun ActionSummary(execution: ExecutionState, onWorkspace: () -> Unit, onApprove: () -> Unit, onReject: () -> Unit, approvalBusy: Boolean) {
    Surface(Modifier.fillMaxWidth(), shape = RoundedCornerShape(18.dp), color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = .55f)) {
        Column(Modifier.padding(12.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text("آخرین عملیات: ${execution.action}", style = MaterialTheme.typography.titleSmall)
                    Text("${execution.status} · ${durationText(execution)}", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    if (execution.approvalRequired) Text("این عملیات نیاز به تأیید دارد.", color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.labelSmall)
                }
                TextButton(onClick = onWorkspace, Modifier.heightIn(min = 48.dp)) { Text("مشاهده جزئیات") }
            }
            if (execution.approvalRequired) {
                Spacer(Modifier.height(8.dp)); Text("این عملیات حساس است و بدون تأیید شما اجرا نمی‌شود.", style = MaterialTheme.typography.bodySmall); Spacer(Modifier.height(8.dp))
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(onClick = onApprove, enabled = !approvalBusy, modifier = Modifier.weight(1f).heightIn(min = 48.dp)) { Text(if (approvalBusy) "در حال پردازش…" else "تأیید و اجرا") }
                    OutlinedButton(onClick = onReject, enabled = !approvalBusy, modifier = Modifier.weight(1f).heightIn(min = 48.dp)) { Text("رد") }
                }
            }
        }
    }
}

private fun durationText(execution: ExecutionState): String { val end = execution.finishedAt ?: System.currentTimeMillis(); return "${(end - execution.startedAt).coerceAtLeast(0L)} ms" }
