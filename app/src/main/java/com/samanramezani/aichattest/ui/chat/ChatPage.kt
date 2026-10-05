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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.samanramezani.aichattest.ui.state.ExecutionState
import com.samanramezani.aichattest.ui.state.UiMessage
import com.woogit.aicore.domain.ChatMessage

@Composable
internal fun ChatPage(
    messages: List<UiMessage>, composer: String, generating: Boolean, approvalBusy: Boolean,
    onComposer: (String) -> Unit, onSend: () -> Unit, onStop: () -> Unit,
    execution: ExecutionState?, onApprove: () -> Unit, onReject: () -> Unit, onWorkspace: () -> Unit,
) {
    Column(Modifier.fillMaxSize()) {
        LazyColumn(
            Modifier.weight(1f).fillMaxWidth(),
            contentPadding = PaddingValues(start = 18.dp, end = 18.dp, top = 8.dp, bottom = 8.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            if (messages.isEmpty()) item { WelcomeState() }
            items(messages, key = { it.id }) { MessageRow(it) }
            if (generating) item {
                Surface(Modifier.wrapContentWidth().padding(start = 4.dp), shape = RoundedCornerShape(18.dp), color = MaterialTheme.colorScheme.surfaceVariant) {
                    Text(if (approvalBusy) "در حال پردازش تأیید…" else "در حال فکر کردن…", Modifier.padding(horizontal = 14.dp, vertical = 10.dp), color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
            if (!generating && execution != null && execution.action != "درخواست Agent") {
                item { ActionSummary(execution, onWorkspace, onApprove, onReject, approvalBusy) }
            }
        }
        Composer(composer, generating, approvalBusy, onComposer, onSend, onStop)
    }
}

@Composable
private fun WelcomeState() {
    Column(Modifier.fillMaxWidth().padding(horizontal = 22.dp, vertical = 50.dp), horizontalAlignment = Alignment.CenterHorizontally) {
        Surface(Modifier.size(70.dp), shape = RoundedCornerShape(24.dp), color = MaterialTheme.colorScheme.primaryContainer) {
            Box(contentAlignment = Alignment.Center) { Text("AI", style = MaterialTheme.typography.titleLarge, color = MaterialTheme.colorScheme.primary) }
        }
        Spacer(Modifier.height(18.dp))
        Text("چه کاری می‌توانم برایت انجام بدهم؟", style = MaterialTheme.typography.headlineSmall, textAlign = TextAlign.Center)
        Spacer(Modifier.height(7.dp))
        Text("پیامت را بنویس؛ پاسخ روی خود دستگاه و با مدل محلی تولید می‌شود.", Modifier.widthIn(max = 560.dp), textAlign = TextAlign.Center, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Spacer(Modifier.height(18.dp))
        Surface(Modifier.widthIn(max = 600.dp).fillMaxWidth(), shape = RoundedCornerShape(18.dp), color = MaterialTheme.colorScheme.secondaryContainer) {
            Text("خصوصی و محلی · بدون API · llama.cpp + OpenCL", Modifier.padding(14.dp), textAlign = TextAlign.Center, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSecondaryContainer)
        }
    }
}

@Composable
private fun Composer(
    value: String, generating: Boolean, approvalBusy: Boolean,
    onValueChange: (String) -> Unit, onSend: () -> Unit, onStop: () -> Unit,
) {
    Surface(
        Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 10.dp),
        shape = RoundedCornerShape(28.dp), color = MaterialTheme.colorScheme.surface, shadowElevation = 5.dp,
    ) {
        Row(Modifier.fillMaxWidth().padding(7.dp), verticalAlignment = Alignment.Bottom) {
            TextField(
                value = value, onValueChange = onValueChange, modifier = Modifier.weight(1f),
                placeholder = { Text("پیام خود را بنویسید…") }, maxLines = 6,
                shape = RoundedCornerShape(21.dp),
                colors = TextFieldDefaults.colors(
                    focusedContainerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = .55f),
                    unfocusedContainerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = .55f),
                    disabledContainerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = .4f),
                    focusedIndicatorColor = Color.Transparent, unfocusedIndicatorColor = Color.Transparent,
                ),
            )
            Spacer(Modifier.width(8.dp))
            FilledIconButton(
                onClick = if (generating) onStop else onSend,
                enabled = !approvalBusy && (generating || value.isNotBlank()),
                modifier = Modifier.size(50.dp).semantics { contentDescription = if (generating) "توقف تولید" else "ارسال پیام" },
                colors = IconButtonDefaults.filledIconButtonColors(
                    containerColor = if (generating) MaterialTheme.colorScheme.tertiary else MaterialTheme.colorScheme.primary,
                ),
            ) {
                Icon(if (generating) Icons.Default.Stop else Icons.Default.ArrowUpward, if (generating) "توقف تولید" else "ارسال پیام")
            }
        }
    }
}

@Composable
private fun MessageRow(message: UiMessage) {
    val user = message.role == ChatMessage.Role.USER
    Column(Modifier.fillMaxWidth(), horizontalAlignment = if (user) Alignment.End else Alignment.Start) {
        Surface(
            Modifier.widthIn(max = 760.dp),
            shape = RoundedCornerShape(
                topStart = 20.dp, topEnd = 20.dp,
                bottomStart = if (user) 20.dp else 6.dp,
                bottomEnd = if (user) 6.dp else 20.dp,
            ),
            color = if (user) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.surface,
            shadowElevation = if (user) 0.dp else 1.dp,
        ) {
            Column(Modifier.padding(horizontal = 15.dp, vertical = 12.dp)) {
                Text(
                    if (user) "شما" else "مدل محلی",
                    style = MaterialTheme.typography.labelSmall,
                    color = if (user) MaterialTheme.colorScheme.onPrimary.copy(alpha = .78f) else MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Spacer(Modifier.height(4.dp))
                Text(message.text.ifBlank { "…" }, color = if (user) MaterialTheme.colorScheme.onPrimary else MaterialTheme.colorScheme.onSurface, style = MaterialTheme.typography.bodyLarge)
            }
        }
    }
}

@Composable
private fun ActionSummary(
    execution: ExecutionState, onWorkspace: () -> Unit, onApprove: () -> Unit, onReject: () -> Unit, approvalBusy: Boolean,
) {
    Surface(Modifier.fillMaxWidth(), shape = RoundedCornerShape(20.dp), color = MaterialTheme.colorScheme.tertiaryContainer) {
        Column(Modifier.padding(15.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text("عملیات", style = MaterialTheme.typography.titleSmall)
                    Text(execution.action, style = MaterialTheme.typography.bodyLarge)
                    Text(execution.status + " · " + durationText(execution), style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
                TextButton(onClick = onWorkspace, Modifier.heightIn(min = 48.dp)) { Text("جزئیات") }
            }
            if (execution.approvalRequired) {
                Text("این عملیات حساس است و قبل از اجرا به تأیید شما نیاز دارد.", style = MaterialTheme.typography.bodySmall)
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(onClick = onApprove, enabled = !approvalBusy, Modifier.weight(1f).heightIn(min = 48.dp)) { Text(if (approvalBusy) "در حال اجرا…" else "تأیید و اجرا") }
                    OutlinedButton(onClick = onReject, enabled = !approvalBusy, Modifier.weight(1f).heightIn(min = 48.dp)) { Text("رد") }
                }
            }
        }
    }
}

private fun durationText(execution: ExecutionState): String {
    val end = execution.finishedAt ?: System.currentTimeMillis()
    return ((end - execution.startedAt).coerceAtLeast(0L)).toString() + " ms"
}
