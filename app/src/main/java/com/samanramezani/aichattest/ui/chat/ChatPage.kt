package com.samanramezani.aichattest.ui.chat

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowUpward
import androidx.compose.material.icons.filled.ContentCopy
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
import com.samanramezani.aichattest.ui.state.ExecutionStep
import com.samanramezani.aichattest.ui.state.StepState
import com.samanramezani.aichattest.ui.state.UiMessage
import com.woogit.aicore.domain.ChatMessage

@Composable
internal fun ChatPage(
    messages: List<UiMessage>, composer: String, generating: Boolean, approvalBusy: Boolean,
    onComposer: (String) -> Unit, onSend: () -> Unit, onStop: () -> Unit,
    execution: ExecutionState?, onApprove: () -> Unit, onReject: () -> Unit, onWorkspace: () -> Unit, onCopy: (String) -> Unit, onVoice: () -> Unit,
) {
    Column(Modifier.fillMaxSize()) {
        Row(Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 4.dp), horizontalArrangement = Arrangement.End) {
            OutlinedButton(onClick = onVoice, modifier = Modifier.heightIn(min = 42.dp)) { Text("مکالمه صوتی زنده") }
        }
        LazyColumn(Modifier.weight(1f).fillMaxWidth(), contentPadding = PaddingValues(horizontal = 16.dp, vertical = 8.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
            if (messages.isEmpty()) item { WelcomeState() }
            items(messages, key = { it.id }) { MessageRow(it, onCopy) }
            if (generating) item { AgentProgress(execution, approvalBusy) }
            if (!generating && execution != null && execution.action != "درخواست Agent") item { ActionSummary(execution, onWorkspace, onApprove, onReject, approvalBusy) }
        }
        Composer(composer, generating, approvalBusy, onComposer, onSend, onStop)
    }
}

@Composable private fun WelcomeState() {
    Column(Modifier.fillMaxWidth().padding(horizontal = 20.dp, vertical = 54.dp), horizontalAlignment = Alignment.CenterHorizontally) {
        Surface(Modifier.size(76.dp), shape = RoundedCornerShape(26.dp), color = MaterialTheme.colorScheme.primaryContainer) {
            Box(contentAlignment = Alignment.Center) { Text("✦", style = MaterialTheme.typography.headlineLarge, color = MaterialTheme.colorScheme.primary) }
        }
        Spacer(Modifier.height(20.dp))
        Text("شروع کنیم.", style = MaterialTheme.typography.headlineLarge, textAlign = TextAlign.Center)
        Spacer(Modifier.height(8.dp))
        Text("هر چیزی را بنویس. پاسخ همین‌جا و روی خود دستگاه تولید می‌شود.", Modifier.widthIn(max = 520.dp), textAlign = TextAlign.Center, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Spacer(Modifier.height(22.dp))
        Row(Modifier.widthIn(max = 560.dp).fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            HintChip("خلاصه‌کردن", Modifier.weight(1f))
            HintChip("ایده‌پردازی", Modifier.weight(1f))
            HintChip("نوشتن کد", Modifier.weight(1f))
        }
        Spacer(Modifier.height(18.dp))
        Text("خصوصی · آفلاین · llama.cpp + OpenCL", style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.secondary)
    }
}

@Composable private fun HintChip(text: String, modifier: Modifier = Modifier) {
    Surface(modifier, shape = RoundedCornerShape(14.dp), color = MaterialTheme.colorScheme.surfaceVariant) {
        Text(text, Modifier.padding(horizontal = 10.dp, vertical = 10.dp), textAlign = TextAlign.Center, style = MaterialTheme.typography.labelSmall)
    }
}

@Composable
private fun AgentProgress(execution: ExecutionState?, approvalBusy: Boolean) {
    val steps = execution?.steps.orEmpty()
    Surface(
        Modifier.fillMaxWidth().padding(horizontal = 4.dp),
        shape = RoundedCornerShape(18.dp),
        color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = .55f)
    ) {
        Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(9.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Surface(Modifier.size(8.dp), shape = RoundedCornerShape(50), color = MaterialTheme.colorScheme.primary) {}
                Spacer(Modifier.width(9.dp))
                Text(
                    when {
                        approvalBusy -> "در حال اجرای مرحله تأیید…"
                        steps.any { it.state == StepState.ACTIVE } -> steps.first { it.state == StepState.ACTIVE }.title
                        else -> "در حال آماده‌سازی پاسخ…"
                    },
                    style = MaterialTheme.typography.labelLarge
                )
            }
            LinearProgressIndicator(Modifier.fillMaxWidth().height(3.dp))
            if (steps.isNotEmpty()) {
                Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    steps.takeLast(5).forEach { step -> ProgressStep(step) }
                }
            }
        }
    }
}

@Composable
private fun ProgressStep(step: ExecutionStep) {
    val icon = when (step.state) {
        StepState.DONE -> "✓"
        StepState.FAILED -> "!"
        StepState.ACTIVE -> "•"
        StepState.PENDING -> "○"
    }
    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.Top) {
        Text(icon, color = if (step.state == StepState.FAILED) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.primary, fontWeight = androidx.compose.ui.text.font.FontWeight.Bold)
        Spacer(Modifier.width(8.dp))
        Column(Modifier.weight(1f)) {
            Text(step.title, style = MaterialTheme.typography.bodyMedium)
            if (step.detail.isNotBlank()) Text(step.detail, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}

@Composable private fun Composer(value: String, generating: Boolean, approvalBusy: Boolean, onValueChange: (String) -> Unit, onSend: () -> Unit, onStop: () -> Unit) {
    Surface(Modifier.fillMaxWidth().padding(horizontal = 10.dp, vertical = 8.dp), shape = RoundedCornerShape(24.dp), color = MaterialTheme.colorScheme.surface, tonalElevation = 3.dp) {
        Column(Modifier.padding(8.dp)) {
            TextField(
                value = value, onValueChange = onValueChange, modifier = Modifier.fillMaxWidth(),
                placeholder = { Text("پیام خود را بنویس…") }, minLines = 1, maxLines = 7,
                shape = RoundedCornerShape(18.dp),
                colors = TextFieldDefaults.colors(
                    focusedContainerColor = Color.Transparent, unfocusedContainerColor = Color.Transparent,
                    disabledContainerColor = Color.Transparent, focusedIndicatorColor = Color.Transparent,
                    unfocusedIndicatorColor = Color.Transparent,
                ),
            )
            Row(Modifier.fillMaxWidth().padding(horizontal = 2.dp), verticalAlignment = Alignment.CenterVertically) {
                Text(if (generating) "تولید روی دستگاه در حال اجراست" else "Enter برای ارسال · متن بلند هم مشکلی ندارد", Modifier.weight(1f).padding(start = 10.dp), style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                FilledIconButton(
                    onClick = if (generating) onStop else onSend,
                    enabled = !approvalBusy && (generating || value.isNotBlank()),
                    modifier = Modifier.size(48.dp).semantics { contentDescription = if (generating) "توقف تولید" else "ارسال پیام" },
                ) { Icon(if (generating) Icons.Default.Stop else Icons.Default.ArrowUpward, null) }
            }
        }
    }
}

@Composable
private fun MessageRow(message: UiMessage, onCopy: (String) -> Unit) {
    val user = message.role == ChatMessage.Role.USER
    Column(Modifier.fillMaxWidth(), horizontalAlignment = if (user) Alignment.End else Alignment.Start) {
        if (!user) Text("مدل محلی", Modifier.padding(start = 4.dp, bottom = 5.dp), style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.secondary)
        Surface(
            Modifier.widthIn(max = 760.dp),
            shape = RoundedCornerShape(topStart = 20.dp, topEnd = 20.dp, bottomStart = if (user) 20.dp else 5.dp, bottomEnd = if (user) 5.dp else 20.dp),
            color = if (user) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.surface,
            tonalElevation = if (user) 0.dp else 1.dp,
        ) {
            Column(Modifier.padding(horizontal = 16.dp, vertical = 13.dp)) {
                if (user) {
                    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                        Text(message.text.ifBlank { "…" }, Modifier.weight(1f), color = MaterialTheme.colorScheme.onPrimary, style = MaterialTheme.typography.bodyLarge)
                        if (message.text.isNotBlank()) {
                            IconButton(onClick = { onCopy(message.text) }, modifier = Modifier.size(34.dp).semantics { contentDescription = "کپی پیام" }) {
                                Icon(Icons.Default.ContentCopy, contentDescription = "کپی", tint = MaterialTheme.colorScheme.onPrimary)
                            }
                        }
                    }
                } else {
                    if (message.text.isNotBlank()) {
                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End) {
                            IconButton(onClick = { onCopy(message.text) }, modifier = Modifier.size(34.dp).semantics { contentDescription = "کپی پاسخ" }) {
                                Icon(Icons.Default.ContentCopy, contentDescription = "کپی")
                            }
                        }
                    }
                    if (message.text.isBlank()) {
                        Text("در حال آماده‌سازی پاسخ…", color = MaterialTheme.colorScheme.onSurfaceVariant, style = MaterialTheme.typography.bodyLarge)
                    } else {
                        AssistantMarkdown(message.text)
                    }
                }
            }
        }
    }
}

@Composable private fun ActionSummary(execution: ExecutionState, onWorkspace: () -> Unit, onApprove: () -> Unit, onReject: () -> Unit, approvalBusy: Boolean) {
    Surface(Modifier.fillMaxWidth(), shape = RoundedCornerShape(20.dp), color = MaterialTheme.colorScheme.tertiaryContainer) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(9.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text("اقدام Agent", style = MaterialTheme.typography.labelMedium)
                    Text(execution.action, style = MaterialTheme.typography.titleMedium)
                    Text(execution.status + " · " + durationText(execution), style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
                TextButton(onClick = onWorkspace) { Text("جزئیات") }
            }
            if (execution.approvalRequired) {
                Text("قبل از اجرای این اقدام، تأیید تو لازم است.", style = MaterialTheme.typography.bodySmall)
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(onClick = onApprove, enabled = !approvalBusy, modifier = Modifier.weight(1f).heightIn(min = 48.dp)) { Text(if (approvalBusy) "در حال اجرا…" else "تأیید") }
                    OutlinedButton(onClick = onReject, enabled = !approvalBusy, modifier = Modifier.weight(1f).heightIn(min = 48.dp)) { Text("رد") }
                }
            }
        }
    }
}

private fun durationText(execution: ExecutionState): String {
    val end = execution.finishedAt ?: System.currentTimeMillis()
    return ((end - execution.startedAt).coerceAtLeast(0L)).toString() + " ms"
}
