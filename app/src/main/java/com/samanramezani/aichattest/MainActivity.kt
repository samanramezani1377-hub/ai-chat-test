package com.samanramezani.aichattest

import android.app.Activity
import android.os.Bundle
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.compose.setContent
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.LayoutDirection
import androidx.compose.ui.unit.dp
import com.woogit.aicore.conversation.ConversationMessage
import com.woogit.aicore.conversation.ConversationRecord
import com.woogit.aicore.conversation.ConversationHistoryRepository
import com.woogit.aicore.domain.ChatMessage
import com.woogit.aicore.domain.InferenceSettings
import com.woogit.aicore.domain.ModelDescriptor
import com.woogit.aicore.domain.ModelError
import com.woogit.aicore.domain.ModelResult
import com.samanramezani.aichattest.ui.SettingsScreen
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.text.DateFormat
import java.util.Date
import java.util.UUID

private enum class Destination { CHAT, WORK, DIAGNOSTICS, SETTINGS, ABOUT }
private data class UiMessage(val role: ChatMessage.Role, val text: String, val id: String)
private data class DeletedConversation(val record: ConversationRecord)
private data class ExecutionState(
    val id: String,
    val action: String,
    val status: String,
    val startedAt: Long,
    val finishedAt: Long? = null,
    val error: String? = null,
)

class MainActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val container = (application as? AIChatApplication)?.container ?: AppContainer(this)
        setContent { AppRoot(container) }
    }
}

@Composable
private fun AppRoot(container: AppContainer) {
    var destination by remember { mutableStateOf(Destination.CHAT) }
    var sidebarOpen by remember { mutableStateOf(false) }
    var quickMenuOpen by remember { mutableStateOf(false) }
    var messages by remember { mutableStateOf(emptyList<UiMessage>()) }
    var composer by remember { mutableStateOf("") }
    var generating by remember { mutableStateOf(false) }
    var runtimeStatus by remember { mutableStateOf("خارج از دسترس") }
    var activeModel by remember { mutableStateOf<ModelDescriptor?>(null) }
    var models by remember { mutableStateOf<List<ModelDescriptor>>(emptyList()) }
    var diagnostic by remember { mutableStateOf<String?>(null) }
    var conversations by remember { mutableStateOf<List<ConversationRecord>>(emptyList()) }
    var current by remember { mutableStateOf<ConversationRecord?>(null) }
    var deleted by remember { mutableStateOf<DeletedConversation?>(null) }
    var showRename by remember { mutableStateOf(false) }
    var recentLimit by remember { mutableIntStateOf(6) }
    var execution by remember { mutableStateOf<ExecutionState?>(null) }
    val scope = rememberCoroutineScope()
    val manager = container.modelManager
    val history = container.conversationHistory

    suspend fun loadConversation(record: ConversationRecord) {
        val stored = history.messages(record.id)
        current = record
        messages = stored.map { UiMessage(if (it.role == ConversationMessage.Role.USER) ChatMessage.Role.USER else ChatMessage.Role.ASSISTANT, it.content, it.id) }
        conversations = history.recent(recentLimit)
    }

    fun refreshModels() {
        if (manager == null) return
        scope.launch(Dispatchers.Default) {
            val listed = manager.models()
            val active = manager.activeModel()
            withContext(Dispatchers.Main) {
                models = (listed as? ModelResult.Success)?.value ?: emptyList()
                activeModel = (active as? ModelResult.Success)?.value
                if (!generating) runtimeStatus = if (activeModel != null) "آماده" else "خارج از دسترس"
            }
        }
    }

    fun refreshHistory() {
        scope.launch(Dispatchers.Default) {
            val records = history.recent(recentLimit)
            withContext(Dispatchers.Main) { conversations = records }
        }
    }

    LaunchedEffect(Unit) {
        if (manager != null) withContext(Dispatchers.Default) { manager.restoreActive() }
        refreshModels()
        val records = history.recent(6)
        val record = records.firstOrNull() ?: history.create("گفت‌وگوی جدید")
        loadConversation(record)
    }

    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri == null || manager == null) return@rememberLauncherForActivityResult
        scope.launch {
            runtimeStatus = "در حال کار"
            diagnostic = null
            val result = withContext(Dispatchers.Default) { manager.import(uri) }
            when (result) {
                is ModelResult.Success -> {
                    val activation = withContext(Dispatchers.Default) { manager.activate(result.value.id) }
                    if (activation is ModelResult.Failure) diagnostic = activation.error.message
                }
                is ModelResult.Failure -> diagnostic = result.error.message
            }
            refreshModels()
        }
    }

    fun send() {
        val record = current ?: return
        if (manager == null || generating) return
        val text = composer.trim()
        if (text.isEmpty()) return
        composer = ""
        val userId = UUID.randomUUID().toString()
        val user = UiMessage(ChatMessage.Role.USER, text, userId)
        messages = messages + user
        generating = true
        runtimeStatus = "در حال تولید"
        diagnostic = null
        val runId = UUID.randomUUID().toString()
        execution = ExecutionState(runId, "تولید پاسخ", "در حال تولید", System.currentTimeMillis())
        scope.launch(Dispatchers.Default) {
            val now = System.currentTimeMillis()
            history.append(record.id, ConversationMessage(userId, ConversationMessage.Role.USER, text, now), now)
            val request = messages.takeLast(24).map { ChatMessage(it.role, it.text) }
            val result = try {
                manager.generate(request, InferenceSettings(maxNewTokens = 512))
            } catch (t: Throwable) {
                ModelResult.Failure(ModelError.Inference("Local generation failed", t))
            }
            withContext(Dispatchers.Main) {
                generating = false
                when (result) {
                    is ModelResult.Success -> {
                        val assistantId = UUID.randomUUID().toString()
                        val answer = UiMessage(ChatMessage.Role.ASSISTANT, result.value.text, assistantId)
                        messages = messages + answer
                        scope.launch(Dispatchers.Default) {
                            val timestamp = System.currentTimeMillis()
                            history.append(record.id, ConversationMessage(assistantId, ConversationMessage.Role.ASSISTANT, answer.text, timestamp), timestamp)
                            val updated = history.recent(recentLimit).firstOrNull { it.id == record.id }
                            withContext(Dispatchers.Main) {
                                if (updated != null) current = updated
                                conversations = history.recent(recentLimit)
                            }
                        }
                        runtimeStatus = "آماده"
                        execution = execution?.copy(status = "انجام شد", finishedAt = System.currentTimeMillis())
                    }
                    is ModelResult.Failure -> {
                        runtimeStatus = "خطا"
                        diagnostic = result.error.message
                        execution = execution?.copy(status = "ناموفق", finishedAt = System.currentTimeMillis(), error = result.error.message)
                    }
                }
            }
        }
    }

    fun stop() {
        if (!generating) return
        scope.launch(Dispatchers.Default) { manager?.stopGeneration() }
        generating = false
        runtimeStatus = if (activeModel != null) "آماده" else "خارج از دسترس"
        execution = execution?.copy(status = "متوقف شد", finishedAt = System.currentTimeMillis())
    }

    fun createConversation() {
        scope.launch(Dispatchers.Default) {
            val record = history.create("گفت‌وگوی جدید")
            withContext(Dispatchers.Main) {
                current = record
                messages = emptyList()
                composer = ""
                conversations = history.recent(recentLimit)
                destination = Destination.CHAT
                sidebarOpen = false
            }
        }
    }

    fun openConversation(record: ConversationRecord) {
        scope.launch(Dispatchers.Default) {
            loadConversation(record)
            withContext(Dispatchers.Main) { destination = Destination.CHAT; sidebarOpen = false }
        }
    }

    fun deleteConversation(record: ConversationRecord) {
        scope.launch(Dispatchers.Default) {
            val removed = history.delete(record.id) ?: return@launch
            withContext(Dispatchers.Main) {
                if (current?.id == removed.id) {
                    val replacement = history.recent(1).firstOrNull()
                    if (replacement != null) {
                        scope.launch(Dispatchers.Default) { loadConversation(replacement) }
                    } else {
                        current = null
                        messages = emptyList()
                    }
                }
                conversations = history.recent(recentLimit)
                deleted = DeletedConversation(removed)
            }
        }
    }

    CompositionLocalProvider(LocalLayoutDirection provides LayoutDirection.Rtl) {
        MaterialTheme(colorScheme = lightColorScheme()) {
            Surface(Modifier.fillMaxSize()) {
                Box(Modifier.fillMaxSize()) {
                    Column(Modifier.fillMaxSize()) {
                        AppHeader(destination, runtimeStatus, activeModel, { destination = it; quickMenuOpen = false }, { quickMenuOpen = !quickMenuOpen }, { sidebarOpen = true })
                        Box(Modifier.weight(1f).fillMaxWidth()) {
                            when (destination) {
                                Destination.CHAT -> ChatPage(messages, composer, generating, { composer = it }, ::send, ::stop)
                                Destination.WORK -> WorkspacePage(execution, activeModel)
                                Destination.DIAGNOSTICS -> DiagnosticsPage(diagnostic, execution)
                                Destination.SETTINGS -> SettingsScreen(models, activeModel, diagnostic, { picker.launch(arrayOf("application/octet-stream", "application/gzip", "*/*")) }, ::refreshModels, { id -> scope.launch(Dispatchers.Default) { manager?.activate(id); refreshModels() } }, { id -> scope.launch(Dispatchers.Default) { manager?.delete(id); refreshModels() } })
                                Destination.ABOUT -> AboutPage()
                            }
                        }
                    }
                    if (quickMenuOpen) QuickMenu({ destination = Destination.DIAGNOSTICS; quickMenuOpen = false }, { destination = Destination.SETTINGS; quickMenuOpen = false })
                    if (sidebarOpen) Sidebar(
                        current = current,
                        conversations = conversations,
                        destination = destination,
                        onClose = { sidebarOpen = false },
                        onDestination = { destination = it; sidebarOpen = false },
                        onNew = ::createConversation,
                        onOpen = ::openConversation,
                        onRename = { showRename = true },
                        onDelete = ::deleteConversation,
                        onMore = { recentLimit += 5; refreshHistory() },
                        canShowMore = conversations.size >= recentLimit,
                    )
                    deleted?.let { item ->
                        UndoBar(
                            onUndo = {
                                scope.launch(Dispatchers.Default) {
                                    if (history.restore(item.record)) {
                                        loadConversation(item.record)
                                        withContext(Dispatchers.Main) { deleted = null; conversations = history.recent(recentLimit) }
                                    }
                                }
                            },
                            onExpire = {
                                scope.launch(Dispatchers.Default) { history.purge(item.record.id) }
                                deleted = null
                            },
                        )
                    }
                    if (showRename) {
                        RenameDialog(current?.title.orEmpty(), onDismiss = { showRename = false }) { title ->
                            val id = current?.id ?: return@RenameDialog
                            scope.launch(Dispatchers.Default) {
                                history.rename(id, title, System.currentTimeMillis())
                                val updated = history.recent(recentLimit).firstOrNull { it.id == id }
                                withContext(Dispatchers.Main) { if (updated != null) current = updated; conversations = history.recent(recentLimit); showRename = false }
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable private fun AppHeader(destination: Destination, status: String, model: ModelDescriptor?, onDestination: (Destination) -> Unit, onQuick: () -> Unit, onSidebar: () -> Unit) {
    Surface(Modifier.fillMaxWidth().padding(10.dp), shape = MaterialTheme.shapes.large, tonalElevation = 2.dp, shadowElevation = 1.dp) {
        Row(Modifier.fillMaxWidth().heightIn(min = 64.dp).padding(6.dp), verticalAlignment = Alignment.CenterVertically) {
            Row(horizontalArrangement = Arrangement.spacedBy(2.dp)) {
                TextButton(onClick = { onDestination(Destination.CHAT) }) { Text(if (destination == Destination.CHAT) "گفت‌وگو" else "گفت‌وگو") }
                TextButton(onClick = { onDestination(Destination.WORK) }) { Text("کار") }
                IconButton(onClick = onQuick) { Icon(Icons.Default.MoreVert, "منوی سریع") }
            }
            Spacer(Modifier.weight(1f))
            Column(horizontalAlignment = Alignment.CenterHorizontally) { Text("● $status", style = MaterialTheme.typography.labelLarge); Text(model?.displayName ?: "مدل محلی", style = MaterialTheme.typography.labelSmall) }
            Spacer(Modifier.weight(1f))
            IconButton(onClick = onSidebar, modifier = Modifier.size(48.dp).semantics { contentDescription = "باز کردن نوار کناری" }) { Icon(Icons.Default.Menu, "باز کردن نوار کناری") }
        }
    }
}

@Composable private fun QuickMenu(onDiagnostics: () -> Unit, onSettings: () -> Unit) {
    Surface(Modifier.padding(top = 76.dp, start = 12.dp).width(190.dp), shape = MaterialTheme.shapes.large, tonalElevation = 6.dp) { Column(Modifier.padding(8.dp)) { TextButton(onClick = onDiagnostics, Modifier.fillMaxWidth()) { Text("لاگ") }; TextButton(onClick = onSettings, Modifier.fillMaxWidth()) { Text("تنظیمات") } } }
}

@Composable private fun Sidebar(current: ConversationRecord?, conversations: List<ConversationRecord>, destination: Destination, onClose: () -> Unit, onDestination: (Destination) -> Unit, onNew: () -> Unit, onOpen: (ConversationRecord) -> Unit, onRename: () -> Unit, onDelete: (ConversationRecord) -> Unit, onMore: () -> Unit, canShowMore: Boolean) {
    Box(Modifier.fillMaxSize()) {
        Spacer(Modifier.fillMaxSize().clickable { onClose() })
        Surface(Modifier.fillMaxHeight().fillMaxWidth(.86f).align(Alignment.CenterEnd), shape = MaterialTheme.shapes.extraLarge, tonalElevation = 7.dp) {
            LazyColumn(Modifier.fillMaxSize().padding(18.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                item { Button(onClick = onNew, Modifier.fillMaxWidth()) { Icon(Icons.Default.Add, null); Spacer(Modifier.width(6.dp)); Text("گفت‌وگوی جدید") } }
                item { Text("ابزار", style = MaterialTheme.typography.labelLarge, modifier = Modifier.padding(top = 16.dp)) }
                item { SideNav("فضای کار", destination == Destination.WORK) { onDestination(Destination.WORK) } }
                item { SideNav("عیب‌یابی", destination == Destination.DIAGNOSTICS) { onDestination(Destination.DIAGNOSTICS) } }
                item { SideNav("تنظیمات", destination == Destination.SETTINGS) { onDestination(Destination.SETTINGS) } }
                item { Text("اطلاعات", style = MaterialTheme.typography.labelLarge, modifier = Modifier.padding(top = 16.dp)) }
                item { SideNav("درباره برنامه", destination == Destination.ABOUT) { onDestination(Destination.ABOUT) } }
                item { Text("اخیر", style = MaterialTheme.typography.labelLarge, modifier = Modifier.padding(top = 16.dp)) }
                if (conversations.isEmpty()) item { Text("گفت‌وگوی ذخیره‌شده‌ای وجود ندارد.", color = MaterialTheme.colorScheme.onSurfaceVariant) }
                items(conversations, key = { it.id }) { record ->
                    RecentConversation(record, record.id == current?.id, { onOpen(record) }, onRename, { onDelete(record) })
                }
                if (canShowMore) item { TextButton(onClick = onMore, Modifier.fillMaxWidth()) { Text("مشاهده بیشتر") } }
                item { TextButton(onClick = onClose, Modifier.fillMaxWidth()) { Text("بستن") } }
            }
        }
    }
}

@Composable private fun SideNav(text: String, active: Boolean, onClick: () -> Unit) { TextButton(onClick = onClick, Modifier.fillMaxWidth()) { Text(if (active) "● $text" else text, Modifier.fillMaxWidth()) } }

@Composable private fun RecentConversation(record: ConversationRecord, active: Boolean, onOpen: () -> Unit, onRename: () -> Unit, onDelete: () -> Unit) {
    var menu by remember { mutableStateOf(false) }
    val time = remember(record.updatedAtEpochMs) { DateFormat.getTimeInstance(DateFormat.SHORT).format(Date(record.updatedAtEpochMs)) }
    Column {
        Surface(Modifier.fillMaxWidth().clickable(onClick = onOpen), shape = MaterialTheme.shapes.medium, tonalElevation = if (active) 2.dp else 0.dp) {
            Row(Modifier.padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
                Text("●", color = if (active) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.outline)
                Spacer(Modifier.width(8.dp)); Text(record.title, Modifier.weight(1f)); Text(time, style = MaterialTheme.typography.labelSmall)
            }
        }
        TextButton(onClick = { menu = !menu }, Modifier.align(Alignment.End)) { Text("گزینه‌ها") }
        if (menu) Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End) {
            TextButton(onClick = { menu = false; onOpen() }) { Text("باز کردن") }
            TextButton(onClick = { menu = false; onRename() }) { Text("تغییر نام") }
            TextButton(onClick = { menu = false; onDelete() }, colors = ButtonDefaults.textButtonColors(contentColor = MaterialTheme.colorScheme.error)) { Text("حذف") }
        }
    }
}

@Composable private fun UndoBar(onUndo: () -> Unit, onExpire: () -> Unit) {
    var remaining by remember { mutableIntStateOf(10) }
    LaunchedEffect(Unit) { while (remaining > 0) { delay(1000); remaining-- }; onExpire() }
    Surface(Modifier.fillMaxWidth().padding(12.dp), shape = MaterialTheme.shapes.large, tonalElevation = 6.dp) {
        Row(Modifier.padding(10.dp), verticalAlignment = Alignment.CenterVertically) { Text("گفت‌وگو حذف شد", Modifier.weight(1f)); TextButton(onClick = onUndo) { Text("بازگردانی") }; Text("$remaining") }
    }
}

@Composable private fun RenameDialog(initial: String, onDismiss: () -> Unit, onConfirm: (String) -> Unit) {
    var value by remember(initial) { mutableStateOf(initial) }
    AlertDialog(onDismissRequest = onDismiss, title = { Text("تغییر نام گفت‌وگو") }, text = { TextField(value, { value = it }, singleLine = true, label = { Text("عنوان") }) }, confirmButton = { TextButton(onClick = { onConfirm(value) }, enabled = value.isNotBlank()) { Text("ذخیره") } }, dismissButton = { TextButton(onClick = onDismiss) { Text("انصراف") } })
}

@Composable private fun ChatPage(messages: List<UiMessage>, composer: String, generating: Boolean, onComposer: (String) -> Unit, onSend: () -> Unit, onStop: () -> Unit) {
    Column(Modifier.fillMaxSize().padding(horizontal = 16.dp)) {
        LazyColumn(Modifier.weight(1f).fillMaxWidth(), contentPadding = PaddingValues(vertical = 20.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
            if (messages.isEmpty()) item { Column(Modifier.fillMaxWidth().padding(top = 70.dp), horizontalAlignment = Alignment.CenterHorizontally) { Text("گفت‌وگو", style = MaterialTheme.typography.headlineMedium); Text("پیام خود را بنویسید و گفتگو را شروع کنید.", color = MaterialTheme.colorScheme.onSurfaceVariant) } }
            items(messages, key = { it.id }) { MessageRow(it) }
            if (generating) item { Text("در حال تولید پاسخ…", color = MaterialTheme.colorScheme.onSurfaceVariant) }
        }
        Surface(Modifier.fillMaxWidth().padding(bottom = 12.dp), shape = MaterialTheme.shapes.extraLarge, tonalElevation = 3.dp) {
            Row(Modifier.padding(8.dp), verticalAlignment = Alignment.Bottom) {
                TextField(composer, onComposer, Modifier.weight(1f), placeholder = { Text("پیام خود را بنویسید…") }, maxLines = 6)
                Spacer(Modifier.width(8.dp)); FilledIconButton(onClick = if (generating) onStop else onSend, enabled = generating || composer.isNotBlank()) { Icon(if (generating) Icons.Default.Stop else Icons.Default.Send, null) }
            }
        }
    }
}

@Composable private fun MessageRow(message: UiMessage) {
    val user = message.role == ChatMessage.Role.USER
    Column(Modifier.fillMaxWidth(), horizontalAlignment = if (user) Alignment.Start else Alignment.End) {
        Text(if (user) "شما" else "مدل", style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        if (user) Surface(color = MaterialTheme.colorScheme.primaryContainer.copy(alpha = .70f), shape = MaterialTheme.shapes.large) { Text(message.text, Modifier.padding(13.dp)) } else Text(message.text, Modifier.fillMaxWidth(.94f))
    }
}

@Composable private fun WorkspacePage(execution: ExecutionState?, model: ModelDescriptor?) {
    SimplePage("فضای کار") {
        Text("خلاصه اجرای جاری", style = MaterialTheme.typography.titleLarge)
        if (execution == null) Text("اجرای فعالی ثبت نشده است.", color = MaterialTheme.colorScheme.onSurfaceVariant)
        else {
            Text(execution.action, style = MaterialTheme.typography.titleMedium)
            Text(execution.status)
            Text("شناسه Execution: ${execution.id}", style = MaterialTheme.typography.bodySmall)
            Text("مدل: ${model?.displayName ?: "مدل محلی"}", style = MaterialTheme.typography.bodySmall)
        }
        Text("Timeline مرکزی", style = MaterialTheme.typography.titleLarge)
        if (execution == null) Text("با اجرای یک عملیات واقعی، رویداد همان Execution اینجا ثبت می‌شود.", color = MaterialTheme.colorScheme.onSurfaceVariant)
        else {
            Text("● ${execution.action}")
            Text("${execution.status} · ${durationText(execution)}", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            execution.error?.let { Text(it, color = MaterialTheme.colorScheme.error) }
        }
        Text("تاریخچه عملکرد", style = MaterialTheme.typography.titleLarge)
        Text("آمار فقط پس از دریافت داده واقعی Runtime نمایش داده می‌شود.", color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

private fun durationText(execution: ExecutionState): String {
    val end = execution.finishedAt ?: System.currentTimeMillis()
    return "${(end - execution.startedAt).coerceAtLeast(0L)} ms"
}

@Composable private fun DiagnosticsPage(error: String?, execution: ExecutionState?) {
    SimplePage("عیب‌یابی") {
        Text("خطاها", style = MaterialTheme.typography.titleLarge)
        if (error == null) Text("خطایی برای نمایش ثبت نشده است.", color = MaterialTheme.colorScheme.onSurfaceVariant) else Text(error, color = MaterialTheme.colorScheme.error)
        Text("گزارش Execution / Trace", style = MaterialTheme.typography.titleLarge)
        if (execution == null) Text("Execution ثبت‌شده‌ای وجود ندارد.", color = MaterialTheme.colorScheme.onSurfaceVariant)
        else {
            Text("Execution: ${execution.id}", style = MaterialTheme.typography.bodySmall)
            Text("Action: ${execution.action}")
            Text("وضعیت: ${execution.status}")
            Text("شروع: ${DateFormat.getDateTimeInstance().format(Date(execution.startedAt))}", style = MaterialTheme.typography.bodySmall)
            execution.finishedAt?.let { Text("پایان: ${DateFormat.getDateTimeInstance().format(Date(it))}", style = MaterialTheme.typography.bodySmall) }
            execution.error?.let { Text("Error: $it", color = MaterialTheme.colorScheme.error) }
        }
        Text("عملکرد", style = MaterialTheme.typography.titleLarge)
        Text("هیچ معیار ساختگی نمایش داده نمی‌شود؛ داده عملکرد پس از اتصال واقعی Runtime ثبت خواهد شد.", color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text("گزارش کار حساس", style = MaterialTheme.typography.titleLarge)
        Text("گزارش حساسیتی برای این Execution ثبت نشده است.", color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@Composable private fun AboutPage() { SimplePage("درباره برنامه") { Text("نسخه و مشخصات برنامه", style = MaterialTheme.typography.titleLarge); Text("رابط کاربری فارسی و RTL بر اساس قرارداد UI v1.0.0.", color = MaterialTheme.colorScheme.onSurfaceVariant) } }

@Composable private fun SimplePage(title: String, content: @Composable ColumnScope.() -> Unit) { LazyColumn(Modifier.fillMaxSize().padding(20.dp), contentPadding = PaddingValues(bottom = 32.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) { item { Text(title, style = MaterialTheme.typography.headlineMedium) }; item { Column(verticalArrangement = Arrangement.spacedBy(10.dp), content = content) } } }
