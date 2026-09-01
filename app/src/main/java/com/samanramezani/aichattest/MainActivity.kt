package com.samanramezani.aichattest

import android.app.Activity
import android.os.Bundle
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.input.key.Key
import androidx.compose.ui.input.key.KeyEventType
import androidx.compose.ui.input.key.isShiftPressed
import androidx.compose.ui.input.key.onPreviewKeyEvent
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.LayoutDirection
import androidx.compose.ui.unit.dp
import androidx.compose.ui.graphics.Color
import com.woogit.aicore.conversation.ConversationMessage
import com.woogit.aicore.conversation.ConversationRecord
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
    val requestPreview: String = "",
    val resultPreview: String? = null,
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
    var runtimeDetailsOpen by remember { mutableStateOf(false) }
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
    var renameTarget by remember { mutableStateOf<ConversationRecord?>(null) }
    var recentLimit by remember { mutableIntStateOf(5) }
    var execution by remember { mutableStateOf<ExecutionState?>(null) }
    val scope = rememberCoroutineScope()
    val manager = container.modelManager
    val history = container.conversationHistory

    suspend fun loadConversation(record: ConversationRecord) {
        val stored = history.messages(record.id)
        current = record
        messages = stored.map {
            UiMessage(
                if (it.role == ConversationMessage.Role.USER) ChatMessage.Role.USER else ChatMessage.Role.ASSISTANT,
                it.content,
                it.id,
            )
        }
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
        val records = history.recent(recentLimit)
        val record = records.firstOrNull() ?: history.create("گفت‌وگوی جدید")
        loadConversation(record)
    }

    BackHandler(enabled = sidebarOpen || quickMenuOpen) {
        if (sidebarOpen) sidebarOpen = false else quickMenuOpen = false
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
        messages = messages + UiMessage(ChatMessage.Role.USER, text, userId)
        generating = true
        runtimeStatus = "در حال تولید"
        diagnostic = null
        execution = ExecutionState(UUID.randomUUID().toString(), "تولید پاسخ", "در حال تولید", System.currentTimeMillis(), requestPreview = text)
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
                        val finished = System.currentTimeMillis()
                        execution = execution?.copy(status = "انجام شد", finishedAt = finished, resultPreview = answer.text.take(240))
                        scope.launch(Dispatchers.Default) {
                            history.append(record.id, ConversationMessage(assistantId, ConversationMessage.Role.ASSISTANT, answer.text, finished), finished)
                            val updated = history.recent(recentLimit).firstOrNull { it.id == record.id }
                            withContext(Dispatchers.Main) {
                                if (updated != null) current = updated
                                conversations = history.recent(recentLimit)
                            }
                        }
                        runtimeStatus = "آماده"
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

    fun deactivateModel() {
        if (manager == null || activeModel == null) return
        scope.launch(Dispatchers.Default) {
            val result = manager.deactivate()
            withContext(Dispatchers.Main) {
                when (result) {
                    is ModelResult.Success -> {
                        activeModel = null
                        runtimeStatus = "خارج از دسترس"
                        diagnostic = null
                        refreshModels()
                    }
                    is ModelResult.Failure -> {
                        runtimeStatus = "خطا"
                        diagnostic = result.error.message
                    }
                }
            }
        }
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
            val replacement = if (current?.id == removed.id) history.recent(1).firstOrNull() else null
            withContext(Dispatchers.Main) {
                if (replacement != null) {
                    scope.launch(Dispatchers.Default) { loadConversation(replacement) }
                } else if (current?.id == removed.id) {
                    current = null
                    messages = emptyList()
                }
                conversations = history.recent(recentLimit)
                deleted = DeletedConversation(removed)
            }
        }
    }

    fun restoreDeleted(record: ConversationRecord) {
        scope.launch(Dispatchers.Default) {
            if (history.restore(record)) {
                loadConversation(record)
                withContext(Dispatchers.Main) {
                    conversations = history.recent(recentLimit)
                    deleted = null
                }
            }
        }
    }

    CompositionLocalProvider(LocalLayoutDirection provides LayoutDirection.Rtl) {
        AppTheme {
            Surface(Modifier.fillMaxSize()) {
                Box(Modifier.fillMaxSize()) {
                    Column(Modifier.fillMaxSize()) {
                        AppHeader(
                            destination = destination,
                            status = runtimeStatus,
                            model = activeModel,
                            onDestination = { destination = it; quickMenuOpen = false },
                            onQuick = { quickMenuOpen = !quickMenuOpen },
                            onSidebar = { quickMenuOpen = false; sidebarOpen = true },
                            onRuntime = { runtimeDetailsOpen = true },
                        )
                        Box(Modifier.weight(1f).fillMaxWidth()) {
                            when (destination) {
                                Destination.CHAT -> ChatPage(messages, composer, generating, { composer = it }, ::send, ::stop, execution) { destination = Destination.WORK }
                                Destination.WORK -> WorkspacePage(execution, activeModel) { destination = Destination.DIAGNOSTICS }
                                Destination.DIAGNOSTICS -> DiagnosticsPage(diagnostic, execution)
                                Destination.SETTINGS -> SettingsScreen(
                                    models, activeModel, diagnostic,
                                    onImport = { picker.launch(arrayOf("application/octet-stream", "application/gzip", "*/*")) },
                                    onRefresh = ::refreshModels,
                                    onActivate = { id -> scope.launch(Dispatchers.Default) { manager?.activate(id); refreshModels() } },
                                    onDeactivate = ::deactivateModel,
                                    onDelete = { id -> scope.launch(Dispatchers.Default) { manager?.delete(id); refreshModels() } },
                                )
                                Destination.ABOUT -> AboutPage()
                            }
                        }
                    }
                    if (quickMenuOpen) QuickMenu(
                        onDiagnostics = { destination = Destination.DIAGNOSTICS; quickMenuOpen = false },
                        onSettings = { destination = Destination.SETTINGS; quickMenuOpen = false },
                    )
                    if (sidebarOpen) Sidebar(
                        current, conversations, destination,
                        onClose = { sidebarOpen = false },
                        onDestination = { destination = it; sidebarOpen = false },
                        onNew = ::createConversation,
                        onOpen = ::openConversation,
                        onRename = { renameTarget = it },
                        onDelete = ::deleteConversation,
                        onMore = { recentLimit += 5; refreshHistory() },
                        canShowMore = conversations.size >= recentLimit,
                    )
                    deleted?.let { item ->
                        UndoBar(
                            onUndo = { restoreDeleted(item.record) },
                            onDismiss = {
                                scope.launch(Dispatchers.Default) { history.purge(item.record.id) }
                                deleted = null
                            },
                            onExpire = {
                                scope.launch(Dispatchers.Default) { history.purge(item.record.id) }
                                deleted = null
                            },
                        )
                    }
                }
            }
        }
    }

    if (renameTarget != null) {
        val target = renameTarget!!
        RenameDialog(target.title, { renameTarget = null }) { title ->
            scope.launch(Dispatchers.Default) {
                history.rename(target.id, title, System.currentTimeMillis())
                val updated = history.recent(recentLimit).firstOrNull { it.id == target.id }
                withContext(Dispatchers.Main) {
                    if (updated?.id == current?.id) current = updated
                    conversations = history.recent(recentLimit)
                    renameTarget = null
                }
            }
        }
    }

    if (runtimeDetailsOpen) {
        RuntimeDetailsDialog(
            status = runtimeStatus,
            model = activeModel,
            runtimeName = container.modelRuntime.runtimeInfo().name,
            runtimeVersion = container.modelRuntime.runtimeInfo().version,
            backend = container.modelRuntime.runtimeInfo().backend,
            onDismiss = { runtimeDetailsOpen = false },
        )
    }
}

@Composable
private fun AppTheme(content: @Composable () -> Unit) {
    val colors = lightColorScheme(
        primary = Color(0xFF376A9A), onPrimary = Color.White,
        primaryContainer = Color(0xFFDCEEFF), onPrimaryContainer = Color(0xFF102A43),
        background = Color(0xFFF7F9FC), surface = Color(0xFFFDFEFF), surfaceVariant = Color(0xFFEAF0F6),
        onSurface = Color(0xFF18212B), onSurfaceVariant = Color(0xFF5E6B78),
    )
    MaterialTheme(colorScheme = colors, shapes = Shapes(large = RoundedCornerShape(20.dp), extraLarge = RoundedCornerShape(28.dp)), content = content)
}

@Composable
private fun AppHeader(destination: Destination, status: String, model: ModelDescriptor?, onDestination: (Destination) -> Unit, onQuick: () -> Unit, onSidebar: () -> Unit, onRuntime: () -> Unit) {
    Surface(Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 10.dp), shape = RoundedCornerShape(22.dp), color = MaterialTheme.colorScheme.surface.copy(alpha = .92f), tonalElevation = 1.dp) {
        Row(Modifier.fillMaxWidth().heightIn(min = 68.dp).padding(6.dp), verticalAlignment = Alignment.CenterVertically) {
            Row(horizontalArrangement = Arrangement.spacedBy(2.dp)) {
                HeaderNav("گفت‌وگو", destination == Destination.CHAT) { onDestination(Destination.CHAT) }
                HeaderNav("کار", destination == Destination.WORK) { onDestination(Destination.WORK) }
                IconButton(onClick = onQuick, modifier = Modifier.size(48.dp).semantics { contentDescription = "منوی سریع" }) { Icon(Icons.Default.MoreVert, "منوی سریع") }
            }
            Spacer(Modifier.weight(1f))
            TextButton(onClick = onRuntime, modifier = Modifier.heightIn(min = 48.dp).semantics { contentDescription = "جزئیات وضعیت Runtime" }) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text("● $status", style = MaterialTheme.typography.labelLarge)
                    Text(model?.displayName ?: "مدل محلی", style = MaterialTheme.typography.labelSmall)
                }
            }
            Spacer(Modifier.weight(1f))
            IconButton(onClick = onSidebar, modifier = Modifier.size(48.dp).semantics { contentDescription = "باز کردن نوار کناری" }) { Icon(Icons.Default.Menu, "باز کردن نوار کناری") }
        }
    }
}

@Composable private fun HeaderNav(text: String, active: Boolean, onClick: () -> Unit) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        TextButton(onClick = onClick, modifier = Modifier.heightIn(min = 48.dp)) { Text(text) }
        if (active) Box(Modifier.width(34.dp).height(2.dp).background(MaterialTheme.colorScheme.primary))
    }
}

@Composable private fun QuickMenu(onDiagnostics: () -> Unit, onSettings: () -> Unit) {
    Surface(Modifier.padding(top = 82.dp, start = 12.dp).widthIn(max = 220.dp), shape = RoundedCornerShape(18.dp), tonalElevation = 6.dp) {
        Column(Modifier.padding(8.dp)) {
            TextButton(onClick = onDiagnostics, Modifier.fillMaxWidth().heightIn(min = 48.dp)) { Icon(Icons.Default.BugReport, null); Spacer(Modifier.width(8.dp)); Text("عیب‌یابی") }
            TextButton(onClick = onSettings, Modifier.fillMaxWidth().heightIn(min = 48.dp)) { Icon(Icons.Default.Settings, null); Spacer(Modifier.width(8.dp)); Text("تنظیمات") }
        }
    }
}

@Composable
private fun Sidebar(current: ConversationRecord?, conversations: List<ConversationRecord>, destination: Destination, onClose: () -> Unit, onDestination: (Destination) -> Unit, onNew: () -> Unit, onOpen: (ConversationRecord) -> Unit, onRename: (ConversationRecord) -> Unit, onDelete: (ConversationRecord) -> Unit, onMore: () -> Unit, canShowMore: Boolean) {
    Box(Modifier.fillMaxSize()) {
        Spacer(Modifier.fillMaxSize().background(Color.Black.copy(alpha = .24f)).clickable(onClick = onClose).semantics { contentDescription = "بستن نوار کناری" })
        Surface(Modifier.fillMaxHeight().fillMaxWidth(.88f).align(Alignment.CenterEnd), shape = RoundedCornerShape(topStart = 28.dp, bottomStart = 28.dp), tonalElevation = 8.dp) {
            LazyColumn(Modifier.fillMaxSize().padding(18.dp), verticalArrangement = Arrangement.spacedBy(5.dp)) {
                item { Text("منو", style = MaterialTheme.typography.headlineSmall); Spacer(Modifier.height(8.dp)) }
                item { Button(onClick = onNew, Modifier.fillMaxWidth().heightIn(min = 48.dp)) { Icon(Icons.Default.Add, null); Spacer(Modifier.width(6.dp)); Text("گفت‌وگوی جدید") } }
                item { SectionLabel("ابزارها") }
                item { SideNav("فضای کار", destination == Destination.WORK) { onDestination(Destination.WORK) } }
                item { SideNav("عیب‌یابی", destination == Destination.DIAGNOSTICS) { onDestination(Destination.DIAGNOSTICS) } }
                item { SideNav("تنظیمات", destination == Destination.SETTINGS) { onDestination(Destination.SETTINGS) } }
                item { SectionLabel("اطلاعات") }
                item { SideNav("درباره برنامه", destination == Destination.ABOUT) { onDestination(Destination.ABOUT) } }
                item { SectionLabel("اخیر") }
                if (conversations.isEmpty()) item { Text("گفت‌وگوی ذخیره‌شده‌ای وجود ندارد.", color = MaterialTheme.colorScheme.onSurfaceVariant) }
                items(conversations, key = { it.id }) { record ->
                    RecentConversation(record, record.id == current?.id, onOpen = { onOpen(record) }, onRename = { onRename(record) }, onDelete = { onDelete(record) })
                }
                if (canShowMore) item { TextButton(onClick = onMore, Modifier.fillMaxWidth().heightIn(min = 48.dp)) { Text("مشاهده بیشتر") } }
                item { OutlinedButton(onClick = onClose, Modifier.fillMaxWidth().heightIn(min = 48.dp)) { Text("بستن") } }
            }
        }
    }
}

@Composable private fun SectionLabel(text: String) { Text(text, style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(top = 14.dp, bottom = 3.dp)) }

@Composable private fun SideNav(text: String, active: Boolean, onClick: () -> Unit) {
    TextButton(onClick = onClick, Modifier.fillMaxWidth().heightIn(min = 48.dp)) { Text(if (active) "● $text" else text, Modifier.fillMaxWidth()) }
}

@Composable
private fun RecentConversation(record: ConversationRecord, active: Boolean, onOpen: () -> Unit, onRename: () -> Unit, onDelete: () -> Unit) {
    var menu by remember { mutableStateOf(false) }
    val time = remember(record.updatedAtEpochMs) { DateFormat.getTimeInstance(DateFormat.SHORT).format(Date(record.updatedAtEpochMs)) }
    Box(Modifier.fillMaxWidth()) {
        Surface(
            Modifier.fillMaxWidth().combinedClickable(onClick = onOpen, onLongClick = { menu = true }).semantics { contentDescription = "گفت‌وگو ${record.title}. برای گزینه‌ها لمس طولانی کنید." },
            shape = RoundedCornerShape(16.dp),
            color = if (active) MaterialTheme.colorScheme.primaryContainer.copy(alpha = .55f) else Color.Transparent,
        ) {
            Row(Modifier.padding(12.dp).heightIn(min = 48.dp), verticalAlignment = Alignment.CenterVertically) {
                Text("●", color = if (active) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.outline)
                Spacer(Modifier.width(8.dp))
                Column(Modifier.weight(1f)) {
                    Text(record.title, maxLines = 1)
                    Text("${record.messageCount} پیام", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
                Text(time, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
        TextButton(onClick = { menu = !menu }, Modifier.align(Alignment.TopEnd).heightIn(min = 48.dp).semantics { contentDescription = "گزینه‌های ${record.title}" }) { Text("گزینه‌ها") }
        DropdownMenu(expanded = menu, onDismissRequest = { menu = false }) {
            DropdownMenuItem(text = { Text("باز کردن") }, onClick = { menu = false; onOpen() })
            DropdownMenuItem(text = { Text("تغییر نام") }, onClick = { menu = false; onRename() })
            DropdownMenuItem(text = { Text("حذف", color = MaterialTheme.colorScheme.error) }, onClick = { menu = false; onDelete() })
        }
    }
}

@Composable
private fun UndoBar(onUndo: () -> Unit, onDismiss: () -> Unit, onExpire: () -> Unit) {
    var remaining by remember { mutableIntStateOf(10) }
    LaunchedEffect(Unit) {
        while (remaining > 0) { delay(1000); remaining-- }
        onExpire()
    }
    Box(Modifier.fillMaxSize().padding(12.dp), contentAlignment = Alignment.BottomCenter) {
        Surface(Modifier.fillMaxWidth().widthIn(max = 620.dp), shape = RoundedCornerShape(18.dp), tonalElevation = 6.dp) {
            Row(Modifier.padding(8.dp), verticalAlignment = Alignment.CenterVertically) {
                Text("گفت‌وگو حذف شد", Modifier.weight(1f))
                TextButton(onClick = onUndo, Modifier.heightIn(min = 48.dp)) { Text("بازگردانی") }
                Text("$remaining ثانیه", style = MaterialTheme.typography.labelMedium)
                IconButton(onClick = onDismiss, modifier = Modifier.size(48.dp).semantics { contentDescription = "بستن پیام حذف" }) { Icon(Icons.Default.Close, "بستن") }
            }
        }
    }
}

@Composable private fun RenameDialog(initial: String, onDismiss: () -> Unit, onConfirm: (String) -> Unit) {
    var value by remember(initial) { mutableStateOf(initial) }
    AlertDialog(onDismissRequest = onDismiss, title = { Text("تغییر نام گفت‌وگو") }, text = { TextField(value, { value = it }, singleLine = true, label = { Text("عنوان") }) }, confirmButton = { TextButton(onClick = { onConfirm(value.trim()) }, enabled = value.isNotBlank(), modifier = Modifier.heightIn(min = 48.dp)) { Text("ذخیره") } }, dismissButton = { TextButton(onClick = onDismiss, modifier = Modifier.heightIn(min = 48.dp)) { Text("انصراف") } })
}

@Composable
private fun RuntimeDetailsDialog(status: String, model: ModelDescriptor?, runtimeName: String, runtimeVersion: String, backend: String?, onDismiss: () -> Unit) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("وضعیت Runtime") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("وضعیت: $status")
                Text("مدل: ${model?.displayName ?: "مدلی فعال نیست"}")
                Text("Runtime: $runtimeName")
                Text("نسخه: $runtimeVersion")
                Text("Backend: ${backend ?: "ارائه نشده"}")
            }
        },
        confirmButton = { TextButton(onClick = onDismiss, modifier = Modifier.heightIn(min = 48.dp)) { Text("بستن") } },
    )
}

@Composable private fun ChatPage(messages: List<UiMessage>, composer: String, generating: Boolean, onComposer: (String) -> Unit, onSend: () -> Unit, onStop: () -> Unit, execution: ExecutionState?, onWorkspace: () -> Unit) {
    Column(Modifier.fillMaxSize().padding(horizontal = 16.dp)) {
        LazyColumn(Modifier.weight(1f).fillMaxWidth(), contentPadding = PaddingValues(vertical = 20.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
            if (messages.isEmpty()) item { Column(Modifier.fillMaxWidth().padding(top = 64.dp), horizontalAlignment = Alignment.CenterHorizontally) { Text("گفت‌وگو", style = MaterialTheme.typography.headlineMedium); Spacer(Modifier.height(6.dp)); Text("پیام خود را بنویسید و گفتگو را شروع کنید.", color = MaterialTheme.colorScheme.onSurfaceVariant) } }
            items(messages, key = { it.id }) { MessageRow(it) }
            if (generating) item { Text("در حال تولید پاسخ…", color = MaterialTheme.colorScheme.onSurfaceVariant) }
            if (!generating && execution != null) item { ActionSummary(execution, onWorkspace) }
        }
        Surface(Modifier.fillMaxWidth().padding(bottom = 12.dp), shape = RoundedCornerShape(24.dp), color = MaterialTheme.colorScheme.surface.copy(alpha = .96f), tonalElevation = 2.dp) {
            Row(Modifier.padding(8.dp), verticalAlignment = Alignment.Bottom) {
                TextField(
                    value = composer,
                    onValueChange = onComposer,
                    modifier = Modifier.weight(1f).onPreviewKeyEvent { event ->
                        event.type == KeyEventType.KeyDown && event.key == Key.Enter && event.isShiftPressed
                            .also { if (it) onSend() }
                    },
                    placeholder = { Text("پیام خود را بنویسید…") },
                    maxLines = 6,
                )
                Spacer(Modifier.width(8.dp))
                FilledIconButton(onClick = if (generating) onStop else onSend, enabled = generating || composer.isNotBlank(), modifier = Modifier.size(48.dp).semantics { contentDescription = if (generating) "توقف تولید" else "ارسال پیام" }) { Icon(if (generating) Icons.Default.Stop else Icons.Default.Send, if (generating) "توقف تولید" else "ارسال پیام") }
            }
        }
    }
}

@Composable private fun MessageRow(message: UiMessage) {
    val user = message.role == ChatMessage.Role.USER
    Column(Modifier.fillMaxWidth(), horizontalAlignment = if (user) Alignment.Start else Alignment.End) {
        Text(if (user) "شما" else "مدل", style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        if (user) Surface(color = MaterialTheme.colorScheme.primaryContainer.copy(alpha = .70f), shape = RoundedCornerShape(18.dp)) { Text(message.text, Modifier.padding(13.dp)) }
        else Text(message.text, Modifier.fillMaxWidth(.94f))
    }
}

@Composable private fun ActionSummary(execution: ExecutionState, onWorkspace: () -> Unit) {
    Surface(Modifier.fillMaxWidth(), shape = RoundedCornerShape(18.dp), color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = .55f)) {
        Row(Modifier.padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) { Text("آخرین عملیات: ${execution.action}", style = MaterialTheme.typography.titleSmall); Text("${execution.status} · ${durationText(execution)}", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant) }
            TextButton(onClick = onWorkspace, Modifier.heightIn(min = 48.dp)) { Text("مشاهده جزئیات") }
        }
    }
}

@Composable private fun WorkspacePage(execution: ExecutionState?, model: ModelDescriptor?, onDiagnostics: () -> Unit) {
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
                        ExpandLayer("پردازش و Trace همان Event") { Text("Trace فقط از داده واقعی Execution قابل نمایش است.", color = MaterialTheme.colorScheme.onSurfaceVariant) }
                        ExpandLayer("Preview / Result") { Text(execution.resultPreview ?: "نتیجه‌ای ثبت نشده است.", style = MaterialTheme.typography.bodySmall) }
                        ExpandLayer("Verification / Evidence") { Text("Evidence برای این Event ثبت نشده است.", color = MaterialTheme.colorScheme.onSurfaceVariant) }
                        ExpandLayer("Recovery") { Text(execution.error ?: "بازیابی لازم نیست.", color = MaterialTheme.colorScheme.onSurfaceVariant) }
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

private fun durationText(execution: ExecutionState): String {
    val end = execution.finishedAt ?: System.currentTimeMillis()
    return "${(end - execution.startedAt).coerceAtLeast(0L)} ms"
}

@Composable private fun DiagnosticsPage(error: String?, execution: ExecutionState?) {
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
            Text("Verification", style = MaterialTheme.typography.titleMedium)
            Text("Verification ثبت‌شده‌ای برای این Execution وجود ندارد.", color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        Text("عملکرد", style = MaterialTheme.typography.titleLarge)
        Text("هیچ معیار ساختگی نمایش داده نمی‌شود.", color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text("گزارش کار حساس", style = MaterialTheme.typography.titleLarge)
        Text("گزارش حساسیتی برای این Execution ثبت نشده است.", color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@Composable private fun AboutPage() {
    SimplePage("درباره برنامه") {
        Text("نسخه و مشخصات برنامه", style = MaterialTheme.typography.titleLarge)
        Text("رابط کاربری فارسی و RTL بر اساس قرارداد UI v1.0.0.", color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text("مدیریت مدل محلی از مسیر تنظیمات → هوش مصنوعی → مدل انجام می‌شود.", color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@Composable private fun SimplePage(title: String, content: @Composable ColumnScope.() -> Unit) {
    LazyColumn(Modifier.fillMaxSize().padding(20.dp), contentPadding = PaddingValues(bottom = 32.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        item { Text(title, style = MaterialTheme.typography.headlineMedium) }
        item { Column(verticalArrangement = Arrangement.spacedBy(10.dp), content = content) }
    }
}
