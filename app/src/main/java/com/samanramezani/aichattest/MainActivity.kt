package com.samanramezani.aichattest

import android.app.Activity
import android.os.Bundle
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.compose.setContent
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
import com.woogit.aicore.domain.ChatMessage
import com.woogit.aicore.domain.InferenceSettings
import com.woogit.aicore.domain.ModelDescriptor
import com.woogit.aicore.domain.ModelError
import com.woogit.aicore.domain.ModelResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private enum class Destination { CHAT, WORK, DIAGNOSTICS, SETTINGS, ABOUT }
private data class UiMessage(val role: ChatMessage.Role, val text: String, val id: Long = System.nanoTime())

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
    var deletedMessages by remember { mutableStateOf<List<UiMessage>?>(null) }
    var conversationTitle by remember { mutableStateOf("گفت‌وگوی فعلی") }
    val scope = rememberCoroutineScope()
    val manager = container.modelManager

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

    LaunchedEffect(Unit) {
        if (manager != null) withContext(Dispatchers.Default) { manager.restoreActive() }
        refreshModels()
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
        if (manager == null || generating) return
        val text = composer.trim()
        if (text.isEmpty()) return
        composer = ""
        messages = messages + UiMessage(ChatMessage.Role.USER, text)
        generating = true
        runtimeStatus = "در حال تولید"
        diagnostic = null
        scope.launch(Dispatchers.Default) {
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
                        messages = messages + UiMessage(ChatMessage.Role.ASSISTANT, result.value.text)
                        runtimeStatus = "آماده"
                    }
                    is ModelResult.Failure -> {
                        runtimeStatus = "خطا"
                        diagnostic = result.error.message
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
    }

    CompositionLocalProvider(LocalLayoutDirection provides LayoutDirection.Rtl) {
        MaterialTheme(colorScheme = lightColorScheme()) {
            Surface(Modifier.fillMaxSize()) {
                Box(Modifier.fillMaxSize()) {
                    Column(Modifier.fillMaxSize()) {
                        AppHeader(destination, runtimeStatus, activeModel, onDestination = { destination = it; quickMenuOpen = false }, onQuick = { quickMenuOpen = !quickMenuOpen }, onSidebar = { sidebarOpen = true })
                        Box(Modifier.weight(1f).fillMaxWidth()) {
                            when (destination) {
                                Destination.CHAT -> ChatPage(messages, composer, generating, { composer = it }, ::send, ::stop)
                                Destination.WORK -> WorkspacePage(generating, runtimeStatus, activeModel)
                                Destination.DIAGNOSTICS -> DiagnosticsPage(diagnostic, container)
                                Destination.SETTINGS -> SettingsScreen(models, activeModel, diagnostic, onImport = { picker.launch(arrayOf("application/octet-stream", "application/gzip", "*/*")) }, onRefresh = ::refreshModels, onActivate = { id -> scope.launch(Dispatchers.Default) { manager?.activate(id); refreshModels() } }, onDelete = { id -> scope.launch(Dispatchers.Default) { manager?.delete(id); refreshModels() } })
                                Destination.ABOUT -> AboutPage()
                            }
                        }
                    }
                    if (quickMenuOpen) QuickMenu({ destination = Destination.DIAGNOSTICS; quickMenuOpen = false }, { destination = Destination.SETTINGS; quickMenuOpen = false })
                    if (sidebarOpen) Sidebar(conversationTitle, messages.isNotEmpty(), destination, { sidebarOpen = false }, { destination = it; sidebarOpen = false }, { messages = emptyList(); composer = ""; conversationTitle = "گفت‌وگوی فعلی"; destination = Destination.CHAT; sidebarOpen = false }, { conversationTitle = it; sidebarOpen = false }, { deletedMessages = messages; messages = emptyList(); sidebarOpen = false })
                    deletedMessages?.let { deleted -> UndoBar(onUndo = { messages = deleted; deletedMessages = null }, onDismiss = { deletedMessages = null }) }
                }
            }
        }
    }
}

@Composable private fun AppHeader(destination: Destination, status: String, model: ModelDescriptor?, onDestination: (Destination) -> Unit, onQuick: () -> Unit, onSidebar: () -> Unit) {
    Surface(Modifier.fillMaxWidth().padding(10.dp), shape = MaterialTheme.shapes.large, tonalElevation = 2.dp, shadowElevation = 1.dp) {
        Row(Modifier.fillMaxWidth().heightIn(min = 64.dp).padding(6.dp), verticalAlignment = Alignment.CenterVertically) {
            IconButton(onClick = onSidebar, modifier = Modifier.size(48.dp).semantics { contentDescription = "باز کردن نوار کناری" }) { Icon(Icons.Default.Menu, null) }
            Spacer(Modifier.weight(1f))
            Column(horizontalAlignment = Alignment.End) { Text("● $status", style = MaterialTheme.typography.labelLarge); Text(model?.displayName ?: "مدل محلی", style = MaterialTheme.typography.labelSmall) }
            Spacer(Modifier.weight(1f))
            TextButton(onClick = { onDestination(Destination.CHAT) }) { Text("گفت‌وگو") }
            TextButton(onClick = { onDestination(Destination.WORK) }) { Text("کار") }
            IconButton(onClick = onQuick) { Icon(Icons.Default.MoreVert, "منوی سریع") }
        }
    }
}

@Composable private fun QuickMenu(onDiagnostics: () -> Unit, onSettings: () -> Unit) {
    Surface(Modifier.padding(top = 76.dp, end = 12.dp).width(190.dp), shape = MaterialTheme.shapes.large, tonalElevation = 6.dp) { Column(Modifier.padding(8.dp)) { TextButton(onClick = onDiagnostics, Modifier.fillMaxWidth()) { Text("لاگ") }; TextButton(onClick = onSettings, Modifier.fillMaxWidth()) { Text("تنظیمات") } } }
}

@Composable private fun Sidebar(title: String, hasConversation: Boolean, destination: Destination, onClose: () -> Unit, onDestination: (Destination) -> Unit, onNew: () -> Unit, onRename: (String) -> Unit, onDelete: () -> Unit) {
    Box(Modifier.fillMaxSize()) {
        Surface(Modifier.fillMaxSize().padding(end = 0.dp), color = MaterialTheme.colorScheme.scrim.copy(alpha = .28f)) { }
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
                if (hasConversation) item { RecentConversation(title, onRename, onDelete) } else item { Text("گفت‌وگوی ذخیره‌شده‌ای وجود ندارد.", color = MaterialTheme.colorScheme.onSurfaceVariant) }
                item { TextButton(onClick = onClose, Modifier.fillMaxWidth()) { Text("بستن") } }
            }
        }
    }
}

@Composable private fun SideNav(text: String, active: Boolean, onClick: () -> Unit) { TextButton(onClick = onClick, Modifier.fillMaxWidth()) { Text(if (active) "● $text" else text, Modifier.fillMaxWidth()) } }

@Composable private fun RecentConversation(title: String, onRename: (String) -> Unit, onDelete: () -> Unit) {
    var menu by remember { mutableStateOf(false) }
    Column {
        Surface(Modifier.fillMaxWidth(), shape = MaterialTheme.shapes.medium, tonalElevation = 1.dp) { Row(Modifier.padding(12.dp), verticalAlignment = Alignment.CenterVertically) { Text("●", color = MaterialTheme.colorScheme.primary); Spacer(Modifier.width(8.dp)); Text(title, Modifier.weight(1f)); Text("اکنون", style = MaterialTheme.typography.labelSmall) } }
        TextButton(onClick = { menu = !menu }, Modifier.align(Alignment.End)) { Text("گزینه‌ها") }
        if (menu) Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End) { TextButton(onClick = { menu = false }) { Text("باز کردن") }; TextButton(onClick = { menu = false; onRename("گفت‌وگوی فعلی") }) { Text("تغییر نام") }; TextButton(onClick = { menu = false; onDelete() }, colors = ButtonDefaults.textButtonColors(contentColor = MaterialTheme.colorScheme.error)) { Text("حذف") } }
    }
}

@Composable private fun UndoBar(onUndo: () -> Unit, onDismiss: () -> Unit) {
    var remaining by remember { mutableIntStateOf(10) }
    LaunchedEffect(Unit) { while (remaining > 0) { kotlinx.coroutines.delay(1000); remaining-- }; onDismiss() }
    Surface(Modifier.align(Alignment.BottomCenter).padding(12.dp).fillMaxWidth(), shape = MaterialTheme.shapes.large, tonalElevation = 6.dp) { Row(Modifier.padding(10.dp), verticalAlignment = Alignment.CenterVertically) { Text("گفتگو حذف شد", Modifier.weight(1f)); TextButton(onClick = onUndo) { Text("بازگردانی") }; Text("$remaining") } }
}

@Composable private fun ChatPage(messages: List<UiMessage>, composer: String, generating: Boolean, onComposer: (String) -> Unit, onSend: () -> Unit, onStop: () -> Unit) {
    Column(Modifier.fillMaxSize().padding(horizontal = 16.dp)) {
        LazyColumn(Modifier.weight(1f).fillMaxWidth(), contentPadding = PaddingValues(vertical = 20.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
            if (messages.isEmpty()) item { Column(Modifier.fillMaxWidth().padding(top = 70.dp), horizontalAlignment = Alignment.CenterHorizontally) { Text("گفت‌وگو", style = MaterialTheme.typography.headlineMedium); Text("پیام خود را بنویسید و گفتگو را شروع کنید.", color = MaterialTheme.colorScheme.onSurfaceVariant) } }
            items(messages, key = { it.id }) { message -> MessageRow(message) }
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

@Composable private fun MessageRow(message: UiMessage) { val user = message.role == ChatMessage.Role.USER; Column(Modifier.fillMaxWidth(), horizontalAlignment = if (user) Alignment.Start else Alignment.End) { Text(if (user) "شما" else "مدل", style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant); if (user) Surface(color = MaterialTheme.colorScheme.primaryContainer.copy(alpha = .70f), shape = MaterialTheme.shapes.large) { Text(message.text, Modifier.padding(13.dp)) } else Text(message.text, Modifier.fillMaxWidth(.94f)) } }

@Composable private fun WorkspacePage(generating: Boolean, status: String, model: ModelDescriptor?) { SimplePage("فضای کار") { Text("خلاصه اجرای جاری", style = MaterialTheme.typography.titleLarge); if (generating) Text("در حال تولید پاسخ") else Text("اجرای عملیاتی فعالی ثبت نشده است.", color = MaterialTheme.colorScheme.onSurfaceVariant); Text("Timeline مرکزی", style = MaterialTheme.typography.titleLarge); Text("وقتی Action واقعی اجرا شود، رویدادهای همان Execution اینجا نمایش داده می‌شوند.", color = MaterialTheme.colorScheme.onSurfaceVariant); Text("تاریخچه عملکرد", style = MaterialTheme.typography.titleLarge); Text(if (model != null) "Runtime: $status · مدل: ${model.displayName}" else "داده عملکردی ثبت نشده است.", color = MaterialTheme.colorScheme.onSurfaceVariant) } }

@Composable private fun DiagnosticsPage(error: String?, container: AppContainer) { val info = container.modelRuntime.runtimeInfo(); SimplePage("عیب‌یابی") { Text("خطاها", style = MaterialTheme.typography.titleLarge); Text(error ?: "خطایی ثبت نشده است.", color = if (error == null) MaterialTheme.colorScheme.onSurfaceVariant else MaterialTheme.colorScheme.error); Text("گزارش Execution / Trace", style = MaterialTheme.typography.titleLarge); Text("Runtime: ${info.name} ${info.version} · Backend: ${info.backend ?: "نامشخص"}"); Text("عملکرد", style = MaterialTheme.typography.titleLarge); Text("تا زمانی که Runtime داده واقعی عملکردی ثبت نکند، عدد ساختگی نمایش داده نمی‌شود.", color = MaterialTheme.colorScheme.onSurfaceVariant); Text("گزارش کار حساس", style = MaterialTheme.typography.titleLarge); Text("گزارش ثبت‌شده‌ای وجود ندارد.", color = MaterialTheme.colorScheme.onSurfaceVariant) } }

@Composable private fun AboutPage() { SimplePage("درباره برنامه") { Text("AI Chat Test", style = MaterialTheme.typography.headlineMedium); Text("نسخه ${BuildConfig.VERSION_NAME}"); Text("رابط کاربری فارسی و راست‌به‌چپ"); Text("Runtime محلی GGUF") } }

@Composable private fun SimplePage(title: String, content: @Composable ColumnScope.() -> Unit) { LazyColumn(Modifier.fillMaxSize().padding(20.dp), contentPadding = PaddingValues(bottom = 24.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) { item { Text(title, style = MaterialTheme.typography.headlineMedium) }; item(content) } }
