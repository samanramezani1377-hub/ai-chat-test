package com.samanramezani.aichattest

import android.app.Activity
import android.os.Bundle
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.compose.setContent
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.selected
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.input.KeyboardOptions
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.woogit.aicore.domain.ChatMessage
import com.woogit.aicore.domain.InferenceSettings
import com.woogit.aicore.domain.ModelDescriptor
import com.woogit.aicore.domain.ModelError
import com.woogit.aicore.domain.ModelResult
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
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
    var runtimeStatus by remember { mutableStateOf("آماده") }
    var activeModel by remember { mutableStateOf<ModelDescriptor?>(null) }
    var models by remember { mutableStateOf<List<ModelDescriptor>>(emptyList()) }
    var diagnostic by remember { mutableStateOf<String?>(null) }
    var undoVisible by remember { mutableStateOf(false) }
    val scope = rememberCoroutineScope()

    fun refreshModels() {
        scope.launch(Dispatchers.Default) {
            val manager = container.modelManager
            val listed = manager?.models()
            val active = manager?.activeModel()
            withContext(Dispatchers.Main) {
                models = when (listed) { is ModelResult.Success -> listed.value; else -> emptyList() }
                activeModel = when (active) { is ModelResult.Success -> active.value; else -> null }
                if (!generating) runtimeStatus = if (activeModel != null) "آماده" else "خارج از دسترس"
            }
        }
    }

    LaunchedEffect(Unit) {
        container.modelManager?.let { manager ->
            withContext(Dispatchers.Default) { manager.restoreActive() }
        }
        refreshModels()
    }

    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri == null) return@rememberLauncherForActivityResult
        val manager = container.modelManager ?: return@rememberLauncherForActivityResult
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

    val send: () -> Unit = send@{
        val manager = container.modelManager ?: return@send
        val text = composer.trim()
        if (text.isEmpty() || generating) return@send
        composer = ""
        messages = messages + UiMessage(ChatMessage.Role.USER, text)
        generating = true
        runtimeStatus = "در حال تولید"
        diagnostic = null
        scope.launch(Dispatchers.Default) {
            val request = messages.takeLast(24).map { ChatMessage(it.role, it.text) }
            val result = try {
                manager.generate(request, InferenceSettings(maxNewTokens = 512))
            } catch (t: CancellationException) {
                throw t
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

    val stop: () -> Unit = {
        scope.launch(Dispatchers.Default) { container.modelManager?.stopGeneration() }
        runtimeStatus = "در حال کار"
    }

    CompositionLocalProvider(LocalLayoutDirection provides androidx.compose.ui.unit.LayoutDirection.Rtl) {
        MaterialTheme(colorScheme = appColors()) {
            Surface(Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
                Box(Modifier.fillMaxSize()) {
                    Column(Modifier.fillMaxSize()) {
                        Header(destination, runtimeStatus, activeModel, onDestination = { destination = it; quickMenuOpen = false }, onQuickMenu = { quickMenuOpen = !quickMenuOpen }, onSidebar = { sidebarOpen = true })
                        Box(Modifier.fillMaxSize()) {
                            when (destination) {
                                Destination.CHAT -> ChatScreen(messages, composer, generating, onComposer = { composer = it }, onSend = send, onStop = stop)
                                Destination.WORK -> WorkspaceScreen()
                                Destination.DIAGNOSTICS -> DiagnosticsScreen(diagnostic, container)
                                Destination.SETTINGS -> SettingsScreen(models, activeModel, diagnostic, onImport = { picker.launch(arrayOf("application/octet-stream", "application/gzip", "*/*")) }, onRefresh = ::refreshModels)
                                Destination.ABOUT -> AboutScreen()
                            }
                        }
                    }
                    if (quickMenuOpen) QuickMenu(onLogs = { destination = Destination.DIAGNOSTICS; quickMenuOpen = false }, onSettings = { destination = Destination.SETTINGS; quickMenuOpen = false })
                    if (sidebarOpen) Sidebar(hasConversation = messages.isNotEmpty(), onClose = { sidebarOpen = false }, onDestination = { destination = it; sidebarOpen = false }, onNewChat = { messages = emptyList(); composer = ""; destination = Destination.CHAT; sidebarOpen = false }, onDelete = { undoVisible = true })
                    if (undoVisible) UndoBar(onUndo = { undoVisible = false }, onDismiss = { undoVisible = false })
                }
            }
        }
    }
}

@Composable private fun appColors() = lightColorScheme(primary = Color(0xFF356AE6), onPrimary = Color.White, primaryContainer = Color(0xFFE6EDFF), onPrimaryContainer = Color(0xFF16336F), surface = Color(0xFFFBFCFE), surfaceContainer = Color(0xFFF0F3F8), background = Color(0xFFF7F8FB), error = Color(0xFFBA1A1A), outline = Color(0xFFD8DCE5))

@Composable
private fun Header(destination: Destination, status: String, model: ModelDescriptor?, onDestination: (Destination) -> Unit, onQuickMenu: () -> Unit, onSidebar: () -> Unit) {
    Surface(Modifier.fillMaxWidth().padding(horizontal = 10.dp, vertical = 8.dp), shape = RoundedCornerShape(22.dp), tonalElevation = 2.dp, shadowElevation = 1.dp) {
        Row(Modifier.fillMaxWidth().height(68.dp).padding(horizontal = 8.dp), verticalAlignment = Alignment.CenterVertically) {
            Row(Modifier.weight(.9f).clip(RoundedCornerShape(18.dp)).background(MaterialTheme.colorScheme.surfaceContainer).padding(4.dp), verticalAlignment = Alignment.CenterVertically) {
                HeaderTab("گفت‌وگو", destination == Destination.CHAT) { onDestination(Destination.CHAT) }
                HeaderTab("کار", destination == Destination.WORK) { onDestination(Destination.WORK) }
                IconButton(onClick = onQuickMenu, modifier = Modifier.semantics { contentDescription = "منوی سریع" }) { Icon(Icons.Default.MoreVert, null) }
            }
            Spacer(Modifier.width(12.dp))
            RuntimeStatus(status, model, Modifier.weight(1f))
            Spacer(Modifier.width(8.dp))
            IconButton(onClick = onSidebar, modifier = Modifier.size(48.dp).clip(CircleShape).background(MaterialTheme.colorScheme.surfaceContainer).semantics { contentDescription = "باز کردن منوی اصلی" }) { Icon(Icons.Default.Menu, null) }
        }
    }
}

@Composable private fun HeaderTab(text: String, active: Boolean, onClick: () -> Unit) { Column(horizontalAlignment = Alignment.CenterHorizontally) { TextButton(onClick = onClick, modifier = Modifier.height(48.dp), colors = ButtonDefaults.textButtonColors(contentColor = MaterialTheme.colorScheme.onSurface)) { Text(text) }; AnimatedVisibility(active) { HorizontalDivider(Modifier.width(34.dp), thickness = 2.dp, color = MaterialTheme.colorScheme.primary) } } }

@Composable private fun RuntimeStatus(status: String, model: ModelDescriptor?, modifier: Modifier) {
    val active = status == "در حال کار" || status == "در حال تولید"
    val alpha by animateFloatAsState(if (active) .55f else 1f, label = "runtime")
    Row(modifier, verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.Center) {
        Box(Modifier.size(9.dp).clip(CircleShape).background(runtimeColor(status).copy(alpha = alpha)).semantics { contentDescription = "وضعیت Runtime: $status" })
        Spacer(Modifier.width(8.dp))
        Column(horizontalAlignment = Alignment.Start) { Text(status, style = MaterialTheme.typography.labelLarge); Text(model?.displayName ?: "مدل محلی", style = MaterialTheme.typography.labelSmall, maxLines = 1) }
    }
}

private fun runtimeColor(status: String) = when (status) { "آماده" -> Color(0xFF238636); "در حال کار" -> Color(0xFFD97706); "در حال تولید" -> Color(0xFF356AE6); "خارج از دسترس" -> Color(0xFF6B7280); else -> Color(0xFFBA1A1A) }

@Composable private fun QuickMenu(onLogs: () -> Unit, onSettings: () -> Unit) { Surface(Modifier.padding(start = 14.dp, top = 82.dp).width(190.dp), shape = RoundedCornerShape(18.dp), tonalElevation = 6.dp, shadowElevation = 3.dp) { Column(Modifier.padding(8.dp)) { TextButton(onClick = onLogs, Modifier.fillMaxWidth()) { Text("لاگ") }; TextButton(onClick = onSettings, Modifier.fillMaxWidth()) { Text("تنظیمات") } } } }

@Composable
private fun Sidebar(hasConversation: Boolean, onClose: () -> Unit, onDestination: (Destination) -> Unit, onNewChat: () -> Unit, onDelete: () -> Unit) {
    Box(Modifier.fillMaxSize().background(Color.Black.copy(alpha = .24f)).clickable { onClose() }) {
        Surface(Modifier.fillMaxHeight().fillMaxWidth(.84f).align(Alignment.CenterEnd).clickable(enabled = false) {}, shape = RoundedCornerShape(topStart = 26.dp, bottomStart = 26.dp), tonalElevation = 6.dp, shadowElevation = 8.dp) {
            LazyColumn(Modifier.fillMaxSize().padding(20.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                item { Button(onClick = onNewChat, Modifier.fillMaxWidth().height(52.dp), shape = RoundedCornerShape(16.dp)) { Icon(Icons.Default.Add, null); Spacer(Modifier.width(8.dp)); Text("گفت‌وگوی جدید") } }
                item { SidebarHeading("ابزار") }
                item { SidebarItem("فضای کار") { onDestination(Destination.WORK) } }
                item { SidebarItem("عیب‌یابی") { onDestination(Destination.DIAGNOSTICS) } }
                item { SidebarItem("تنظیمات") { onDestination(Destination.SETTINGS) } }
                item { SidebarHeading("اطلاعات") }
                item { SidebarItem("درباره برنامه") { onDestination(Destination.ABOUT) } }
                item { SidebarHeading("اخیر") }
                if (hasConversation) item { RecentItem(onDelete) }
            }
        }
    }
}

@Composable private fun SidebarHeading(text: String) { Text(text, style = MaterialTheme.typography.labelMedium, modifier = Modifier.padding(top = 14.dp, bottom = 4.dp)) }
@Composable private fun SidebarItem(text: String, onClick: () -> Unit) { TextButton(onClick = onClick, Modifier.fillMaxWidth().height(48.dp)) { Text(text, Modifier.fillMaxWidth()) } }

@Composable
private fun RecentItem(onDelete: () -> Unit) {
    var menu by remember { mutableStateOf(false) }
    Box(Modifier.fillMaxWidth()) {
        Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(14.dp)).background(MaterialTheme.colorScheme.surfaceContainer).pointerInput(Unit) { detectTapGestures(onLongPress = { menu = true }) }.padding(horizontal = 10.dp, vertical = 9.dp), verticalAlignment = Alignment.CenterVertically) {
            Box(Modifier.size(8.dp).clip(CircleShape).background(MaterialTheme.colorScheme.primary))
            Spacer(Modifier.width(8.dp))
            Text("گفت‌وگوی فعلی", Modifier.weight(1f), maxLines = 1)
            Text("همین الان", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        if (menu) Surface(Modifier.align(Alignment.TopStart).padding(top = 46.dp), shape = RoundedCornerShape(14.dp), tonalElevation = 5.dp, shadowElevation = 2.dp) { Column(Modifier.padding(6.dp)) { TextButton(onClick = { menu = false }) { Text("باز کردن") }; TextButton(onClick = { menu = false }) { Text("تغییر نام") }; TextButton(onClick = { menu = false; onDelete() }, colors = ButtonDefaults.textButtonColors(contentColor = MaterialTheme.colorScheme.error)) { Text("حذف") } } }
    }
}

@Composable
private fun UndoBar(onUndo: () -> Unit, onDismiss: () -> Unit) {
    var remaining by remember { mutableIntStateOf(10) }
    LaunchedEffect(Unit) { while (remaining > 0) { delay(1000); remaining-- }; onDismiss() }
    Surface(Modifier.align(Alignment.BottomCenter).padding(12.dp).fillMaxWidth(), shape = RoundedCornerShape(18.dp), tonalElevation = 5.dp, shadowElevation = 3.dp) {
        Row(Modifier.padding(horizontal = 14.dp, vertical = 10.dp), verticalAlignment = Alignment.CenterVertically) { Text("گفتگو حذف شد", Modifier.weight(1f)); TextButton(onClick = onUndo) { Text("بازگردانی") }; Text("$remaining"); IconButton(onClick = onDismiss) { Icon(Icons.Default.Close, "بستن") } }
    }
}

@Composable
private fun ChatScreen(messages: List<UiMessage>, composer: String, generating: Boolean, onComposer: (String) -> Unit, onSend: () -> Unit, onStop: () -> Unit) {
    val listState = rememberLazyListState()
    LaunchedEffect(messages.size) { if (messages.isNotEmpty()) listState.animateScrollToItem(messages.lastIndex) }
    Column(Modifier.fillMaxSize()) {
        LazyColumn(state = listState, modifier = Modifier.weight(1f).fillMaxWidth(), contentPadding = PaddingValues(start = 18.dp, end = 18.dp, top = 28.dp, bottom = 18.dp), verticalArrangement = Arrangement.spacedBy(20.dp)) {
            if (messages.isEmpty()) item { EmptyChat() }
            items(messages, key = { it.id }) { MessageView(it) }
            if (generating) item { Text("در حال تولید پاسخ…", color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(horizontal = 8.dp)) }
        }
        Composer(composer, generating, onComposer, onSend, onStop)
    }
}

@Composable private fun EmptyChat() { Column(Modifier.fillMaxWidth().padding(top = 80.dp), horizontalAlignment = Alignment.CenterHorizontally) { Text("گفت‌وگو", style = MaterialTheme.typography.headlineMedium); Spacer(Modifier.height(8.dp)); Text("پیام خود را بنویسید تا گفت‌وگو آغاز شود.", color = MaterialTheme.colorScheme.onSurfaceVariant) } }

@Composable private fun MessageView(message: UiMessage) { val user = message.role == ChatMessage.Role.USER; Column(Modifier.fillMaxWidth(), horizontalAlignment = if (user) Alignment.Start else Alignment.End) { Text(if (user) "شما" else "مدل", style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant); Spacer(Modifier.height(5.dp)); if (user) Surface(color = MaterialTheme.colorScheme.primaryContainer.copy(alpha = .72f), shape = RoundedCornerShape(18.dp)) { Text(message.text, Modifier.padding(horizontal = 15.dp, vertical = 11.dp), lineHeight = 24.sp) } else Text(message.text, Modifier.fillMaxWidth(.94f), lineHeight = 25.sp) } }

@Composable
private fun Composer(value: String, generating: Boolean, onValue: (String) -> Unit, onSend: () -> Unit, onStop: () -> Unit) {
    Surface(Modifier.fillMaxWidth().padding(horizontal = 14.dp, vertical = 12.dp), shape = RoundedCornerShape(24.dp), tonalElevation = 3.dp, shadowElevation = 2.dp) {
        Row(Modifier.padding(8.dp), verticalAlignment = Alignment.Bottom) {
            BasicTextField(value, onValue, Modifier.weight(1f).heightIn(min = 46.dp, max = 150.dp).padding(horizontal = 10.dp, vertical = 12.dp), textStyle = LocalTextStyle.current.copy(fontSize = 16.sp), keyboardOptions = KeyboardOptions(imeAction = ImeAction.Default), decorationBox = { inner -> if (value.isEmpty()) Text("پیام خود را بنویسید…", color = MaterialTheme.colorScheme.onSurfaceVariant); inner() })
            FilledIconButton(onClick = if (generating) onStop else onSend, enabled = generating || value.isNotBlank(), modifier = Modifier.size(48.dp).semantics { contentDescription = if (generating) "توقف" else "ارسال" }) { Icon(if (generating) Icons.Default.Stop else Icons.Default.Send, null) }
        }
    }
}

@Composable private fun WorkspaceScreen() { SimplePage("فضای کار") { Text("خلاصه اجرای جاری", style = MaterialTheme.typography.titleLarge); Text("اجرای فعالی ثبت نشده است.", color = MaterialTheme.colorScheme.onSurfaceVariant); HorizontalDivider(); Text("Timeline مرکزی", style = MaterialTheme.typography.titleLarge); Text("رویدادهای واقعی Action، Verification و نتیجه در این Timeline نمایش داده می‌شوند.", color = MaterialTheme.colorScheme.onSurfaceVariant); HorizontalDivider(); Text("تاریخچه عملکرد", style = MaterialTheme.typography.titleLarge); Text("سوابق واقعی عملکرد پس از ثبت توسط Runtime در این بخش قابل بررسی است.", color = MaterialTheme.colorScheme.onSurfaceVariant) } }

@Composable private fun DiagnosticsScreen(error: String?, container: AppContainer) {
    val runtime = remember { container.modelRuntime.runtimeInfo() }
    SimplePage("عیب‌یابی") {
        Text("خطاها", style = MaterialTheme.typography.titleLarge)
        if (error != null) Text(error, color = MaterialTheme.colorScheme.error) else Text("خطای ثبت‌شده‌ای برای نمایش وجود ندارد.", color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text("گزارش Execution / Trace", style = MaterialTheme.typography.titleLarge)
        Text("Trace فقط از رویدادهای واقعی Core نمایش داده می‌شود.", color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text("عملکرد", style = MaterialTheme.typography.titleLarge)
        Text("Metricهای واقعی Runtime در صورت ثبت‌شدن نمایش داده می‌شوند.", color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text("Runtime: ${runtime.name} ${runtime.version}", fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace, fontSize = 13.sp)
        Text("Backend: ${runtime.backend ?: "نامشخص"}", fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace, fontSize = 13.sp)
        Text("گزارش کار حساس", style = MaterialTheme.typography.titleLarge)
        Text("گزارش حساس فقط در صورت وجود اجرای واقعی نمایش داده می‌شود.", color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@Composable
private fun SettingsScreen(models: List<ModelDescriptor>, active: ModelDescriptor?, error: String?, onImport: () -> Unit, onRefresh: () -> Unit) {
    var section by remember { mutableStateOf("مدل") }
    SimplePage("تنظیمات") {
        Text("هوش مصنوعی", style = MaterialTheme.typography.titleLarge)
        SettingsRow("مدل", section == "مدل") { section = "مدل" }
        SettingsRow("استنتاج", section == "استنتاج") { section = "استنتاج" }
        SettingsRow("زمینه", section == "زمینه") { section = "زمینه" }
        SettingsRow("عامل", section == "عامل") { section = "عامل" }
        when (section) {
            "مدل" -> {
                Text("مدل فعلی", style = MaterialTheme.typography.titleMedium)
                if (active != null) ModelSummary(active, true) else Text("مدل فعالی وجود ندارد.", color = MaterialTheme.colorScheme.onSurfaceVariant)
                Button(onClick = onImport, Modifier.fillMaxWidth().height(50.dp), shape = RoundedCornerShape(16.dp)) { Icon(Icons.Default.FileOpen, null); Spacer(Modifier.width(8.dp)); Text("انتخاب فایل مدل از گوشی") }
                Text("مدیریت مدل‌ها", style = MaterialTheme.typography.titleMedium)
                models.forEach { ModelSummary(it, it.id == active?.id) }
                if (error != null) Text(error, color = MaterialTheme.colorScheme.error)
                TextButton(onClick = onRefresh) { Text("به‌روزرسانی") }
            }
            "استنتاج" -> Text("کنترل‌های استنتاج فقط زمانی نمایش داده می‌شوند که به تنظیمات واقعی Runtime متصل باشند.", color = MaterialTheme.colorScheme.onSurfaceVariant)
            else -> Text("تنظیمات این بخش پس از اتصال قابلیت واقعی Core نمایش داده می‌شوند.", color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        Spacer(Modifier.height(6.dp)); Text("فضای کار", style = MaterialTheme.typography.titleMedium); Text("لاگ و عیب‌یابی", style = MaterialTheme.typography.titleMedium); Text("عملکرد", style = MaterialTheme.typography.titleMedium); Text("امنیت و تأیید", style = MaterialTheme.typography.titleMedium)
    }
}

@Composable private fun SettingsRow(title: String, selected: Boolean, onClick: () -> Unit) { TextButton(onClick = onClick, Modifier.fillMaxWidth().semantics { this.selected = selected }) { Text(title, Modifier.fillMaxWidth()) } }
@Composable private fun ModelSummary(model: ModelDescriptor, active: Boolean) { Surface(Modifier.fillMaxWidth().padding(vertical = 5.dp), shape = RoundedCornerShape(16.dp), tonalElevation = if (active) 2.dp else 0.dp) { Column(Modifier.padding(14.dp)) { Text(model.displayName, style = MaterialTheme.typography.titleMedium); Text("GGUF · ${model.quantization}"); Text(if (active) "آماده و فعال" else "وضعیت: ${model.state}", style = MaterialTheme.typography.labelMedium) } } }
@Composable private fun AboutScreen() { SimplePage("درباره برنامه") { Text("AI Chat Test", style = MaterialTheme.typography.headlineMedium); Text("اطلاعات نسخه و مشخصات برنامه در این بخش قرار می‌گیرد.") } }
@Composable private fun SimplePage(title: String, content: @Composable ColumnScope.() -> Unit) { Column(Modifier.fillMaxSize().padding(horizontal = 20.dp, vertical = 22.dp), verticalArrangement = Arrangement.spacedBy(14.dp), content = { Text(title, style = MaterialTheme.typography.headlineMedium); content() }) }
