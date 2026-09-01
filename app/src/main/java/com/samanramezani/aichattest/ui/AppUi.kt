package com.samanramezani.aichattest.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Menu
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material.icons.filled.Send
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.ui.unit.LayoutDirection
import androidx.compose.ui.unit.dp

@Composable
fun AppUi() {
    var destination by remember { mutableStateOf(AppDestination.Chat) }
    var sidebarOpen by remember { mutableStateOf(false) }
    var quickMenuOpen by remember { mutableStateOf(false) }
    var text by remember { mutableStateOf("") }
    val recent = remember { mutableStateListOf("گفت‌وگوی فعلی", "گفت‌وگوی ۱", "گفت‌وگوی ۲", "گفت‌وگوی ۳", "گفت‌وگوی ۴") }
    CompositionLocalProvider(LocalLayoutDirection provides LayoutDirection.Rtl) {
        Surface(Modifier.fillMaxSize()) {
            Box(Modifier.fillMaxSize()) {
                Column(Modifier.fillMaxSize()) {
                    Header(destination, { quickMenuOpen = !quickMenuOpen }, { sidebarOpen = true })
                    when (destination) {
                        AppDestination.Chat -> Chat(text, { text = it })
                        AppDestination.Work -> Workspace()
                        AppDestination.Diagnostics -> Diagnostics()
                        AppDestination.Settings -> Settings()
                        AppDestination.About -> About()
                    }
                }
                if (quickMenuOpen) QuickMenu({ destination = AppDestination.Diagnostics; quickMenuOpen = false }, { destination = AppDestination.Settings; quickMenuOpen = false })
                if (sidebarOpen) Sidebar(destination, recent, { destination = it; sidebarOpen = false }, { sidebarOpen = false }, { repeat(5) { recent.add("گفت‌وگوی ${recent.size}") } })
            }
        }
    }
}

enum class AppDestination { Chat, Work, Diagnostics, Settings, About }

@Composable private fun Header(destination: AppDestination, onQuick: () -> Unit, onSidebar: () -> Unit) {
    Surface(tonalElevation = 2.dp, shadowElevation = 1.dp, shape = RoundedCornerShape(bottomStart = 18.dp, bottomEnd = 18.dp)) {
        Row(Modifier.fillMaxWidth().height(72.dp).padding(horizontal = 10.dp), verticalAlignment = Alignment.CenterVertically) {
            Row(horizontalArrangement = Arrangement.spacedBy(2.dp)) {
                TextButton(onClick = {}) { Text("گفت‌وگو") }
                TextButton(onClick = {}) { Text("کار") }
                IconButton(onClick = onQuick) { Icon(Icons.Default.MoreVert, "منوی سریع") }
            }
            Spacer(Modifier.weight(1f))
            Column(horizontalAlignment = Alignment.CenterHorizontally) { Text("● آماده"); Text("مدل محلی", style = MaterialTheme.typography.labelSmall) }
            Spacer(Modifier.weight(1f))
            IconButton(onClick = onSidebar, modifier = Modifier.semantics { contentDescription = "باز کردن نوار کناری" }) { Icon(Icons.Default.Menu, "باز کردن نوار کناری") }
        }
    }
}

@Composable private fun Sidebar(destination: AppDestination, recent: List<String>, onDestination: (AppDestination) -> Unit, onClose: () -> Unit, onMore: () -> Unit) {
    Box(Modifier.fillMaxSize().background(MaterialTheme.colorScheme.scrim.copy(alpha = .28f)).clickable { onClose() }) {
        Surface(Modifier.fillMaxHeight().widthIn(min = 290.dp, max = 360.dp).align(Alignment.CenterEnd), tonalElevation = 5.dp, shape = RoundedCornerShape(topStart = 24.dp, bottomStart = 24.dp)) {
            LazyColumn(Modifier.padding(20.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                item { Nav("گفت‌وگوی جدید", false) { onDestination(AppDestination.Chat) } }
                item { Text("ابزار", style = MaterialTheme.typography.labelMedium, modifier = Modifier.padding(top = 10.dp)) }
                item { Nav("فضای کار", destination == AppDestination.Work) { onDestination(AppDestination.Work) } }
                item { Nav("عیب‌یابی", destination == AppDestination.Diagnostics) { onDestination(AppDestination.Diagnostics) } }
                item { Nav("تنظیمات", destination == AppDestination.Settings) { onDestination(AppDestination.Settings) } }
                item { Text("اطلاعات", style = MaterialTheme.typography.labelMedium, modifier = Modifier.padding(top = 10.dp)) }
                item { Nav("درباره برنامه", destination == AppDestination.About) { onDestination(AppDestination.About) } }
                item { Text("اخیر", style = MaterialTheme.typography.labelMedium, modifier = Modifier.padding(top = 10.dp)) }
                items(recent) { title -> Row(Modifier.fillMaxWidth().padding(vertical = 7.dp)) { Text("●  $title", Modifier.weight(1f)); Text("اکنون", style = MaterialTheme.typography.labelSmall) } }
                item { TextButton(onClick = onMore, Modifier.fillMaxWidth()) { Text("مشاهده بیشتر") } }
            }
        }
    }
}

@Composable private fun Nav(text: String, active: Boolean, onClick: () -> Unit) { Text(text, style = if (active) MaterialTheme.typography.bodyLarge else MaterialTheme.typography.bodyMedium, modifier = Modifier.fillMaxWidth().clickable { onClick() }.padding(12.dp)) }

@Composable private fun QuickMenu(onLog: () -> Unit, onSettings: () -> Unit) { Surface(Modifier.padding(start = 12.dp, top = 76.dp).width(180.dp), tonalElevation = 6.dp, shape = RoundedCornerShape(16.dp)) { Column(Modifier.padding(8.dp)) { TextButton(onClick = onLog, Modifier.fillMaxWidth()) { Text("لاگ") }; TextButton(onClick = onSettings, Modifier.fillMaxWidth()) { Text("تنظیمات") } } } }

@Composable private fun Chat(text: String, onText: (String) -> Unit) {
    Column(Modifier.fillMaxSize().padding(horizontal = 18.dp)) {
        LazyColumn(Modifier.weight(1f), contentPadding = PaddingValues(vertical = 24.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
            item { Text("گفت‌وگوی فعال", style = MaterialTheme.typography.headlineSmall) }
        }
        Surface(Modifier.fillMaxWidth().padding(bottom = 14.dp), tonalElevation = 3.dp, shape = RoundedCornerShape(24.dp)) {
            Row(Modifier.padding(8.dp), verticalAlignment = Alignment.Bottom) {
                OutlinedTextField(text, onText, Modifier.weight(1f), placeholder = { Text("پیام خود را بنویسید…") }, maxLines = 6, keyboardOptions = KeyboardOptions(imeAction = ImeAction.Default))
                Spacer(Modifier.width(8.dp)); FilledIconButton(onClick = {}, enabled = text.isNotBlank()) { Icon(Icons.Default.Send, "ارسال") }
            }
        }
    }
}

@Composable private fun Workspace() = Screen("فضای کار", listOf("خلاصه اجرای جاری", "Timeline مرکزی", "Event / Action", "تاریخچه عملکرد"))
@Composable private fun Diagnostics() = Screen("عیب‌یابی", listOf("خطاها", "گزارش Execution / Trace", "عملکرد", "گزارش کار حساس"))
@Composable private fun Settings() = Screen("تنظیمات", listOf("هوش مصنوعی", "مدل", "استنتاج", "زمینه", "عامل", "فضای کار", "لاگ و عیب‌یابی", "عملکرد", "امنیت و تأیید", "Qwen3-1.7B · GGUF · Q6_K"))
@Composable private fun About() = Screen("درباره برنامه", listOf("اطلاعات نسخه و مشخصات برنامه"))

@Composable private fun Screen(title: String, rows: List<String>) { LazyColumn(Modifier.fillMaxSize().padding(24.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) { item { Text(title, style = MaterialTheme.typography.headlineMedium) }; items(rows) { Text(it, style = MaterialTheme.typography.titleMedium, modifier = Modifier.fillMaxWidth().padding(vertical = 8.dp)) } } }
