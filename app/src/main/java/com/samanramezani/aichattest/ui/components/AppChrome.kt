package com.samanramezani.aichattest.ui.components

import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.combinedClickable
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import com.samanramezani.aichattest.ui.navigation.AppDestination
import com.woogit.aicore.domain.ModelDescriptor
import com.woogit.aicore.conversation.ConversationRecord
import kotlinx.coroutines.delay
import java.text.DateFormat
import java.util.Date

@Composable
internal fun AppTheme(content: @Composable () -> Unit) {
    val colors = lightColorScheme(
        primary = Color(0xFF376A9A), onPrimary = Color.White,
        primaryContainer = Color(0xFFDCEEFF), onPrimaryContainer = Color(0xFF102A43),
        background = Color(0xFFF7F9FC), surface = Color(0xFFFDFEFF),
        surfaceVariant = Color(0xFFEAF0F6), onSurface = Color(0xFF18212B),
        onSurfaceVariant = Color(0xFF5E6B78),
    )
    MaterialTheme(
        colorScheme = colors,
        shapes = Shapes(large = RoundedCornerShape(20.dp), extraLarge = RoundedCornerShape(28.dp)),
        content = content,
    )
}

@Composable
internal fun AppHeader(destination: AppDestination, status: String, model: ModelDescriptor?, onDestination: (AppDestination) -> Unit, onQuick: () -> Unit, onSidebar: () -> Unit, onRuntime: () -> Unit) {
    Surface(Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 10.dp), shape = RoundedCornerShape(22.dp), color = MaterialTheme.colorScheme.surface.copy(alpha = .92f), tonalElevation = 1.dp) {
        Row(Modifier.fillMaxWidth().heightIn(min = 68.dp).padding(6.dp), verticalAlignment = Alignment.CenterVertically) {
            IconButton(onClick = onSidebar, modifier = Modifier.size(48.dp).semantics { contentDescription = "باز کردن نوار کناری" }) { Icon(Icons.Default.Menu, "باز کردن نوار کناری") }
            Spacer(Modifier.weight(1f))
            TextButton(onClick = onRuntime, modifier = Modifier.heightIn(min = 48.dp).semantics { contentDescription = "جزئیات وضعیت Runtime" }) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text("● $status", style = MaterialTheme.typography.labelLarge)
                    Text(model?.displayName ?: "مدل محلی", style = MaterialTheme.typography.labelSmall)
                }
            }
            Spacer(Modifier.weight(1f))
            Row(horizontalArrangement = Arrangement.spacedBy(2.dp)) {
                HeaderNav("گفت‌وگو", destination == AppDestination.CHAT) { onDestination(AppDestination.CHAT) }
                HeaderNav("کار", destination == AppDestination.WORK) { onDestination(AppDestination.WORK) }
                IconButton(onClick = onQuick, modifier = Modifier.size(48.dp).semantics { contentDescription = "منوی سریع" }) { Icon(Icons.Default.MoreVert, "منوی سریع") }
            }
        }
    }
}

@Composable
private fun HeaderNav(text: String, active: Boolean, onClick: () -> Unit) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        TextButton(onClick = onClick, modifier = Modifier.heightIn(min = 48.dp)) { Text(text) }
        if (active) Box(Modifier.width(34.dp).height(2.dp).background(MaterialTheme.colorScheme.primary))
    }
}

@Composable
internal fun QuickMenu(onDiagnostics: () -> Unit, onSettings: () -> Unit) {
    Surface(Modifier.padding(top = 82.dp, start = 12.dp).widthIn(max = 220.dp), shape = RoundedCornerShape(18.dp), tonalElevation = 6.dp) {
        Column(Modifier.padding(8.dp)) {
            TextButton(onClick = onDiagnostics, Modifier.fillMaxWidth().heightIn(min = 48.dp)) { Icon(Icons.Default.BugReport, null); Spacer(Modifier.width(8.dp)); Text("عیب‌یابی") }
            TextButton(onClick = onSettings, Modifier.fillMaxWidth().heightIn(min = 48.dp)) { Icon(Icons.Default.Settings, null); Spacer(Modifier.width(8.dp)); Text("تنظیمات") }
        }
    }
}

@Composable
internal fun Sidebar(current: ConversationRecord?, conversations: List<ConversationRecord>, destination: AppDestination, onClose: () -> Unit, onDestination: (AppDestination) -> Unit, onNew: () -> Unit, onOpen: (ConversationRecord) -> Unit, onRename: (ConversationRecord) -> Unit, onDelete: (ConversationRecord) -> Unit, onMore: () -> Unit, canShowMore: Boolean) {
    Box(Modifier.fillMaxSize()) {
        Spacer(Modifier.fillMaxSize().background(Color.Black.copy(alpha = .24f)).clickable(onClick = onClose).semantics { contentDescription = "بستن نوار کناری" })
        Surface(Modifier.fillMaxHeight().fillMaxWidth(.88f).align(Alignment.CenterEnd), shape = RoundedCornerShape(topStart = 28.dp, bottomStart = 28.dp), tonalElevation = 8.dp) {
            LazyColumn(Modifier.fillMaxSize().padding(18.dp), verticalArrangement = Arrangement.spacedBy(5.dp)) {
                item { Text("منو", style = MaterialTheme.typography.headlineSmall); Spacer(Modifier.height(8.dp)) }
                item { Button(onClick = onNew, Modifier.fillMaxWidth().heightIn(min = 48.dp)) { Icon(Icons.Default.Add, null); Spacer(Modifier.width(6.dp)); Text("گفت‌وگوی جدید") } }
                item { SectionLabel("ابزارها") }
                item { SideNav("فضای کار", destination == AppDestination.WORK) { onDestination(AppDestination.WORK) } }
                item { SideNav("عیب‌یابی", destination == AppDestination.DIAGNOSTICS) { onDestination(AppDestination.DIAGNOSTICS) } }
                item { SideNav("تنظیمات", destination == AppDestination.SETTINGS) { onDestination(AppDestination.SETTINGS) } }
                item { SectionLabel("اطلاعات") }
                item { SideNav("درباره برنامه", destination == AppDestination.ABOUT) { onDestination(AppDestination.ABOUT) } }
                item { SectionLabel("اخیر") }
                if (conversations.isEmpty()) item { Text("گفت‌وگوی ذخیره‌شده‌ای وجود ندارد.", color = MaterialTheme.colorScheme.onSurfaceVariant) }
                items(conversations, key = { it.id }) { record -> RecentConversation(record, record.id == current?.id, { onOpen(record) }, { onRename(record) }, { onDelete(record) }) }
                if (canShowMore) item { TextButton(onClick = onMore, Modifier.fillMaxWidth().heightIn(min = 48.dp)) { Text("مشاهده بیشتر") } }
                item { OutlinedButton(onClick = onClose, Modifier.fillMaxWidth().heightIn(min = 48.dp)) { Text("بستن") } }
            }
        }
    }
}

@Composable private fun SectionLabel(text: String) { Text(text, style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(top = 14.dp, bottom = 3.dp)) }
@Composable private fun SideNav(text: String, active: Boolean, onClick: () -> Unit) { TextButton(onClick = onClick, Modifier.fillMaxWidth().heightIn(min = 48.dp)) { Text(if (active) "● $text" else text, Modifier.fillMaxWidth()) } }

@OptIn(ExperimentalFoundationApi::class)
@Composable private fun RecentConversation(record: ConversationRecord, active: Boolean, onOpen: () -> Unit, onLongPress: () -> Unit, onDelete: () -> Unit) {
    var menu by remember { mutableStateOf(false) }
    val time = remember(record.updatedAtEpochMs) { DateFormat.getTimeInstance(DateFormat.SHORT).format(Date(record.updatedAtEpochMs)) }
    Box(Modifier.fillMaxWidth()) {
        Surface(Modifier.fillMaxWidth().combinedClickable(onClick = onOpen, onLongClick = { menu = true }).semantics { contentDescription = "گفت‌وگو ${record.title}. برای گزینه‌ها لمس طولانی کنید." }, shape = RoundedCornerShape(16.dp), color = if (active) MaterialTheme.colorScheme.primaryContainer.copy(alpha = .55f) else Color.Transparent) {
            Row(Modifier.padding(12.dp).heightIn(min = 48.dp), verticalAlignment = Alignment.CenterVertically) {
                Text("●", color = if (active) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.outline)
                Spacer(Modifier.width(8.dp))
                Column(Modifier.weight(1f)) { Text(record.title, maxLines = 1); Text("${record.messageCount} پیام", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant) }
                Text(time, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
        DropdownMenu(expanded = menu, onDismissRequest = { menu = false }) {
            DropdownMenuItem(text = { Text("باز کردن") }, onClick = { menu = false; onOpen() })
            DropdownMenuItem(text = { Text("تغییر نام") }, onClick = { menu = false; onLongPress() })
            DropdownMenuItem(text = { Text("حذف", color = MaterialTheme.colorScheme.error) }, onClick = { menu = false; onDelete() })
        }
    }
}

@Composable
internal fun UndoBar(onUndo: () -> Unit, onDismiss: () -> Unit, onExpire: () -> Unit) {
    var remaining by remember { mutableIntStateOf(10) }
    LaunchedEffect(Unit) { while (remaining > 0) { delay(1000); remaining-- }; onExpire() }
    Box(Modifier.fillMaxSize().padding(12.dp), contentAlignment = Alignment.BottomCenter) {
        Surface(Modifier.fillMaxWidth().widthIn(max = 620.dp), shape = RoundedCornerShape(18.dp), tonalElevation = 6.dp) {
            Row(Modifier.padding(8.dp), verticalAlignment = Alignment.CenterVertically) {
                Text("گفت‌وگو حذف شد", Modifier.weight(1f)); TextButton(onClick = onUndo, Modifier.heightIn(min = 48.dp)) { Text("بازگردانی") }; Text("$remaining ثانیه", style = MaterialTheme.typography.labelMedium); IconButton(onClick = onDismiss, modifier = Modifier.size(48.dp).semantics { contentDescription = "بستن پیام حذف" }) { Icon(Icons.Default.Close, "بستن") }
            }
        }
    }
}

@Composable
internal fun RenameDialog(initial: String, onDismiss: () -> Unit, onConfirm: (String) -> Unit) {
    var value by remember(initial) { mutableStateOf(initial) }
    AlertDialog(onDismissRequest = onDismiss, title = { Text("تغییر نام گفت‌وگو") }, text = { TextField(value, { value = it }, singleLine = true, label = { Text("عنوان") }) }, confirmButton = { TextButton(onClick = { onConfirm(value.trim()) }, enabled = value.isNotBlank(), modifier = Modifier.heightIn(min = 48.dp)) { Text("ذخیره") } }, dismissButton = { TextButton(onClick = onDismiss, modifier = Modifier.heightIn(min = 48.dp)) { Text("انصراف") } })
}

@Composable
internal fun RuntimeDetailsDialog(status: String, model: ModelDescriptor?, runtimeName: String, runtimeVersion: String, backend: String?, onDismiss: () -> Unit) {
    AlertDialog(onDismissRequest = onDismiss, title = { Text("وضعیت Runtime") }, text = { Column(verticalArrangement = Arrangement.spacedBy(8.dp)) { Text("وضعیت: $status"); Text("مدل: ${model?.displayName ?: "مدلی فعال نیست"}"); Text("Runtime: $runtimeName"); Text("نسخه: $runtimeVersion"); Text("Backend: ${backend ?: "ارائه نشده"}") } }, confirmButton = { TextButton(onClick = onDismiss, modifier = Modifier.heightIn(min = 48.dp)) { Text("بستن") } })
}
