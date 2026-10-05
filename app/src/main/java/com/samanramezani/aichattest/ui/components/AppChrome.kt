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
import androidx.compose.ui.AbsoluteAlignment
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import androidx.compose.ui.zIndex
import com.samanramezani.aichattest.ui.navigation.AppDestination
import com.woogit.aicore.domain.ModelDescriptor
import com.woogit.aicore.conversation.ConversationRecord
import kotlinx.coroutines.delay
import java.text.DateFormat
import java.util.Date

private val Ink = Color(0xFFEDEDF3)
private val Muted = Color(0xFF9B9BA8)
private val Canvas = Color(0xFF101116)
private val Surface = Color(0xFF171820)
private val Surface2 = Color(0xFF20222C)
private val Accent = Color(0xFF9B8CFF)
private val Accent2 = Color(0xFF67D6C4)
private val Warning = Color(0xFFFFB86B)

@Composable
internal fun AppTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = darkColorScheme(
            primary = Accent, onPrimary = Color(0xFF18142E),
            primaryContainer = Color(0xFF302B55), onPrimaryContainer = Color(0xFFE8E2FF),
            secondary = Accent2, onSecondary = Color(0xFF08201C),
            secondaryContainer = Color(0xFF193D38), onSecondaryContainer = Color(0xFFD0FFF6),
            tertiary = Warning, onTertiary = Color(0xFF2A1806),
            background = Canvas, onBackground = Ink,
            surface = Surface, onSurface = Ink,
            surfaceVariant = Surface2, onSurfaceVariant = Muted,
            outline = Color(0xFF41434E), error = Color(0xFFFF716D),
            errorContainer = Color(0xFF4A2022), onErrorContainer = Color(0xFFFFDAD8),
        ),
        shapes = Shapes(
            small = RoundedCornerShape(12.dp), medium = RoundedCornerShape(16.dp),
            large = RoundedCornerShape(22.dp), extraLarge = RoundedCornerShape(28.dp),
        ),
        content = content,
    )
}

@Composable
internal fun AppHeader(
    destination: AppDestination, status: String, model: ModelDescriptor?,
    onDestination: (AppDestination) -> Unit, onQuick: () -> Unit,
    onSidebar: () -> Unit, onRuntime: () -> Unit,
) {
    val title = when (destination) {
        AppDestination.CHAT -> "گفت‌وگو"
        AppDestination.WORK, AppDestination.WORKSPACE -> "فضای کار"
        AppDestination.DIAGNOSTICS -> "عیب‌یابی"
        AppDestination.SETTINGS -> "تنظیمات"
        AppDestination.ABOUT -> "درباره"
    }
    Row(Modifier.fillMaxWidth().height(70.dp).padding(horizontal = 12.dp), verticalAlignment = Alignment.CenterVertically) {
        IconButton(onClick = onSidebar, Modifier.size(48.dp).semantics { contentDescription = "باز کردن منو" }) {
            Icon(Icons.Default.Menu, "منو")
        }
        Column(Modifier.weight(1f).padding(horizontal = 8.dp)) {
            Text(title, style = MaterialTheme.typography.titleLarge)
            Text(if (destination == AppDestination.CHAT) "اجرای محلی روی دستگاه" else "AI Chat · local runtime",
                style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        Surface(
            Modifier.clickable(onClick = onRuntime).heightIn(min = 44.dp),
            shape = RoundedCornerShape(14.dp),
            color = if (model != null) MaterialTheme.colorScheme.secondaryContainer else MaterialTheme.colorScheme.surfaceVariant,
        ) {
            Row(Modifier.padding(horizontal = 12.dp), verticalAlignment = Alignment.CenterVertically) {
                Box(Modifier.size(7.dp).background(if (model != null) Accent2 else Warning, RoundedCornerShape(50)))
                Spacer(Modifier.width(7.dp))
                Column {
                    Text(if (model != null) "آماده" else "بدون مدل", style = MaterialTheme.typography.labelLarge)
                    Text("OpenCL", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
        }
        IconButton(onClick = onQuick, Modifier.size(48.dp).semantics { contentDescription = "گزینه‌های بیشتر" }) {
            Icon(Icons.Default.MoreVert, "بیشتر")
        }
    }
}

@Composable
internal fun AppQuickMenu(onDiagnostics: () -> Unit, onSettings: () -> Unit) {
    Box(Modifier.fillMaxSize().padding(top = 66.dp, start = 12.dp), contentAlignment = Alignment.TopStart) {
        Surface(Modifier.width(230.dp), shape = RoundedCornerShape(18.dp), color = MaterialTheme.colorScheme.surface, tonalElevation = 8.dp) {
            Column(Modifier.padding(6.dp)) {
                DropdownMenuItem(text = { Text("عیب‌یابی") }, leadingIcon = { Icon(Icons.Default.BugReport, null) }, onClick = onDiagnostics)
                DropdownMenuItem(text = { Text("تنظیمات") }, leadingIcon = { Icon(Icons.Default.Settings, null) }, onClick = onSettings)
            }
        }
    }
}

@Composable
internal fun Sidebar(
    current: ConversationRecord?, conversations: List<ConversationRecord>, destination: AppDestination,
    onClose: () -> Unit, onDestination: (AppDestination) -> Unit, onNew: () -> Unit,
    onOpen: (ConversationRecord) -> Unit, onRename: (ConversationRecord) -> Unit,
    onDelete: (ConversationRecord) -> Unit, onMore: () -> Unit, canShowMore: Boolean,
) {
    Box(Modifier.fillMaxSize()) {
        Spacer(Modifier.fillMaxSize().background(Color.Black.copy(alpha = .62f)).clickable(onClick = onClose))
        Surface(
            Modifier.fillMaxHeight().fillMaxWidth(.88f).align(AbsoluteAlignment.CenterRight).zIndex(10f),
            shape = RoundedCornerShape(topStart = 28.dp, bottomStart = 28.dp),
            color = MaterialTheme.colorScheme.surface, tonalElevation = 10.dp,
        ) {
            LazyColumn(Modifier.fillMaxSize().padding(16.dp), contentPadding = PaddingValues(bottom = 28.dp), verticalArrangement = Arrangement.spacedBy(5.dp)) {
                item {
                    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                        Column(Modifier.weight(1f)) {
                            Text("AI Chat", style = MaterialTheme.typography.headlineSmall)
                            Text("هوش مصنوعی، بدون سرور", color = MaterialTheme.colorScheme.onSurfaceVariant, style = MaterialTheme.typography.bodySmall)
                        }
                        IconButton(onClick = onClose) { Icon(Icons.Default.Close, "بستن") }
                    }
                }
                item {
                    Button(onClick = onNew, Modifier.fillMaxWidth().height(52.dp), shape = RoundedCornerShape(16.dp)) {
                        Icon(Icons.Default.Add, null); Spacer(Modifier.width(8.dp)); Text("گفت‌وگوی جدید")
                    }
                }
                item { DrawerSection("برنامه") }
                item { DrawerItem(Icons.Default.ChatBubbleOutline, "گفت‌وگو", destination == AppDestination.CHAT) { onDestination(AppDestination.CHAT) } }
                item { DrawerItem(Icons.Default.WorkOutline, "فضای کار", destination == AppDestination.WORK || destination == AppDestination.WORKSPACE) { onDestination(AppDestination.WORK) } }
                item { DrawerItem(Icons.Default.Settings, "تنظیمات", destination == AppDestination.SETTINGS) { onDestination(AppDestination.SETTINGS) } }
                item { DrawerItem(Icons.Default.BugReport, "عیب‌یابی", destination == AppDestination.DIAGNOSTICS) { onDestination(AppDestination.DIAGNOSTICS) } }
                item { DrawerSection("اخیر") }
                if (conversations.isEmpty()) item { Text("هنوز گفت‌وگویی وجود ندارد.", Modifier.padding(12.dp), color = MaterialTheme.colorScheme.onSurfaceVariant) }
                items(conversations, key = { it.id }) { record -> RecentConversation(record, record.id == current?.id, onOpen, onRename, onDelete) }
                if (canShowMore) item { TextButton(onClick = onMore, Modifier.fillMaxWidth()) { Text("گفت‌وگوهای بیشتر") } }
                item { DrawerSection("اطلاعات") }
                item { DrawerItem(Icons.Default.Info, "درباره برنامه", destination == AppDestination.ABOUT) { onDestination(AppDestination.ABOUT) } }
            }
        }
    }
}

@Composable private fun DrawerSection(text: String) {
    Text(text, Modifier.padding(start = 12.dp, top = 12.dp, bottom = 3.dp), style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
}

@Composable private fun DrawerItem(icon: androidx.compose.ui.graphics.vector.ImageVector, text: String, active: Boolean, onClick: () -> Unit) {
    Surface(Modifier.fillMaxWidth().clickable(onClick = onClick), shape = RoundedCornerShape(14.dp), color = if (active) MaterialTheme.colorScheme.primaryContainer else Color.Transparent) {
        Row(Modifier.height(50.dp).padding(horizontal = 12.dp), verticalAlignment = Alignment.CenterVertically) {
            Icon(icon, null, tint = if (active) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant)
            Spacer(Modifier.width(12.dp))
            Text(text, style = if (active) MaterialTheme.typography.titleSmall else MaterialTheme.typography.bodyLarge)
        }
    }
}

@OptIn(ExperimentalFoundationApi::class)
@Composable private fun RecentConversation(record: ConversationRecord, active: Boolean, onOpen: (ConversationRecord) -> Unit, onRename: (ConversationRecord) -> Unit, onDelete: (ConversationRecord) -> Unit) {
    var menu by remember { mutableStateOf(false) }
    val time = remember(record.updatedAtEpochMs) { DateFormat.getTimeInstance(DateFormat.SHORT).format(Date(record.updatedAtEpochMs)) }
    Box(Modifier.fillMaxWidth()) {
        Surface(Modifier.fillMaxWidth().combinedClickable(onClick = { onOpen(record) }, onLongClick = { menu = true }), shape = RoundedCornerShape(14.dp), color = if (active) MaterialTheme.colorScheme.surfaceVariant else Color.Transparent) {
            Row(Modifier.padding(horizontal = 12.dp, vertical = 9.dp), verticalAlignment = Alignment.CenterVertically) {
                Box(Modifier.size(7.dp).background(if (active) Accent else MaterialTheme.colorScheme.outline, RoundedCornerShape(50)))
                Spacer(Modifier.width(9.dp))
                Column(Modifier.weight(1f)) {
                    Text(record.title, maxLines = 1, style = MaterialTheme.typography.bodyLarge)
                    Text(record.messageCount.toString() + " پیام · " + time, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
        }
        DropdownMenu(expanded = menu, onDismissRequest = { menu = false }) {
            DropdownMenuItem(text = { Text("باز کردن") }, onClick = { menu = false; onOpen(record) })
            DropdownMenuItem(text = { Text("تغییر نام") }, onClick = { menu = false; onRename(record) })
            DropdownMenuItem(text = { Text("حذف", color = MaterialTheme.colorScheme.error) }, onClick = { menu = false; onDelete(record) })
        }
    }
}

@Composable internal fun UndoBar(onUndo: () -> Unit, onDismiss: () -> Unit, onExpire: () -> Unit) {
    var remaining by remember { mutableIntStateOf(10) }
    LaunchedEffect(Unit) { while (remaining > 0) { delay(1000); remaining-- }; onExpire() }
    Box(Modifier.fillMaxSize().padding(12.dp), contentAlignment = Alignment.BottomCenter) {
        Surface(Modifier.fillMaxWidth().widthIn(max = 620.dp), shape = RoundedCornerShape(16.dp), color = MaterialTheme.colorScheme.inverseSurface) {
            Row(Modifier.padding(6.dp), verticalAlignment = Alignment.CenterVertically) {
                Text("گفت‌وگو حذف شد", Modifier.weight(1f), color = MaterialTheme.colorScheme.inverseOnSurface)
                TextButton(onClick = onUndo) { Text("بازگردانی", color = MaterialTheme.colorScheme.inversePrimary) }
                Text(remaining.toString(), style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.inverseOnSurface)
                IconButton(onClick = onDismiss) { Icon(Icons.Default.Close, "بستن", tint = MaterialTheme.colorScheme.inverseOnSurface) }
            }
        }
    }
}

@Composable internal fun RenameDialog(initial: String, onDismiss: () -> Unit, onConfirm: (String) -> Unit) {
    var value by remember(initial) { mutableStateOf(initial) }
    AlertDialog(onDismissRequest = onDismiss, title = { Text("نام گفت‌وگو") }, text = { OutlinedTextField(value, { value = it }, singleLine = true, label = { Text("عنوان") }, modifier = Modifier.fillMaxWidth()) }, confirmButton = { TextButton(onClick = { onConfirm(value.trim()) }, enabled = value.isNotBlank()) { Text("ذخیره") } }, dismissButton = { TextButton(onClick = onDismiss) { Text("انصراف") } })
}

@Composable internal fun RuntimeDetailsDialog(status: String, model: ModelDescriptor?, runtimeName: String, runtimeVersion: String, backend: String?, onDismiss: () -> Unit) {
    AlertDialog(onDismissRequest = onDismiss, title = { Text("اجرای محلی") }, text = { Column(verticalArrangement = Arrangement.spacedBy(10.dp)) { RuntimeLine("وضعیت", status); RuntimeLine("مدل", model?.displayName ?: "مدلی فعال نیست"); RuntimeLine("Runtime", runtimeName); RuntimeLine("نسخه", runtimeVersion); RuntimeLine("Backend", backend ?: "OpenCL") } }, confirmButton = { TextButton(onClick = onDismiss) { Text("بستن") } })
}

@Composable private fun RuntimeLine(label: String, value: String) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) { Text(label, color = MaterialTheme.colorScheme.onSurfaceVariant); Text(value, style = MaterialTheme.typography.titleSmall) }
}
