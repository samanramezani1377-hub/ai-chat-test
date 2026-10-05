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

private val Ink = Color(0xFF1B1B22)
private val Paper = Color(0xFFF7F5F0)
private val Violet = Color(0xFF6552E8)
private val Mint = Color(0xFF1E8A78)
private val Coral = Color(0xFFE46B4D)

@Composable
internal fun AppTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = lightColorScheme(
            primary = Violet, onPrimary = Color.White,
            primaryContainer = Color(0xFFE7E2FF), onPrimaryContainer = Color(0xFF21175E),
            secondary = Mint, onSecondary = Color.White,
            secondaryContainer = Color(0xFFD8F2EA), onSecondaryContainer = Color(0xFF073B31),
            tertiary = Coral, onTertiary = Color.White,
            tertiaryContainer = Color(0xFFFFDDD4), onTertiaryContainer = Color(0xFF4D180C),
            background = Paper, onBackground = Ink,
            surface = Color(0xFFFFFCF8), onSurface = Ink,
            surfaceVariant = Color(0xFFECE9E3), onSurfaceVariant = Color(0xFF66636C),
            outline = Color(0xFFC9C5CE), error = Color(0xFFB3261E),
            errorContainer = Color(0xFFFFDAD6), onErrorContainer = Color(0xFF410002),
        ),
        shapes = Shapes(
            small = RoundedCornerShape(14.dp), medium = RoundedCornerShape(18.dp),
            large = RoundedCornerShape(24.dp), extraLarge = RoundedCornerShape(30.dp),
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
    Surface(
        Modifier.fillMaxWidth().padding(horizontal = 14.dp, vertical = 10.dp),
        shape = RoundedCornerShape(24.dp), color = MaterialTheme.colorScheme.surface, shadowElevation = 1.dp,
    ) {
        Row(Modifier.fillMaxWidth().heightIn(min = 64.dp).padding(horizontal = 6.dp), verticalAlignment = Alignment.CenterVertically) {
            IconButton(onClick = onSidebar, Modifier.size(48.dp).semantics { contentDescription = "باز کردن منو" }) {
                Icon(Icons.Default.Menu, "منو")
            }
            Column(Modifier.weight(1f).padding(horizontal = 8.dp)) {
                Text(title, style = MaterialTheme.typography.titleLarge)
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(Modifier.size(7.dp).background(if (model != null) Mint else MaterialTheme.colorScheme.outline, RoundedCornerShape(50)))
                    Spacer(Modifier.width(6.dp))
                    Text(if (model != null) "مدل محلی آماده است" else "مدل محلی انتخاب نشده", style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
            TextButton(onClick = onRuntime, Modifier.heightIn(min = 48.dp), contentPadding = PaddingValues(horizontal = 10.dp)) {
                Column(horizontalAlignment = Alignment.End) {
                    Text(status, style = MaterialTheme.typography.labelLarge)
                    Text("llama.cpp · OpenCL", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
            IconButton(onClick = onQuick, Modifier.size(48.dp).semantics { contentDescription = "گزینه‌های بیشتر" }) {
                Icon(Icons.Default.MoreVert, "گزینه‌های بیشتر")
            }
        }
    }
}

@Composable
internal fun QuickMenu(onDiagnostics: () -> Unit, onSettings: () -> Unit) {
    Box(Modifier.fillMaxSize().padding(top = 82.dp, start = 14.dp), contentAlignment = Alignment.TopStart) {
        Surface(Modifier.widthIn(max = 250.dp), shape = RoundedCornerShape(20.dp), color = MaterialTheme.colorScheme.surface, shadowElevation = 10.dp) {
            Column(Modifier.padding(8.dp)) {
                Text("دسترسی سریع", Modifier.padding(horizontal = 12.dp, vertical = 8.dp), style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
                TextButton(onClick = onDiagnostics, Modifier.fillMaxWidth().heightIn(min = 48.dp)) { Icon(Icons.Default.BugReport, null); Spacer(Modifier.width(10.dp)); Text("عیب‌یابی", Modifier.fillMaxWidth()) }
                TextButton(onClick = onSettings, Modifier.fillMaxWidth().heightIn(min = 48.dp)) { Icon(Icons.Default.Settings, null); Spacer(Modifier.width(10.dp)); Text("تنظیمات", Modifier.fillMaxWidth()) }
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
        Spacer(Modifier.fillMaxSize().background(Color.Black.copy(alpha = .28f)).clickable(onClick = onClose).semantics { contentDescription = "بستن منو" })
        Surface(
            Modifier.fillMaxHeight().fillMaxWidth(.90f).align(AbsoluteAlignment.CenterRight).zIndex(10f),
            shape = RoundedCornerShape(topStart = 30.dp, bottomStart = 30.dp),
            color = MaterialTheme.colorScheme.background, shadowElevation = 16.dp,
        ) {
            LazyColumn(Modifier.fillMaxSize().padding(18.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                item {
                    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                        Column(Modifier.weight(1f)) {
                            Text("AI Chat", style = MaterialTheme.typography.headlineSmall)
                            Text("دستیار محلی روی دستگاه", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                        IconButton(onClick = onClose, Modifier.size(48.dp)) { Icon(Icons.Default.Close, "بستن") }
                    }
                    Spacer(Modifier.height(10.dp))
                }
                item {
                    Button(onClick = onNew, Modifier.fillMaxWidth().heightIn(min = 52.dp), shape = RoundedCornerShape(18.dp)) {
                        Icon(Icons.Default.Add, null); Spacer(Modifier.width(8.dp)); Text("گفت‌وگوی جدید")
                    }
                }
                item { DrawerSection("برنامه") }
                item { DrawerItem(Icons.Default.ChatBubbleOutline, "گفت‌وگو", destination == AppDestination.CHAT) { onDestination(AppDestination.CHAT) } }
                item { DrawerItem(Icons.Default.WorkOutline, "فضای کار", destination == AppDestination.WORK || destination == AppDestination.WORKSPACE) { onDestination(AppDestination.WORK) } }
                item { DrawerItem(Icons.Default.Settings, "تنظیمات", destination == AppDestination.SETTINGS) { onDestination(AppDestination.SETTINGS) } }
                item { DrawerItem(Icons.Default.BugReport, "عیب‌یابی", destination == AppDestination.DIAGNOSTICS) { onDestination(AppDestination.DIAGNOSTICS) } }
                item { DrawerSection("گفت‌وگوهای اخیر") }
                if (conversations.isEmpty()) item { Text("هنوز گفت‌وگویی ذخیره نشده است.", Modifier.padding(12.dp), color = MaterialTheme.colorScheme.onSurfaceVariant) }
                items(conversations, key = { it.id }) { record ->
                    RecentConversation(record, record.id == current?.id, { onOpen(record) }, { onRename(record) }, { onDelete(record) })
                }
                if (canShowMore) item { TextButton(onClick = onMore, Modifier.fillMaxWidth().heightIn(min = 48.dp)) { Text("نمایش گفت‌وگوهای بیشتر") } }
                item { DrawerSection("اطلاعات") }
                item { DrawerItem(Icons.Default.Info, "درباره برنامه", destination == AppDestination.ABOUT) { onDestination(AppDestination.ABOUT) } }
            }
        }
    }
}

@Composable private fun DrawerSection(text: String) {
    Text(text, Modifier.padding(start = 12.dp, top = 14.dp, bottom = 4.dp), style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
}

@Composable private fun DrawerItem(icon: androidx.compose.ui.graphics.vector.ImageVector, text: String, active: Boolean, onClick: () -> Unit) {
    Surface(Modifier.fillMaxWidth().clickable(onClick = onClick), shape = RoundedCornerShape(16.dp), color = if (active) MaterialTheme.colorScheme.primaryContainer else Color.Transparent) {
        Row(Modifier.fillMaxWidth().heightIn(min = 52.dp).padding(horizontal = 12.dp), verticalAlignment = Alignment.CenterVertically) {
            Icon(icon, null, tint = if (active) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant)
            Spacer(Modifier.width(12.dp))
            Text(text, style = if (active) MaterialTheme.typography.titleSmall else MaterialTheme.typography.bodyLarge)
        }
    }
}

@OptIn(ExperimentalFoundationApi::class)
@Composable private fun RecentConversation(record: ConversationRecord, active: Boolean, onOpen: () -> Unit, onRename: () -> Unit, onDelete: () -> Unit) {
    var menu by remember { mutableStateOf(false) }
    val time = remember(record.updatedAtEpochMs) { DateFormat.getTimeInstance(DateFormat.SHORT).format(Date(record.updatedAtEpochMs)) }
    Box(Modifier.fillMaxWidth()) {
        Surface(
            Modifier.fillMaxWidth().combinedClickable(onClick = onOpen, onLongClick = { menu = true }),
            shape = RoundedCornerShape(16.dp), color = if (active) MaterialTheme.colorScheme.surfaceVariant else Color.Transparent,
        ) {
            Row(Modifier.padding(horizontal = 12.dp, vertical = 10.dp).heightIn(min = 52.dp), verticalAlignment = Alignment.CenterVertically) {
                Box(Modifier.size(9.dp).background(if (active) Violet else MaterialTheme.colorScheme.outline, RoundedCornerShape(50)))
                Spacer(Modifier.width(10.dp))
                Column(Modifier.weight(1f)) {
                    Text(record.title, maxLines = 1, style = MaterialTheme.typography.bodyLarge)
                    Text(record.messageCount.toString() + " پیام", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
                Text(time, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
        DropdownMenu(expanded = menu, onDismissRequest = { menu = false }) {
            DropdownMenuItem(text = { Text("باز کردن") }, onClick = { menu = false; onOpen() })
            DropdownMenuItem(text = { Text("تغییر نام") }, onClick = { menu = false; onRename() })
            DropdownMenuItem(text = { Text("حذف", color = MaterialTheme.colorScheme.error) }, onClick = { menu = false; onDelete() })
        }
    }
}

@Composable internal fun UndoBar(onUndo: () -> Unit, onDismiss: () -> Unit, onExpire: () -> Unit) {
    var remaining by remember { mutableIntStateOf(10) }
    LaunchedEffect(Unit) { while (remaining > 0) { delay(1000); remaining-- }; onExpire() }
    Box(Modifier.fillMaxSize().padding(14.dp), contentAlignment = Alignment.BottomCenter) {
        Surface(Modifier.fillMaxWidth().widthIn(max = 640.dp), shape = RoundedCornerShape(18.dp), color = MaterialTheme.colorScheme.inverseSurface, shadowElevation = 8.dp) {
            Row(Modifier.padding(8.dp), verticalAlignment = Alignment.CenterVertically) {
                Text("گفت‌وگو حذف شد", Modifier.weight(1f), color = MaterialTheme.colorScheme.inverseOnSurface)
                TextButton(onClick = onUndo, Modifier.heightIn(min = 48.dp)) { Text("بازگردانی", color = MaterialTheme.colorScheme.inversePrimary) }
                Text(remaining.toString() + " ثانیه", style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.inverseOnSurface)
                IconButton(onClick = onDismiss, Modifier.size(48.dp)) { Icon(Icons.Default.Close, "بستن", tint = MaterialTheme.colorScheme.inverseOnSurface) }
            }
        }
    }
}

@Composable internal fun RenameDialog(initial: String, onDismiss: () -> Unit, onConfirm: (String) -> Unit) {
    var value by remember(initial) { mutableStateOf(initial) }
    AlertDialog(onDismissRequest = onDismiss, title = { Text("نام گفت‌وگو") }, text = { TextField(value, { value = it }, singleLine = true, label = { Text("عنوان") }) },
        confirmButton = { TextButton(onClick = { onConfirm(value.trim()) }, enabled = value.isNotBlank(), modifier = Modifier.heightIn(min = 48.dp)) { Text("ذخیره") } },
        dismissButton = { TextButton(onClick = onDismiss, modifier = Modifier.heightIn(min = 48.dp)) { Text("انصراف") } })
}

@Composable internal fun RuntimeDetailsDialog(status: String, model: ModelDescriptor?, runtimeName: String, runtimeVersion: String, backend: String?, onDismiss: () -> Unit) {
    AlertDialog(onDismissRequest = onDismiss, title = { Text("وضعیت اجرای محلی") }, text = {
        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            StatusLine("وضعیت", status)
            StatusLine("مدل", model?.displayName ?: "مدلی فعال نیست")
            StatusLine("Runtime", runtimeName)
            StatusLine("نسخه", runtimeVersion)
            StatusLine("Backend", backend ?: "OpenCL")
        }
    }, confirmButton = { TextButton(onClick = onDismiss, Modifier.heightIn(min = 48.dp)) { Text("بستن") } })
}

@Composable private fun StatusLine(label: String, value: String) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) { Text(label, color = MaterialTheme.colorScheme.onSurfaceVariant); Text(value, style = MaterialTheme.typography.titleSmall) }
}
