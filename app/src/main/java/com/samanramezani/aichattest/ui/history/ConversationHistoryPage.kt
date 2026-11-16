package com.samanramezani.aichattest.ui.history

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.ChatBubbleOutline
import androidx.compose.material.icons.filled.DeleteOutline
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.woogit.aicore.conversation.ConversationRecord
import java.text.DateFormat
import java.util.Date

@Composable
internal fun ConversationHistoryPage(
    conversations: List<ConversationRecord>,
    currentId: String?,
    searchIndex: Map<String, String> = emptyMap(),
    previews: Map<String, String> = emptyMap(),
    onOpen: (ConversationRecord) -> Unit,
    onNew: () -> Unit,
    onRename: (ConversationRecord) -> Unit,
    onDelete: (ConversationRecord) -> Unit,
    onMore: () -> Unit,
    canShowMore: Boolean,
) {
    var query by remember { mutableStateOf("") }
    val filtered = remember(conversations, query, searchIndex) {
        if (query.isBlank()) conversations
        else conversations.filter { record ->
            record.title.contains(query.trim(), ignoreCase = true) ||
                searchIndex[record.id].orEmpty().contains(query.trim(), ignoreCase = true)
        }
    }

    Column(Modifier.fillMaxSize().padding(horizontal = 16.dp)) {
        Row(
            Modifier.fillMaxWidth().padding(top = 12.dp, bottom = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(Modifier.weight(1f)) {
                Text("تاریخچه گفتگوها", style = MaterialTheme.typography.headlineSmall)
                Text(
                    "${conversations.size} گفت‌وگو در فهرست بارگذاری‌شده",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            FilledTonalButton(onClick = onNew) {
                Icon(Icons.Default.Add, contentDescription = null)
                Spacer(Modifier.width(6.dp))
                Text("گفتگوی جدید")
            }
        }
        OutlinedTextField(
            value = query,
            onValueChange = { query = it },
            modifier = Modifier.fillMaxWidth(),
            singleLine = true,
            shape = RoundedCornerShape(16.dp),
            leadingIcon = { Icon(Icons.Default.History, contentDescription = null) },
            placeholder = { Text("جستجو در عنوان گفتگوها") },
        )
        Spacer(Modifier.height(10.dp))
        if (filtered.isEmpty()) {
            Box(Modifier.weight(1f).fillMaxWidth(), contentAlignment = Alignment.Center) {
                Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Icon(Icons.Default.ChatBubbleOutline, contentDescription = null, tint = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.size(38.dp))
                    Text(if (query.isBlank()) "هنوز گفتگویی ذخیره نشده است." else "گفتگویی با این عنوان پیدا نشد.")
                    if (query.isNotBlank()) TextButton(onClick = { query = "" }) { Text("پاک‌کردن جستجو") }
                }
            }
        } else {
            LazyColumn(
                modifier = Modifier.weight(1f).fillMaxWidth(),
                contentPadding = PaddingValues(bottom = 28.dp, top = 4.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                items(filtered, key = { it.id }) { record ->
                    HistoryRow(
                        record = record,
                        active = record.id == currentId,
                        preview = previews[record.id].orEmpty(),
                        onOpen = { onOpen(record) },
                        onRename = { onRename(record) },
                        onDelete = { onDelete(record) },
                    )
                }
                if (canShowMore && query.isBlank()) {
                    item {
                        OutlinedButton(onClick = onMore, modifier = Modifier.fillMaxWidth()) {
                            Text("بارگذاری گفتگوهای بیشتر")
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun HistoryRow(
    record: ConversationRecord,
    active: Boolean,
    preview: String,
    onOpen: () -> Unit,
    onRename: () -> Unit,
    onDelete: () -> Unit,
) {
    var menuExpanded by remember { mutableStateOf(false) }
    val timestamp = remember(record.updatedAtEpochMs) {
        DateFormat.getDateTimeInstance(DateFormat.MEDIUM, DateFormat.SHORT)
            .format(Date(record.updatedAtEpochMs))
    }
    Surface(
        modifier = Modifier.fillMaxWidth().clickable(onClick = onOpen),
        shape = RoundedCornerShape(18.dp),
        color = if (active) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.surfaceVariant.copy(alpha = .55f),
        tonalElevation = if (active) 1.dp else 0.dp,
    ) {
        Row(
            Modifier.fillMaxWidth().padding(start = 14.dp, end = 6.dp, top = 10.dp, bottom = 10.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text(record.title, style = MaterialTheme.typography.titleSmall, maxLines = 1)
                if (preview.isNotBlank()) Text(preview, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant, maxLines = 2)
                Text(
                    "${record.messageCount} پیام  ·  $timestamp",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            Box {
                IconButton(onClick = { menuExpanded = true }) {
                    Icon(Icons.Default.MoreVert, contentDescription = "عملیات گفتگو")
                }
                DropdownMenu(expanded = menuExpanded, onDismissRequest = { menuExpanded = false }) {
                    DropdownMenuItem(
                        text = { Text("باز کردن") },
                        onClick = { menuExpanded = false; onOpen() },
                    )
                    DropdownMenuItem(
                        text = { Text("تغییر نام") },
                        leadingIcon = { Icon(Icons.Default.Edit, contentDescription = null) },
                        onClick = { menuExpanded = false; onRename() },
                    )
                    DropdownMenuItem(
                        text = { Text("حذف", color = MaterialTheme.colorScheme.error) },
                        leadingIcon = { Icon(Icons.Default.DeleteOutline, contentDescription = null, tint = MaterialTheme.colorScheme.error) },
                        onClick = { menuExpanded = false; onDelete() },
                    )
                }
            }
        }
    }
}
