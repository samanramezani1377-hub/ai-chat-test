package com.samanramezani.aichattest.screen

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import com.samanramezani.aichattest.AIChatApplication
import com.samanramezani.aichattest.InferenceSettingsAccess
import com.woogit.aicore.agent.AgentEvent
import com.woogit.aicore.conversation.ConversationMessage
import com.woogit.aicore.domain.InferenceSettings
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private data class FloatingChatMessage(val isUser: Boolean, val text: String)

/** A compact chat surface hosted directly in the overlay window; it never launches MainActivity. */
@Composable
internal fun FloatingAssistantPopup(onClose: () -> Unit) {
    val context = androidx.compose.ui.platform.LocalContext.current
    val application = context.applicationContext as AIChatApplication
    val container = remember { application.container }
    val scope = rememberCoroutineScope()
    val keyboard = LocalSoftwareKeyboardController.current
    var conversationId by remember { mutableStateOf<String?>(null) }
    var messages by remember { mutableStateOf(emptyList<FloatingChatMessage>()) }
    var draft by remember { mutableStateOf("") }
    var sending by remember { mutableStateOf(false) }
    var status by remember { mutableStateOf("در حال آماده‌سازی دستیار…") }
    val listState = rememberLazyListState()

    LaunchedEffect(Unit) {
        val prefs = context.getSharedPreferences("screen_assistant", android.content.Context.MODE_PRIVATE)
        val savedId = prefs.getString("floating_chat_conversation_id", null)
        val record = withContext(Dispatchers.IO) {
            if (container.providerStore?.load()?.mode != com.samanramezani.aichattest.providers.InferenceProviderMode.CLOUDFLARE) {
                container.modelManager?.restoreActive()
            }
            val existing = savedId?.let { id -> container.conversationHistory.recent(500).firstOrNull { it.id == id } }
            existing ?: container.conversationHistory.create("گفت‌وگوی شناور دینا")
        }
        conversationId = record.id
        if (record.id != savedId) prefs.edit().putString("floating_chat_conversation_id", record.id).apply()
        messages = withContext(Dispatchers.IO) {
            container.conversationHistory.messages(record.id).map {
                FloatingChatMessage(it.role == ConversationMessage.Role.USER, it.content)
            }
        }
        status = if (container.modelManager?.activeModel() is com.woogit.aicore.domain.ModelResult.Success || container.isCloudflareConfigured()) {
            "آماده‌ام؛ می‌توانی درباره صفحه فعلی هم سؤال کنی."
        } else {
            "برای پاسخ‌گویی، ابتدا یک مدل را در برنامه فعال کن."
        }
    }

    LaunchedEffect(messages.size, sending) {
        if (messages.isNotEmpty()) listState.animateScrollToItem(messages.lastIndex)
    }

    fun sendMessage() {
        val text = draft.trim()
        val id = conversationId
        if (text.isEmpty() || id == null || sending) return
        draft = ""
        sending = true
        status = "در حال پاسخ‌گویی…"
        messages = messages + FloatingChatMessage(true, text) + FloatingChatMessage(false, "")
        val responseIndex = messages.lastIndex
        scope.launch {
            try {
                val session = container.createAgentSession(
                    conversationId = id,
                    includeAgentTools = false,
                    eventSink = { event ->
                        if (event is AgentEvent.Token) {
                            withContext(Dispatchers.Main.immediate) {
                                messages = messages.toMutableList().also { list ->
                                    if (responseIndex in list.indices) list[responseIndex] = list[responseIndex].copy(text = list[responseIndex].text + event.value)
                                }
                            }
                        }
                    },
                ) ?: error("Agent Session در دسترس نیست.")
                val result = withContext(Dispatchers.Default) {
                    run { val inferenceSettings = InferenceSettingsAccess.load(context); session.send(text, inferenceSettings, requestedRecentMessages = inferenceSettings.recentMessages) }
                }
                messages = messages.toMutableList().also { list ->
                    if (responseIndex in list.indices) list[responseIndex] = list[responseIndex].copy(text = result.generation.text)
                }
                status = "آماده"
            } catch (t: Throwable) {
                val reason = t.message?.takeIf { it.isNotBlank() } ?: "پاسخ‌گویی ناموفق بود."
                messages = messages.toMutableList().also { list ->
                    if (responseIndex in list.indices) list[responseIndex] = list[responseIndex].copy(text = "خطا: $reason")
                }
                status = "خطا در پاسخ‌گویی"
            } finally {
                sending = false
            }
        }
        keyboard?.hide()
    }

    MaterialTheme {
        Surface(
            modifier = Modifier.fillMaxSize(),
            shape = RoundedCornerShape(22.dp),
            color = MaterialTheme.colorScheme.surface,
            tonalElevation = 8.dp,
            shadowElevation = 14.dp,
        ) {
            Column(Modifier.fillMaxSize().padding(12.dp)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween,
                ) {
                    Column(Modifier.weight(1f)) {
                        Text("دینـا", style = MaterialTheme.typography.titleMedium)
                        Text(status, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                    TextButton(onClick = onClose) { Text("بستن") }
                }
                HorizontalDivider()
                LazyColumn(
                    modifier = Modifier.weight(1f).fillMaxWidth().padding(vertical = 8.dp),
                    state = listState,
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    if (messages.isEmpty()) {
                        item {
                            Text(
                                "سلام! بدون خارج‌شدن از برنامه‌ای که باز است با من گفتگو کن. برای تحلیل صفحه بپرس «این صفحه چی می‌گه؟» یا «اینجا باید چی کار کنم؟»",
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                    }
                    itemsIndexed(messages) { _, message ->
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = if (message.isUser) Arrangement.End else Arrangement.Start,
                        ) {
                            Surface(
                                shape = RoundedCornerShape(14.dp),
                                color = if (message.isUser) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.surfaceVariant,
                                modifier = Modifier.widthIn(max = 280.dp),
                            ) {
                                Text(
                                    message.text.ifBlank { if (sending && !message.isUser) "…" else "" },
                                    modifier = Modifier.padding(horizontal = 12.dp, vertical = 9.dp),
                                    style = MaterialTheme.typography.bodyMedium,
                                    color = if (message.isUser) MaterialTheme.colorScheme.onPrimaryContainer else MaterialTheme.colorScheme.onSurfaceVariant,
                                )
                            }
                        }
                    }
                }
                Row(verticalAlignment = Alignment.Bottom, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedTextField(
                        value = draft,
                        onValueChange = { draft = it },
                        modifier = Modifier.weight(1f),
                        placeholder = { Text("پیامت را بنویس…") },
                        maxLines = 4,
                        enabled = !sending,
                        keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
                        keyboardActions = KeyboardActions(onSend = { sendMessage() }),
                    )
                    Button(onClick = { sendMessage() }, enabled = !sending && draft.isNotBlank() && conversationId != null) {
                        Text("ارسال")
                    }
                }
            }
        }
    }
}
