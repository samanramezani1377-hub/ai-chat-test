package com.samanramezani.aichattest

import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.unit.LayoutDirection
import androidx.compose.ui.platform.LocalLayoutDirection
import com.samanramezani.aichattest.ui.SettingsScreen
import com.samanramezani.aichattest.ui.about.AboutPage
import com.samanramezani.aichattest.ui.chat.ChatPage
import com.samanramezani.aichattest.ui.components.*
import com.samanramezani.aichattest.ui.diagnostics.DiagnosticsPage
import com.samanramezani.aichattest.ui.navigation.AppDestination
import com.samanramezani.aichattest.ui.state.*
import com.samanramezani.aichattest.ui.workspace.WorkspacePage
import com.woogit.aicore.agent.AgentEvent
import com.woogit.aicore.domain.ChatMessage
import com.woogit.aicore.domain.InferenceSettings
import com.woogit.aicore.domain.ModelDescriptor
import com.woogit.aicore.domain.ModelResult
import com.woogit.aicore.conversation.ConversationRecord
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.UUID

@Composable
internal fun MainScreen(container: AppContainer) {
    val history = remember { container.conversationHistory }
    val manager = remember { container.modelManager }
    val scope = rememberCoroutineScope()
    var destination by remember { mutableStateOf(AppDestination.CHAT) }
    var current by remember { mutableStateOf<ConversationRecord?>(null) }
    var conversations by remember { mutableStateOf(emptyList<ConversationRecord>()) }
    var messages by remember { mutableStateOf(emptyList<UiMessage>()) }
    var composer by remember { mutableStateOf("") }
    var generating by remember { mutableStateOf(false) }
    var approvalBusy by remember { mutableStateOf(false) }
    var importBusy by remember { mutableStateOf(false) }
    var activationBusy by remember { mutableStateOf(false) }
    var runtimeStatus by remember { mutableStateOf("خارج از دسترس") }
    var diagnostic by remember { mutableStateOf<String?>(null) }
    var activeModel by remember { mutableStateOf<ModelDescriptor?>(null) }
    var models by remember { mutableStateOf(emptyList<ModelDescriptor>()) }
    var draftModels by remember { mutableStateOf(emptyList<ModelDescriptor>()) }
    var activeDraft by remember { mutableStateOf<ModelDescriptor?>(null) }
    var execution by remember { mutableStateOf<ExecutionState?>(null) }
    var sidebarOpen by remember { mutableStateOf(false) }
    var quickMenuOpen by remember { mutableStateOf(false) }
    var runtimeDetailsOpen by remember { mutableStateOf(false) }
    var renameTarget by remember { mutableStateOf<ConversationRecord?>(null) }
    var deleted by remember { mutableStateOf<DeletedConversation?>(null) }
    var recentLimit by remember { mutableIntStateOf(20) }
    var activeStreamId by remember { mutableStateOf<String?>(null) }

    data class StartupSnapshot(
        val conversations: List<ConversationRecord>,
        val record: ConversationRecord,
        val messages: List<UiMessage>,
    )

    fun loadConversation(record: ConversationRecord) { current = record; messages = history.messages(record.id).map { UiMessage(it.role, it.content, it.id) } }
    fun refreshModels() { scope.launch(Dispatchers.Default) { val listed = manager?.models(); val drafts = manager?.draftModels(); val active = manager?.activeModel(); val draft = (active as? ModelResult.Success)?.value?.let { manager?.draftForModel(it.id) }; withContext(Dispatchers.Main) { models = (listed as? ModelResult.Success)?.value ?: emptyList(); draftModels = (drafts as? ModelResult.Success)?.value ?: emptyList(); activeModel = (active as? ModelResult.Success)?.value; activeDraft = (draft as? ModelResult.Success)?.value; if (!generating && !approvalBusy && !importBusy && !activationBusy) runtimeStatus = if (activeModel != null) "آماده" else "خارج از دسترس" } } }
    fun refreshHistory() { scope.launch(Dispatchers.Default) { val records = history.recent(recentLimit); withContext(Dispatchers.Main) { conversations = records } } }
    LaunchedEffect(Unit) {
        val startup = withContext(Dispatchers.Default) {
            val records = history.recent(recentLimit)
            val record = records.firstOrNull() ?: history.create("گفت‌وگوی جدید")
            val loadedMessages = history.messages(record.id).map { UiMessage(it.role, it.content, it.id) }
            StartupSnapshot(records, record, loadedMessages)
        }
        conversations = startup.conversations
        current = startup.record
        messages = startup.messages
        scope.launch(Dispatchers.Default) {
            manager?.restoreActive()
            refreshModels()
        }
    }
    BackHandler(enabled = sidebarOpen || quickMenuOpen) { if (sidebarOpen) sidebarOpen = false else quickMenuOpen = false }
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri -> if (uri == null || manager == null) return@rememberLauncherForActivityResult; importBusy = true; runtimeStatus = "در حال وارد کردن مدل…"; diagnostic = null; scope.launch { try { val result = withContext(Dispatchers.IO) { manager.import(uri) }; when (result) { is ModelResult.Success -> { runtimeStatus = "مدل وارد شد؛ برای فعال‌سازی آماده است."; diagnostic = null }; is ModelResult.Failure -> diagnostic = result.error.message } } catch (t: Throwable) { diagnostic = t.message ?: "وارد کردن مدل ناموفق بود." } finally { withContext(Dispatchers.Main) { importBusy = false; if (diagnostic == null && activeModel == null) runtimeStatus = "مدل وارد شد؛ برای فعال‌سازی آماده است." else if (diagnostic != null) runtimeStatus = "خطا" }; refreshModels() } } }
    val draftPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri == null || manager == null) return@rememberLauncherForActivityResult
        importBusy = true
        runtimeStatus = "در حال وارد کردن مدل درفت…"
        diagnostic = null
        scope.launch {
            try {
                when (val result = withContext(Dispatchers.IO) { manager.importDraft(uri) }) {
                    is ModelResult.Success -> runtimeStatus = "مدل درفت وارد شد؛ آن را به مدل اصلی متصل کن."
                    is ModelResult.Failure -> { runtimeStatus = "خطا"; diagnostic = result.error.message }
                }
            } catch (t: Throwable) {
                runtimeStatus = "خطا"
                diagnostic = t.message ?: "وارد کردن مدل درفت ناموفق بود."
            } finally {
                withContext(Dispatchers.Main) { importBusy = false; refreshModels() }
            }
        }
    }
    fun assignDraft(draftId: String?) {
        val m = manager ?: return
        val target = activeModel ?: return
        scope.launch(Dispatchers.Default) {
            val result = m.assignDraft(target.id, draftId)
            withContext(Dispatchers.Main) {
                when (result) {
                    is ModelResult.Success -> {
                        val activated = m.activate(target.id)
                        if (activated is ModelResult.Success) {
                            activeDraft = draftModels.firstOrNull { it.id == draftId }
                            runtimeStatus = "آماده"
                            diagnostic = null
                        } else {
                            runtimeStatus = "خطا"
                            diagnostic = (activated as ModelResult.Failure).error.message
                        }
                    }
                    is ModelResult.Failure -> { runtimeStatus = "خطا"; diagnostic = result.error.message }
                }
                refreshModels()
            }
        }
    }
    fun deleteDraft(id: String) {
        scope.launch(Dispatchers.Default) {
            val result = manager?.deleteDraft(id)
            withContext(Dispatchers.Main) {
                if (result is ModelResult.Failure) { runtimeStatus = "خطا"; diagnostic = result.error.message }
                refreshModels()
            }
        }
    }
    fun activateModel(id: String) { val m = manager ?: return; if (generating || approvalBusy || importBusy || activationBusy) return; activationBusy = true; runtimeStatus = "در حال فعال‌سازی…"; diagnostic = null; scope.launch(Dispatchers.Default) { val result = m.activate(id); withContext(Dispatchers.Main) { activationBusy = false; when (result) { is ModelResult.Success -> { activeModel = result.value; runtimeStatus = "آماده"; diagnostic = null }; is ModelResult.Failure -> { runtimeStatus = "خطا"; diagnostic = result.error.message } }; refreshModels() } } }
    fun updateStreamMessage(streamId: String, append: String? = null, replace: String? = null) { if (activeStreamId != streamId) return; messages = messages.map { message -> if (message.id != streamId) message else message.copy(text = replace ?: (message.text + append.orEmpty())) } }
    fun send() { val record = current ?: return; if (manager == null || generating || approvalBusy || importBusy || activationBusy) return; if (activeModel == null) { runtimeStatus = "خارج از دسترس"; diagnostic = "برای ارسال پیام ابتدا یک مدل محلی را فعال کنید."; return }; val text = composer.trim(); if (text.isEmpty()) return; composer = ""; messages = messages + UiMessage(ChatMessage.Role.USER, text, UUID.randomUUID().toString()); val streamId = UUID.randomUUID().toString(); activeStreamId = streamId; messages = messages + UiMessage(ChatMessage.Role.ASSISTANT, "", streamId); generating = true; runtimeStatus = "در حال اجرای Agent"; diagnostic = null; execution = ExecutionState(UUID.randomUUID().toString(), "در حال تولید پاسخ", "در حال تولید پاسخ مدل", System.currentTimeMillis(), requestPreview = text); scope.launch(Dispatchers.Default) { val session = container.createAgentSession(record.id) { event -> withContext(Dispatchers.Main.immediate) { when (event) { AgentEvent.Started -> runtimeStatus = "در حال اجرای Agent"; is AgentEvent.Token -> updateStreamMessage(streamId, append = event.value); is AgentEvent.ActionPrepared -> execution = execution?.copy(id = event.executionId, action = event.actionId, status = "در حال اجرای عملیات"); is AgentEvent.ApprovalRequired -> { execution = execution?.copy(id = event.executionId, status = "نیازمند تأیید", approvalRequired = true); runtimeStatus = "نیازمند تأیید"; generating = false; activeStreamId = null; }; is AgentEvent.ActionExecuted -> execution = execution?.copy(status = "عملیات انجام شد"); is AgentEvent.Failed -> { diagnostic = event.message; execution = execution?.copy(status = "ناموفق", error = event.message, finishedAt = System.currentTimeMillis()) }; AgentEvent.Completed -> if (!(execution?.approvalRequired ?: false)) runtimeStatus = "آماده" } } } ?: run { withContext(Dispatchers.Main) { generating = false; activeStreamId = null; runtimeStatus = "خطا"; diagnostic = "Agent Session در دسترس نیست."; execution = execution?.copy(status = "ناموفق", finishedAt = System.currentTimeMillis(), error = diagnostic) }; return@launch }; try { val result = session.send(text, InferenceSettings(maxNewTokens = 512), requestedRecentMessages = 10); withContext(Dispatchers.Main) { updateStreamMessage(streamId, replace = result.generation.text); generating = false; activeStreamId = null; val finalStatus = when { execution?.approvalRequired == true -> "نیازمند تأیید"; result.actionPlan != null -> "عملیات تکمیل شد"; else -> "پاسخ آماده است" }; execution = execution?.copy(status = finalStatus, finishedAt = System.currentTimeMillis(), resultPreview = result.generation.text.take(240)); if (!(execution?.approvalRequired ?: false)) runtimeStatus = "آماده" } } catch (t: Throwable) { withContext(Dispatchers.Main) { generating = false; activeStreamId = null; runtimeStatus = "خطا"; diagnostic = t.message ?: "Local agent execution failed"; execution = execution?.copy(status = "ناموفق", finishedAt = System.currentTimeMillis(), error = diagnostic) } }; val updated = history.recent(recentLimit).firstOrNull { it.id == record.id }; withContext(Dispatchers.Main) { if (updated != null) current = updated; conversations = history.recent(recentLimit) } } }
    fun approveExecution() { val record = current ?: return; val pending = execution ?: return; if (!pending.approvalRequired || approvalBusy) return; approvalBusy = true; generating = true; runtimeStatus = "در حال تأیید عملیات"; diagnostic = null; execution = pending.copy(status = "در حال تأیید"); val streamId = UUID.randomUUID().toString(); activeStreamId = streamId; messages = messages + UiMessage(ChatMessage.Role.ASSISTANT, "", streamId); scope.launch(Dispatchers.Default) { try { val outcome = container.approveAndExecute(pending.id); if (!outcome.success) { withContext(Dispatchers.Main) { approvalBusy = false; generating = false; activeStreamId = null; runtimeStatus = "خطا"; diagnostic = outcome.message; execution = pending.copy(status = "ناموفق", error = outcome.message, finishedAt = System.currentTimeMillis()) }; return@launch }; val session = container.createAgentSession(record.id) { event -> withContext(Dispatchers.Main.immediate) { when (event) { AgentEvent.Started -> runtimeStatus = "در حال دریافت پاسخ نهایی"; is AgentEvent.Token -> updateStreamMessage(streamId, append = event.value); is AgentEvent.Failed -> diagnostic = event.message; else -> Unit } } } ?: error("Agent Session در دسترس نیست."); val result = session.resumeApproved(pending.id, outcome, InferenceSettings(maxNewTokens = 512), requestedRecentMessages = 10); withContext(Dispatchers.Main) { updateStreamMessage(streamId, replace = result.generation.text); approvalBusy = false; generating = false; activeStreamId = null; runtimeStatus = "آماده"; execution = pending.copy(status = "عملیات تأیید و تکمیل شد", finishedAt = System.currentTimeMillis(), resultPreview = result.generation.text.take(240), approvalRequired = false); conversations = history.recent(recentLimit); history.recent(recentLimit).firstOrNull { it.id == record.id }?.let { current = it } } } catch (t: Throwable) { withContext(Dispatchers.Main) { approvalBusy = false; generating = false; activeStreamId = null; runtimeStatus = "خطا"; diagnostic = t.message ?: "Approval continuation failed"; execution = pending.copy(status = "ناموفق", error = diagnostic, finishedAt = System.currentTimeMillis()) } } } }
    fun rejectExecution() { val pending = execution ?: return; if (!pending.approvalRequired || approvalBusy) return; approvalBusy = true; runtimeStatus = "در حال رد عملیات"; execution = pending.copy(status = "در حال رد"); scope.launch(Dispatchers.Default) { val rejected = container.reject(pending.id); withContext(Dispatchers.Main) { approvalBusy = false; generating = false; activeStreamId = null; if (rejected) { runtimeStatus = if (activeModel != null) "آماده" else "خارج از دسترس"; execution = pending.copy(status = "رد شد", approvalRequired = false, finishedAt = System.currentTimeMillis()) } else { runtimeStatus = "خطا"; diagnostic = "رد عملیات انجام نشد. وضعیت Execution را بررسی کنید."; execution = pending.copy(status = "ناموفق", error = diagnostic, finishedAt = System.currentTimeMillis()) } } } }
    fun stop() { if (!generating || approvalBusy) return; activeStreamId = null; scope.launch(Dispatchers.Default) { manager?.stopGeneration() }; generating = false; runtimeStatus = if (activeModel != null) "آماده" else "خارج از دسترس"; execution = execution?.copy(status = "متوقف شد", finishedAt = System.currentTimeMillis()) }
    fun deactivateModel() { val m = manager ?: return; if (activeModel == null) return; scope.launch(Dispatchers.Default) { val result = m.deactivate(); withContext(Dispatchers.Main) { when (result) { is ModelResult.Success -> { activeModel = null; runtimeStatus = "خارج از دسترس"; diagnostic = null; refreshModels() }; is ModelResult.Failure -> { runtimeStatus = "خطا"; diagnostic = result.error.message } } } } }
    fun createConversation() { scope.launch(Dispatchers.Default) { val record = history.create("گفت‌وگوی جدید"); withContext(Dispatchers.Main) { current = record; messages = emptyList(); composer = ""; conversations = history.recent(recentLimit); destination = AppDestination.CHAT; sidebarOpen = false } } }
    fun openConversation(record: ConversationRecord) { loadConversation(record); destination = AppDestination.CHAT; sidebarOpen = false }
    fun deleteConversation(record: ConversationRecord) { scope.launch(Dispatchers.Default) { val removed = history.delete(record.id) ?: return@launch; val replacement = if (current?.id == removed.id) history.recent(1).firstOrNull() else null; withContext(Dispatchers.Main) { if (replacement != null) loadConversation(replacement) else if (current?.id == removed.id) { current = null; messages = emptyList() }; conversations = history.recent(recentLimit); deleted = DeletedConversation(removed) } } }
    fun restoreDeleted(record: ConversationRecord) { scope.launch(Dispatchers.Default) { if (history.restore(record)) withContext(Dispatchers.Main) { loadConversation(record); conversations = history.recent(recentLimit); deleted = null } } }
    CompositionLocalProvider(LocalLayoutDirection provides LayoutDirection.Rtl) { AppTheme { Surface(Modifier.fillMaxSize()) { Box(Modifier.fillMaxSize()) { Column(Modifier.fillMaxSize()) { AppHeader(destination, runtimeStatus, activeModel, { destination = it; quickMenuOpen = false }, { quickMenuOpen = !quickMenuOpen }, { quickMenuOpen = false; sidebarOpen = !sidebarOpen }, { runtimeDetailsOpen = !runtimeDetailsOpen }); when (destination) { AppDestination.CHAT -> ChatPage(messages, composer, generating, approvalBusy, { composer = it }, ::send, ::stop, execution, ::approveExecution, ::rejectExecution, { destination = AppDestination.WORK }); AppDestination.WORK -> WorkspacePage(execution, activeModel, { destination = AppDestination.DIAGNOSTICS }); AppDestination.WORKSPACE -> WorkspacePage(execution, activeModel, { destination = AppDestination.DIAGNOSTICS }); AppDestination.DIAGNOSTICS -> DiagnosticsPage(diagnostic, execution); AppDestination.SETTINGS -> SettingsScreen(models, activeModel, runtimeStatus, diagnostic, { picker.launch(arrayOf("application/octet-stream", "application/*")) }, draftModels, activeDraft, { draftPicker.launch(arrayOf("application/octet-stream", "application/*")) }, ::assignDraft, ::deleteDraft, ::refreshModels, ::activateModel, ::deactivateModel, { id -> scope.launch(Dispatchers.Default) { manager?.delete(id); refreshModels() } }); AppDestination.ABOUT -> AboutPage() } }; if (sidebarOpen) Sidebar(current = current, conversations = conversations.take(recentLimit), destination = destination, onClose = { sidebarOpen = false }, onDestination = { destination = it; sidebarOpen = false }, onNew = ::createConversation, onOpen = ::openConversation, onRename = { renameTarget = it }, onDelete = ::deleteConversation, onMore = { recentLimit += 20; refreshHistory() }, canShowMore = conversations.size >= recentLimit); if (quickMenuOpen) AppQuickMenu({ quickMenuOpen = false; destination = AppDestination.DIAGNOSTICS }, { quickMenuOpen = false; destination = AppDestination.SETTINGS }); if (runtimeDetailsOpen) RuntimeDetailsDialog(status = runtimeStatus, model = activeModel, runtimeName = "llama.cpp", runtimeVersion = "pinned", backend = null, onDismiss = { runtimeDetailsOpen = false }); ApprovalDialog(execution, ::approveExecution, ::rejectExecution); deleted?.let { UndoBar(onUndo = { restoreDeleted(it.record) }, onDismiss = { deleted = null }, onExpire = { deleted = null }) }; renameTarget?.let { RenameDialog(initial = it.title, onDismiss = { renameTarget = null }, onConfirm = { renamed -> scope.launch(Dispatchers.Default) { history.rename(it.id, renamed); withContext(Dispatchers.Main) { renameTarget = null; refreshHistory() } } }) } } } } }
}

private fun Boolean?.orFalse(): Boolean = this == true
