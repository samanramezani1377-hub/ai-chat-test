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
    var runtimeStatus by remember { mutableStateOf("خارج از دسترس") }
    var diagnostic by remember { mutableStateOf<String?>(null) }
    var activeModel by remember { mutableStateOf<ModelDescriptor?>(null) }
    var models by remember { mutableStateOf(emptyList<ModelDescriptor>()) }
    var execution by remember { mutableStateOf<ExecutionState?>(null) }
    var sidebarOpen by remember { mutableStateOf(false) }
    var quickMenuOpen by remember { mutableStateOf(false) }
    var runtimeDetailsOpen by remember { mutableStateOf(false) }
    var renameTarget by remember { mutableStateOf<ConversationRecord?>(null) }
    var deleted by remember { mutableStateOf<DeletedConversation?>(null) }
    var recentLimit by remember { mutableIntStateOf(20) }

    fun loadConversation(record: ConversationRecord) {
        current = record
        messages = history.messages(record.id).map { UiMessage(it.role, it.content, it.id) }
    }

    fun refreshModels() {
        scope.launch(Dispatchers.Default) {
            val listed = manager?.models()
            val active = manager?.activeModel()
            withContext(Dispatchers.Main) {
                models = (listed as? ModelResult.Success)?.value ?: emptyList()
                activeModel = (active as? ModelResult.Success)?.value
                if (!generating && !approvalBusy) runtimeStatus = if (activeModel != null) "آماده" else "خارج از دسترس"
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
        manager?.let { withContext(Dispatchers.Default) { it.restoreActive() } }
        refreshModels()
        val records = history.recent(recentLimit)
        val record = records.firstOrNull() ?: history.create("گفت‌وگوی جدید")
        conversations = history.recent(recentLimit)
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
        if (manager == null || generating || approvalBusy) return
        val text = composer.trim()
        if (text.isEmpty()) return
        composer = ""
        messages = messages + UiMessage(ChatMessage.Role.USER, text, UUID.randomUUID().toString())
        generating = true
        runtimeStatus = "در حال اجرای Agent"
        diagnostic = null
        execution = ExecutionState(UUID.randomUUID().toString(), "درخواست Agent", "در حال اجرا", System.currentTimeMillis(), requestPreview = text)
        scope.launch(Dispatchers.Default) {
            val session = container.createAgentSession(record.id) { event ->
                scope.launch(Dispatchers.Main) {
                    when (event) {
                        AgentEvent.Started -> runtimeStatus = "در حال اجرای Agent"
                        is AgentEvent.Token -> Unit
                        is AgentEvent.ActionPrepared -> execution = execution?.copy(id = event.executionId, action = event.actionId, status = "در حال اجرای عملیات")
                        is AgentEvent.ApprovalRequired -> {
                            execution = execution?.copy(id = event.executionId, status = "نیازمند تأیید", approvalRequired = true)
                            runtimeStatus = "نیازمند تأیید"
                            generating = false
                        }
                        is AgentEvent.ActionExecuted -> execution = execution?.copy(status = "عملیات انجام شد")
                        is AgentEvent.Failed -> {
                            diagnostic = event.message
                            execution = execution?.copy(status = "ناموفق", error = event.message, finishedAt = System.currentTimeMillis())
                        }
                        AgentEvent.Completed -> if (!execution?.approvalRequired.orFalse()) runtimeStatus = "آماده"
                    }
                }
            } ?: run {
                withContext(Dispatchers.Main) {
                    generating = false
                    runtimeStatus = "خطا"
                    diagnostic = "Agent Session در دسترس نیست."
                    execution = execution?.copy(status = "ناموفق", finishedAt = System.currentTimeMillis(), error = diagnostic)
                }
                return@launch
            }
            try {
                val result = session.send(text, InferenceSettings(maxNewTokens = 512), requestedRecentMessages = 24)
                withContext(Dispatchers.Main) {
                    generating = false
                    messages = messages + UiMessage(ChatMessage.Role.ASSISTANT, result.generation.text, UUID.randomUUID().toString())
                    execution = execution?.copy(status = if (result.actionPlan != null) "عملیات تکمیل شد" else "پاسخ آماده است", finishedAt = System.currentTimeMillis(), resultPreview = result.generation.text.take(240))
                    if (!execution?.approvalRequired.orFalse()) runtimeStatus = "آماده"
                }
            } catch (t: Throwable) {
                withContext(Dispatchers.Main) {
                    generating = false
                    runtimeStatus = "خطا"
                    diagnostic = t.message ?: "Local agent execution failed"
                    execution = execution?.copy(status = "ناموفق", finishedAt = System.currentTimeMillis(), error = diagnostic)
                }
            }
            val updated = history.recent(recentLimit).firstOrNull { it.id == record.id }
            withContext(Dispatchers.Main) {
                if (updated != null) current = updated
                conversations = history.recent(recentLimit)
            }
        }
    }

    fun approveExecution() {
        val record = current ?: return
        val pending = execution ?: return
        if (!pending.approvalRequired || approvalBusy) return
        approvalBusy = true
        generating = true
        runtimeStatus = "در حال تأیید عملیات"
        diagnostic = null
        execution = pending.copy(status = "در حال تأیید")
        scope.launch(Dispatchers.Default) {
            try {
                val outcome = container.approveAndExecute(pending.id)
                if (!outcome.success) {
                    withContext(Dispatchers.Main) {
                        approvalBusy = false
                        generating = false
                        runtimeStatus = "خطا"
                        diagnostic = outcome.message
                        execution = pending.copy(status = "ناموفق", error = outcome.message, finishedAt = System.currentTimeMillis())
                    }
                    return@launch
                }
                val session = container.createAgentSession(record.id) { event ->
                    scope.launch(Dispatchers.Main) {
                        when (event) {
                            AgentEvent.Started -> runtimeStatus = "در حال دریافت پاسخ نهایی"
                            is AgentEvent.Token -> Unit
                            is AgentEvent.Failed -> diagnostic = event.message
                            else -> Unit
                        }
                    }
                } ?: error("Agent Session در دسترس نیست.")
                val result = session.resumeApproved(pending.id, outcome, InferenceSettings(maxNewTokens = 512), requestedRecentMessages = 24)
                withContext(Dispatchers.Main) {
                    messages = messages + UiMessage(ChatMessage.Role.ASSISTANT, result.generation.text, UUID.randomUUID().toString())
                    approvalBusy = false
                    generating = false
                    runtimeStatus = "آماده"
                    execution = pending.copy(status = "عملیات تأیید و تکمیل شد", finishedAt = System.currentTimeMillis(), resultPreview = result.generation.text.take(240), approvalRequired = false)
                    conversations = history.recent(recentLimit)
                    history.recent(recentLimit).firstOrNull { it.id == record.id }?.let { current = it }
                }
            } catch (t: Throwable) {
                withContext(Dispatchers.Main) {
                    approvalBusy = false
                    generating = false
                    runtimeStatus = "خطا"
                    diagnostic = t.message ?: "Approval continuation failed"
                    execution = pending.copy(status = "ناموفق", error = diagnostic, finishedAt = System.currentTimeMillis())
                }
            }
        }
    }

    fun rejectExecution() {
        val pending = execution ?: return
        if (!pending.approvalRequired || approvalBusy) return
        approvalBusy = true
        runtimeStatus = "در حال رد عملیات"
        execution = pending.copy(status = "در حال رد")
        scope.launch(Dispatchers.Default) {
            val rejected = container.reject(pending.id)
            withContext(Dispatchers.Main) {
                approvalBusy = false
                generating = false
                if (rejected) {
                    runtimeStatus = if (activeModel != null) "آماده" else "خارج از دسترس"
                    execution = pending.copy(status = "رد شد", approvalRequired = false, finishedAt = System.currentTimeMillis())
                } else {
                    runtimeStatus = "خطا"
                    diagnostic = "رد عملیات انجام نشد. وضعیت Execution را بررسی کنید."
                    execution = pending.copy(status = "ناموفق", error = diagnostic, finishedAt = System.currentTimeMillis())
                }
            }
        }
    }

    fun stop() {
        if (!generating || approvalBusy) return
        scope.launch(Dispatchers.Default) { manager?.stopGeneration() }
        generating = false
        runtimeStatus = if (activeModel != null) "آماده" else "خارج از دسترس"
        execution = execution?.copy(status = "متوقف شد", finishedAt = System.currentTimeMillis())
    }

    fun deactivateModel() {
        val m = manager ?: return
        if (activeModel == null) return
        scope.launch(Dispatchers.Default) {
            val result = m.deactivate()
            withContext(Dispatchers.Main) {
                when (result) {
                    is ModelResult.Success -> { activeModel = null; runtimeStatus = "خارج از دسترس"; diagnostic = null; refreshModels() }
                    is ModelResult.Failure -> { runtimeStatus = "خطا"; diagnostic = result.error.message }
                }
            }
        }
    }

    fun createConversation() {
        scope.launch(Dispatchers.Default) {
            val record = history.create("گفت‌وگوی جدید")
            withContext(Dispatchers.Main) { current = record; messages = emptyList(); composer = ""; conversations = history.recent(recentLimit); destination = AppDestination.CHAT; sidebarOpen = false }
        }
    }

    fun openConversation(record: ConversationRecord) {
        loadConversation(record)
        destination = AppDestination.CHAT
        sidebarOpen = false
    }

    fun deleteConversation(record: ConversationRecord) {
        scope.launch(Dispatchers.Default) {
            val removed = history.delete(record.id) ?: return@launch
            val replacement = if (current?.id == removed.id) history.recent(1).firstOrNull() else null
            withContext(Dispatchers.Main) {
                if (replacement != null) loadConversation(replacement) else if (current?.id == removed.id) { current = null; messages = emptyList() }
                conversations = history.recent(recentLimit)
                deleted = DeletedConversation(removed)
            }
        }
    }

    fun restoreDeleted(record: ConversationRecord) {
        scope.launch(Dispatchers.Default) {
            if (history.restore(record)) withContext(Dispatchers.Main) { loadConversation(record); conversations = history.recent(recentLimit); deleted = null }
        }
    }

    CompositionLocalProvider(LocalLayoutDirection provides LayoutDirection.Rtl) {
        AppTheme {
            Surface(Modifier.fillMaxSize()) {
                Box(Modifier.fillMaxSize()) {
                    Column(Modifier.fillMaxSize()) {
                        AppHeader(destination, runtimeStatus, activeModel, { destination = it; quickMenuOpen = false }, { quickMenuOpen = !quickMenuOpen }, { quickMenuOpen = false; sidebarOpen = true }, { runtimeDetailsOpen = true })
                        Box(Modifier.weight(1f).fillMaxWidth()) {
                            when (destination) {
                                AppDestination.CHAT -> ChatPage(messages, composer, generating, approvalBusy, { composer = it }, ::send, ::stop, execution, ::approveExecution, ::rejectExecution) { destination = AppDestination.WORK }
                                AppDestination.WORK -> WorkspacePage(execution, activeModel) { destination = AppDestination.DIAGNOSTICS }
                                AppDestination.DIAGNOSTICS -> DiagnosticsPage(diagnostic, execution)
                                AppDestination.SETTINGS -> SettingsScreen(models, activeModel, diagnostic, onImport = { picker.launch(arrayOf("application/octet-stream", "application/gzip", "*/*")) }, onRefresh = ::refreshModels, onActivate = { id -> scope.launch(Dispatchers.Default) { manager?.activate(id); refreshModels() } }, onDeactivate = ::deactivateModel, onDelete = { id -> scope.launch(Dispatchers.Default) { manager?.delete(id); refreshModels() } })
                                AppDestination.ABOUT -> AboutPage()
                            }
                        }
                    }
                    if (quickMenuOpen) QuickMenu({ destination = AppDestination.DIAGNOSTICS; quickMenuOpen = false }, { destination = AppDestination.SETTINGS; quickMenuOpen = false })
                    if (sidebarOpen) Sidebar(current, conversations, destination, { sidebarOpen = false }, { destination = it; sidebarOpen = false }, ::createConversation, ::openConversation, { renameTarget = it }, ::deleteConversation, { recentLimit += 5; refreshHistory() }, conversations.size >= recentLimit)
                    deleted?.let { item -> UndoBar({ restoreDeleted(item.record) }, { scope.launch(Dispatchers.Default) { history.purge(item.record.id) }; deleted = null }, { scope.launch(Dispatchers.Default) { history.purge(item.record.id) }; deleted = null }) }
                }
            }
        }
    }

    renameTarget?.let { target ->
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
    if (runtimeDetailsOpen) RuntimeDetailsDialog(runtimeStatus, activeModel, container.modelRuntime.runtimeInfo().name, container.modelRuntime.runtimeInfo().version, container.modelRuntime.runtimeInfo().backend) { runtimeDetailsOpen = false }
}

private fun Boolean?.orFalse() = this == true
