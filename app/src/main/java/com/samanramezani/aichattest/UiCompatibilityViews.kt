package com.samanramezani.aichattest

import androidx.compose.runtime.Composable
import com.samanramezani.aichattest.ui.components.RuntimeDetailsDialog
import com.samanramezani.aichattest.ui.components.Sidebar
import com.samanramezani.aichattest.ui.state.ExecutionState
import com.woogit.aicore.conversation.ConversationRecord
import com.woogit.aicore.domain.ModelDescriptor

@Composable
internal fun AppSidebar(
    conversations: List<ConversationRecord>,
    current: ConversationRecord?,
    recentLimit: Int,
    onNew: () -> Unit,
    onOpen: (ConversationRecord) -> Unit,
    onDelete: (ConversationRecord) -> Unit,
    onMore: () -> Unit,
    onClose: () -> Unit,
) {
    Sidebar(
        current = current,
        conversations = conversations,
        destination = com.samanramezani.aichattest.ui.navigation.AppDestination.CHAT,
        onClose = onClose,
        onDestination = {},
        onNew = onNew,
        onOpen = onOpen,
        onRename = {},
        onDelete = onDelete,
        onMore = onMore,
        canShowMore = conversations.size >= recentLimit,
    )
}

@Composable
internal fun AppQuickMenu(onDiagnostics: () -> Unit, onSettings: () -> Unit) {
    AppQuickMenu(onDiagnostics = onDiagnostics, onSettings = onSettings)
}

@Composable
internal fun RuntimeDetailsDialog(
    model: ModelDescriptor?,
    status: String,
    onDismiss: () -> Unit,
) {
    RuntimeDetailsDialog(
        status = status,
        model = model,
        runtimeName = "llama.cpp",
        runtimeVersion = "N/A",
        backend = null,
        onDismiss = onDismiss,
    )
}
