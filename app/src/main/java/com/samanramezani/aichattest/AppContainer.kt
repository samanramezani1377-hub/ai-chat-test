package com.samanramezani.aichattest

import android.content.Context
import com.woogit.aicore.actions.InMemoryActionRegistry
import com.woogit.aicore.conversation.ConversationHistoryRepository
import com.woogit.aicore.conversation.InMemoryConversationHistoryRepository
import com.woogit.aicore.domain.ActionRegistry
import com.woogit.aicore.runtime.RuntimeAdapter
import com.woogit.aicore.runtime.android.LlamaCppAndroidRuntimeAdapter
import java.nio.file.Files

/** Application composition root. Implementations are wired here, never inside UI screens. */
class AppContainer(context: Context? = null) {
    val actionRegistry: ActionRegistry = InMemoryActionRegistry()
    val modelRuntime: RuntimeAdapter = LlamaCppAndroidRuntimeAdapter()

    /** Conversation metadata boundary used by the application shell. */
    val conversationHistory: ConversationHistoryRepository = InMemoryConversationHistoryRepository()

    val modelManager: AndroidModelManager? = context?.let {
        val directory = it.filesDir.toPath().resolve("models")
        Files.createDirectories(directory)
        AndroidModelManager(it.contentResolver, directory, modelRuntime)
    }
}
