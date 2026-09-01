package com.samanramezani.aichattest

import android.content.Context
import android.os.Build
import com.woogit.aicore.actions.DefaultActionRegistry
import com.woogit.aicore.actions.registerBuiltinFileActions
import com.woogit.aicore.actions.registerProviderActions
import com.woogit.aicore.conversation.ConversationHistoryRepository
import com.woogit.aicore.conversation.InMemoryConversationHistoryRepository
import com.woogit.aicore.domain.ActionRegistry
import com.woogit.aicore.domain.ModelResult
import com.woogit.aicore.runtime.RuntimeAdapter
import com.woogit.aicore.runtime.android.LlamaCppAndroidRuntimeAdapter
import java.nio.file.Files

/** Application composition root. Implementations are wired here, never inside UI screens. */
class AppContainer(context: Context? = null) {
    private val appContext = context?.applicationContext

    val modelRuntime: RuntimeAdapter = LlamaCppAndroidRuntimeAdapter()

    val actionRegistry: ActionRegistry = DefaultActionRegistry().also { registry ->
        if (appContext != null) {
            val workspace = appContext.filesDir.toPath().resolve("workspace")
            Files.createDirectories(workspace)
            registry.registerBuiltinFileActions(workspace)
            registry.registerProviderActions(
                modelInfo = {
                    val active = modelManager?.activeModel()
                    when (active) {
                        is ModelResult.Success -> active.value?.let {
                            "name=${it.displayName}, quantization=${it.quantization}, sizeBytes=${it.sizeBytes}"
                        } ?: "no-active-model"
                        is ModelResult.Failure -> "unavailable: ${active.error.message}"
                        null -> "no-model-manager"
                    }
                },
                deviceInfo = {
                    "manufacturer=${Build.MANUFACTURER}, model=${Build.MODEL}, sdk=${Build.VERSION.SDK_INT}"
                },
            )
        }
    }

    /** Durable conversation metadata when running on Android; Core fallback for non-Android construction. */
    val conversationHistory: ConversationHistoryRepository = appContext?.let {
        AndroidConversationHistoryRepository(it)
    } ?: InMemoryConversationHistoryRepository()

    val modelManager: AndroidModelManager? = appContext?.let {
        val directory = it.filesDir.toPath().resolve("models")
        Files.createDirectories(directory)
        AndroidModelManager(it.contentResolver, directory, modelRuntime)
    }
}
