package com.samanramezani.aichattest

import com.woogit.aicore.actions.InMemoryActionRegistry
import com.woogit.aicore.domain.ActionRegistry
import com.woogit.aicore.runtime.RuntimeAdapter
import com.woogit.aicore.runtime.android.LlamaCppAndroidRuntimeAdapter

/** Application composition root. Implementations are wired here, never inside UI screens. */
class AppContainer {
    val actionRegistry: ActionRegistry = InMemoryActionRegistry()

    /** Real local GGUF runtime used by the application. */
    val modelRuntime: RuntimeAdapter = LlamaCppAndroidRuntimeAdapter()
}
