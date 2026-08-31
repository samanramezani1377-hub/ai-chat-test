package com.samanramezani.aichattest

import com.woogit.aicore.actions.InMemoryActionRegistry
import com.woogit.aicore.domain.ActionRegistry

/** Application composition root. Implementations are wired here, never inside UI screens. */
class AppContainer {
    val actionRegistry: ActionRegistry = InMemoryActionRegistry()
}
