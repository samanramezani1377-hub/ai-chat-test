package com.woogit.aicore.actions

import com.woogit.aicore.settings.CoreSettingsStore

/** Presentation adapter: central settings decide visibility, while debug data stays intact. */
class SettingsBackedActionDebugVisibility(
    private val settingsStore: CoreSettingsStore
) : ActionDebugVisibility {
    override var fullErrorDetailsVisible: Boolean = true
        get() = kotlinx.coroutines.runBlocking { settingsStore.get().showFullErrorDetails }
        set(value) {
            kotlinx.coroutines.runBlocking {
                settingsStore.update(settingsStore.get().copy(showFullErrorDetails = value))
            }
        }
}
