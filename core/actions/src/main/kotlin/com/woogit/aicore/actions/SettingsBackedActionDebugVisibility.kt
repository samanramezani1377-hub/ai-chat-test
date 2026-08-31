package com.woogit.aicore.actions

import com.woogit.aicore.settings.CoreSettings
import com.woogit.aicore.settings.CoreSettingsStore

/**
 * Presentation adapter for central settings.
 * Visibility is kept synchronous; the async store is injected behind a synchronous snapshot.
 */
class SettingsBackedActionDebugVisibility(
    initialSettings: CoreSettings = CoreSettings(),
    private val settingsStore: CoreSettingsStore? = null
) : ActionDebugVisibility {
    override var fullErrorDetailsVisible: Boolean = initialSettings.showFullErrorDetails
        private set

    suspend fun refresh() {
        fullErrorDetailsVisible = settingsStore?.get()?.showFullErrorDetails ?: fullErrorDetailsVisible
    }

    suspend fun setFullErrorDetailsVisible(value: Boolean) {
        fullErrorDetailsVisible = value
        settingsStore?.update(settingsStore.get().copy(showFullErrorDetails = value))
    }
}
