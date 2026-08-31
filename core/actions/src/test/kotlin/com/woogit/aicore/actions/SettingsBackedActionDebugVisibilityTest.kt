package com.woogit.aicore.actions

import com.woogit.aicore.settings.CoreSettings
import com.woogit.aicore.settings.InMemoryCoreSettingsStore
import kotlinx.coroutines.runBlocking
import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class SettingsBackedActionDebugVisibilityTest {
    @Test
    fun visibilityReadsAndWritesCentralSettings() = runBlocking {
        val store = InMemoryCoreSettingsStore(CoreSettings(showFullErrorDetails = true))
        val visibility = SettingsBackedActionDebugVisibility(
            initialSettings = store.get(),
            settingsStore = store
        )

        assertTrue(visibility.fullErrorDetailsVisible)
        visibility.setFullErrorDetailsVisible(false)
        assertFalse(visibility.fullErrorDetailsVisible)
        assertFalse(store.get().showFullErrorDetails)
    }
}
