package com.woogit.aicore.settings

import kotlinx.coroutines.runBlocking
import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class DebugSettingsTest {
    @Test
    fun debugSettingIsReadAndWrittenThroughCentralFacade() = runBlocking {
        val store = InMemoryCoreSettingsStore()
        val categories = InMemorySettingsCategoryRegistry()
        val central = CentralSettingsRegistry(store, categories)
        val debug = DebugSettings(central)

        assertTrue(debug.showFullErrorDetails())
        debug.setShowFullErrorDetails(false)
        assertFalse(debug.showFullErrorDetails())
    }
}
