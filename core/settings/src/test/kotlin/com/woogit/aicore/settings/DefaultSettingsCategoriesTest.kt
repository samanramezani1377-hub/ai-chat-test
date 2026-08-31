package com.woogit.aicore.settings

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class DefaultSettingsCategoriesTest {
    @Test
    fun debugIsARegisteredCentralSettingsCategory() {
        val registry = InMemorySettingsCategoryRegistry()
        DefaultSettingsCategories.registerAll(registry)

        val debug = registry.categories().single { it.id == "debug" }
        assertEquals("Debug", debug.title)
        assertTrue("showFullErrorDetails" in debug.keys)
    }
}
