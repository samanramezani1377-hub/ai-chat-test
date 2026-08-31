package com.woogit.aicore.settings

import kotlinx.coroutines.runBlocking
import kotlin.test.Test
import kotlin.test.assertEquals

class CentralSettingsRegistryTest {
    @Test
    fun facadeOwnsSettingsAndCategoryRegistration() = runBlocking {
        val store = InMemoryCoreSettingsStore()
        val categories = InMemorySettingsCategoryRegistry()
        val central = CentralSettingsRegistry(store, categories)

        central.registerCategory(SettingsCategory("debug", "Debug", setOf("showFullErrorDetails")))
        central.update { it.copy(showFullErrorDetails = false) }

        assertEquals(false, central.get().showFullErrorDetails)
        assertEquals(listOf("debug"), central.categories().map { it.id })
    }
}
