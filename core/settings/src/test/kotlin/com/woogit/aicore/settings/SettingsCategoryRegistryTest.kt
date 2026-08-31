package com.woogit.aicore.settings

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class SettingsCategoryRegistryTest {
    @Test
    fun categoriesAreRegisteredAndKeptExtensible() {
        val registry = InMemorySettingsCategoryRegistry()
        registry.register(SettingsCategory("debug", "Debug", setOf("showFullErrorDetails")))
        registry.register(SettingsCategory("network", "Network", emptySet()))

        assertEquals(listOf("debug", "network"), registry.categories().map { it.id })
        assertFailsWith<IllegalArgumentException> {
            registry.register(SettingsCategory("debug", "Duplicate", emptySet()))
        }
    }
}
