package com.woogit.aicore.settings

/** Extensible category boundary for the central settings core. */
data class SettingsCategory(
    val id: String,
    val title: String,
    val keys: Set<String>
)

interface SettingsCategoryRegistry {
    fun categories(): List<SettingsCategory>
    fun register(category: SettingsCategory)
}

class InMemorySettingsCategoryRegistry : SettingsCategoryRegistry {
    private val values = linkedMapOf<String, SettingsCategory>()

    @Synchronized
    override fun categories(): List<SettingsCategory> = values.values.toList()

    @Synchronized
    override fun register(category: SettingsCategory) {
        require(category.id.isNotBlank()) { "Settings category id cannot be blank" }
        require(category.id !in values) { "Settings category already registered: ${category.id}" }
        values[category.id] = category
    }
}
