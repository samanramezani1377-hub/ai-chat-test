package com.woogit.aicore.settings

object DefaultSettingsCategories {
    val debug = SettingsCategory(
        id = "debug",
        title = "Debug",
        keys = setOf("showFullErrorDetails")
    )

    fun registerAll(registry: SettingsCategoryRegistry) {
        registry.register(debug)
    }
}
