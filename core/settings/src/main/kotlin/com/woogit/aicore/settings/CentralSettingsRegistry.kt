package com.woogit.aicore.settings

/** Central facade for category registration and settings storage. */
class CentralSettingsRegistry(
    private val store: CoreSettingsStore,
    private val categoryRegistry: SettingsCategoryRegistry
) {
    suspend fun get(): CoreSettings = store.get()

    suspend fun update(transform: (CoreSettings) -> CoreSettings): CoreSettings {
        val updated = transform(store.get())
        store.update(updated)
        return updated
    }

    fun categories(): List<SettingsCategory> = categoryRegistry.categories()

    fun registerCategory(category: SettingsCategory) = categoryRegistry.register(category)
}
