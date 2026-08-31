package com.woogit.aicore.settings

/** Central settings contract shared by core capabilities. */
data class CoreSettings(
    val showFullErrorDetails: Boolean = true
)

interface CoreSettingsStore {
    suspend fun get(): CoreSettings
    suspend fun update(settings: CoreSettings)
}

class InMemoryCoreSettingsStore(initial: CoreSettings = CoreSettings()) : CoreSettingsStore {
    private var settings = initial

    @Synchronized
    override suspend fun get(): CoreSettings = settings

    @Synchronized
    override suspend fun update(settings: CoreSettings) {
        this.settings = settings
    }
}
