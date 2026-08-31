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

    override suspend fun get(): CoreSettings = synchronized(this) {
        settings
    }

    override suspend fun update(settings: CoreSettings) {
        synchronized(this) {
            this.settings = settings
        }
    }
}
