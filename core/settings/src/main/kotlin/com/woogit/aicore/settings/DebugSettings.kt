package com.woogit.aicore.settings

/** Typed access to the development-only Debug settings. */
class DebugSettings(private val central: CentralSettingsRegistry) {
    suspend fun showFullErrorDetails(): Boolean = central.get().showFullErrorDetails

    suspend fun setShowFullErrorDetails(value: Boolean) {
        central.update { it.copy(showFullErrorDetails = value) }
    }
}
