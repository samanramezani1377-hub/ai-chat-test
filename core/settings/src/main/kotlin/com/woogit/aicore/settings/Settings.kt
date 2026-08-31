package com.woogit.aicore.settings

interface SettingsRepository {
    fun schemaVersion(): Int
    fun get(key: String): Setting<*>?
    fun set(setting: Setting<*>)
}

data class Setting<T>(
    val key: String,
    val domain: String,
    val type: String,
    val value: T,
    val defaultValue: T,
    val constraints: SettingConstraints? = null,
    val version: Int,
    val source: SettingSource
)

data class SettingConstraints(
    val min: Double? = null,
    val max: Double? = null,
    val allowedValues: Set<String>? = null
)

enum class SettingSource { USER, RUNTIME, SYSTEM, POLICY }

class InMemorySettingsRepository(initialVersion: Int = 1) : SettingsRepository {
    private val values = java.util.concurrent.ConcurrentHashMap<String, Setting<*>>()
    private var version = initialVersion

    override fun schemaVersion(): Int = version

    override fun get(key: String): Setting<*>? = values[key]

    override fun set(setting: Setting<*>) {
        values[setting.key] = setting
        version = maxOf(version, setting.version)
    }
}
