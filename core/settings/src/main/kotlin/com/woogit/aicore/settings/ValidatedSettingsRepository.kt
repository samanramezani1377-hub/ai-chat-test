package com.woogit.aicore.settings

class ValidatedSettingsRepository(
    private val delegate: SettingsRepository
) : SettingsRepository by delegate {
    override fun set(setting: Setting<*>) {
        validate(setting)
        delegate.set(setting)
    }

    private fun validate(setting: Setting<*>) {
        require(setting.key.isNotBlank()) { "setting key must not be blank" }
        require(setting.domain.isNotBlank()) { "setting domain must not be blank" }
        require(setting.type.isNotBlank()) { "setting type must not be blank" }
        require(setting.version > 0) { "setting version must be positive" }

        val constraints = setting.constraints ?: return
        if (setting.value is Number) {
            val value = setting.value.toDouble()
            constraints.min?.let { require(value >= it) { "setting value is below minimum" } }
            constraints.max?.let { require(value <= it) { "setting value is above maximum" } }
        }
        constraints.allowedValues?.let { allowed ->
            require(setting.value.toString() in allowed) { "setting value is not allowed" }
        }
    }
}
