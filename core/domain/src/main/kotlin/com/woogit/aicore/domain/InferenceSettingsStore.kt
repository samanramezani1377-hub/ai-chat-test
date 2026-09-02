package com.woogit.aicore.domain

/** Process-wide inference controls used by the Android app and runtime. */
object InferenceSettingsStore {
    @Volatile
    var current: InferenceSettings = InferenceSettings()
}
