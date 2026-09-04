package com.woogit.aicore.domain

/** Process-wide remote model configuration. The API key is never persisted by Core or committed to source. */
object ApiProviderConfigStore {
    @Volatile
    var current: ApiProviderConfig = ApiProviderPresets.deepSeek
}
