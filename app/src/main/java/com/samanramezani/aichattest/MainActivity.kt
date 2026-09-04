package com.samanramezani.aichattest

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import com.woogit.aicore.domain.ApiProviderConfigStore

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val context = applicationContext
        val stored = ApiConfigStore(context).load()
        ApiProviderConfigStore.current = stored
        val container = (application as? AIChatApplication)?.container ?: AppContainer(this)
        var configured by mutableStateOf(stored.apiKey.isNotBlank())
        setContent {
            if (configured) {
                MainScreen(container)
            } else {
                ApiSetupScreen(stored) { config ->
                    ApiConfigStore(context).save(config)
                    ApiProviderConfigStore.current = config
                    configured = true
                }
            }
        }
    }
}
