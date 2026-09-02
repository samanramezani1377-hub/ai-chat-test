package com.samanramezani.aichattest

import android.app.Activity
import android.os.Bundle
import androidx.activity.compose.setContent

class MainActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val container = (application as? AIChatApplication)?.container ?: AppContainer(this)
        setContent { MainScreen(container) }
    }
}
