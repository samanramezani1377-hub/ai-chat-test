package com.samanramezani.aichattest

import android.app.Application

class AIChatApplication : Application() {
    val container: AppContainer by lazy { AppContainer(this) }
}
