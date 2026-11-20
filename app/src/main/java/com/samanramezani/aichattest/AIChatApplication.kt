package com.samanramezani.aichattest

import android.app.Application

class AIChatApplication : Application() {
    val container: AppContainer by lazy { AppContainer(this) }

    override fun onCreate() {
        super.onCreate()
        // Recover native fatal-signal records from the previous process and
        // install a persistent uncaught Java/Kotlin exception handler.
        CrashLogStore.initialize(this)
    }
}
