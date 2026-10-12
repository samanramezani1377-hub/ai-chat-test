package com.samanramezani.aichattest

import android.os.Bundle
import android.view.KeyEvent
import com.samanramezani.aichattest.voice.VoiceAudioInputGuard
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent

class MainActivity : ComponentActivity() {
    override fun dispatchKeyEvent(event: KeyEvent): Boolean {
        if (event.action == KeyEvent.ACTION_DOWN &&
            (event.keyCode == KeyEvent.KEYCODE_VOLUME_UP || event.keyCode == KeyEvent.KEYCODE_VOLUME_DOWN)
        ) {
            VoiceAudioInputGuard.suppressAfterVolumeKey()
        }
        return super.dispatchKeyEvent(event)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val container = (application as? AIChatApplication)?.container ?: AppContainer(this)
        setContent {
            MainScreen(container)
        }
    }
}
