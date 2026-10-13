package com.samanramezani.aichattest

import android.annotation.SuppressLint
import android.os.Bundle
import android.view.KeyEvent
import com.samanramezani.aichattest.voice.VoiceAudioInputGuard
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent

class MainActivity : ComponentActivity() {
    @SuppressLint("RestrictedApi")
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
        val openVoiceFromWake = intent?.getBooleanExtra(AlwaysOnKoochikService.EXTRA_OPEN_VOICE_FROM_WAKE, false) == true
        if (openVoiceFromWake) {
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O_MR1) {
                setShowWhenLocked(true)
                setTurnScreenOn(true)
            } else {
                window.addFlags(WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED or WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON)
            }
        }
        val container = (application as? AIChatApplication)?.container ?: AppContainer(this)
        setContent {
            MainScreen(container, openVoiceInitially = openVoiceFromWake)
        }
    }
}
