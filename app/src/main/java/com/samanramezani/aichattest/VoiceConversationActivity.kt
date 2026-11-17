package com.samanramezani.aichattest

import android.view.KeyEvent
import android.content.Intent
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.ui.platform.LocalLayoutDirection
import androidx.compose.ui.unit.LayoutDirection
import com.samanramezani.aichattest.ui.components.AppTheme
import com.samanramezani.aichattest.ui.voice.VoiceConversationHost
import com.samanramezani.aichattest.voice.VoiceAudioInputGuard

/** Independent Activity and lifecycle for the full-screen voice conversation. */
class VoiceConversationActivity : ComponentActivity() {
    override fun dispatchKeyEvent(event: KeyEvent): Boolean {
        if (event.action == KeyEvent.ACTION_DOWN &&
            (event.keyCode == KeyEvent.KEYCODE_VOLUME_UP || event.keyCode == KeyEvent.KEYCODE_VOLUME_DOWN)
        ) {
            VoiceAudioInputGuard.suppressAfterVolumeKey()
        }
        return super.dispatchKeyEvent(event)
    }

    override fun onCreate(savedInstanceState: android.os.Bundle?) {
        super.onCreate(savedInstanceState)
        val openedFromWake = intent?.getBooleanExtra(
            AlwaysOnKoochikService.EXTRA_OPEN_VOICE_FROM_ALWAYS_ON, false,
        ) == true
        if (openedFromWake && android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O_MR1) {
            setShowWhenLocked(true)
            setTurnScreenOn(true)
        }
        val container = (application as? AIChatApplication)?.container ?: AppContainer(this)
        setContent {
            CompositionLocalProvider(LocalLayoutDirection provides LayoutDirection.Rtl) {
                AppTheme {
                    VoiceConversationHost(
                        container = container,
                        onBack = { finish() },
                        onHistoryChanged = {},
                        onOpenSettings = {
                            startActivity(
                                Intent(this, MainActivity::class.java)
                                    .putExtra(MainActivity.EXTRA_OPEN_SETTINGS_SECTION, "صدا"),
                            )
                            finish()
                        },
                    )
                }
            }
        }
    }
}
