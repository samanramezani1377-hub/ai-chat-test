package com.samanramezani.aichattest.screen

import android.app.NotificationChannel
import android.app.NotificationManager
import android.util.Log
import android.widget.Toast
import androidx.lifecycle.LifecycleService
import androidx.lifecycle.ViewModelStore
import androidx.lifecycle.ViewModelStoreOwner
import androidx.lifecycle.setViewTreeLifecycleOwner
import androidx.lifecycle.setViewTreeViewModelStoreOwner
import androidx.savedstate.SavedStateRegistry
import androidx.savedstate.SavedStateRegistryController
import androidx.savedstate.SavedStateRegistryOwner
import androidx.savedstate.setViewTreeSavedStateRegistryOwner
import android.content.Intent
import android.content.res.Configuration
import android.graphics.PixelFormat
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Build
import android.provider.Settings
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager
import android.widget.TextView
import androidx.compose.ui.platform.ComposeView
import androidx.compose.ui.platform.ViewCompositionStrategy
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import kotlin.math.abs

/** User-enabled floating Dina bubble and in-place chat window. Neither tap launches the app. */
class FloatingAssistantService : LifecycleService(), SavedStateRegistryOwner, ViewModelStoreOwner {
    private val savedStateController = SavedStateRegistryController.create(this)
    override val savedStateRegistry: SavedStateRegistry
        get() = savedStateController.savedStateRegistry
    override val viewModelStore: ViewModelStore = ViewModelStore()
    companion object {
        const val ACTION_ENABLE = "com.samanramezani.aichattest.screen.FLOATING_ENABLE"
        const val ACTION_DISABLE = "com.samanramezani.aichattest.screen.FLOATING_DISABLE"
        private const val CHANNEL = "floating_assistant"
        private const val NOTIFICATION_ID = 7410
        private const val PREFS = "screen_assistant"
        private const val KEY_X = "bubble_x"
        private const val KEY_Y = "bubble_y"

        @Volatile private var instance: FloatingAssistantService? = null
        val isRunning: Boolean get() = instance != null
        private const val KEY_ENABLED = "overlay_enabled"

        /** Hide every Dina overlay for one compositor frame so capture sees the app beneath it. */
        fun setTemporarilyTransparent(transparent: Boolean) {
            instance?.setOverlayAlpha(if (transparent) 0f else 1f)
        }
    }

    private lateinit var windowManager: WindowManager
    private var bubble: TextView? = null
    private var popup: ComposeView? = null
    private var bubbleParams: WindowManager.LayoutParams? = null
    private var popupParams: WindowManager.LayoutParams? = null
    private var downX = 0f
    private var downY = 0f
    private var startX = 0
    private var startY = 0
    private var moved = false

    override fun onCreate() {
        savedStateController.performAttach()
        super.onCreate()
        savedStateController.performRestore(null)
        instance = this
        getSharedPreferences(PREFS, MODE_PRIVATE).edit().putBoolean(KEY_ENABLED, false).apply()
        createChannel()
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        val params = bubbleParams ?: return
        val size = params.width
        params.x = BubbleBounds.clampX(params.x, resources.displayMetrics.widthPixels, size)
        params.y = BubbleBounds.clampY(params.y, resources.displayMetrics.heightPixels, size)
        bubble?.let { view -> runCatching { windowManager.updateViewLayout(view, params) } }
        getSharedPreferences(PREFS, MODE_PRIVATE).edit().putInt(KEY_X, params.x).putInt(KEY_Y, params.y).apply()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        super.onStartCommand(intent, flags, startId)
        if (intent?.action == ACTION_DISABLE) {
            getSharedPreferences(PREFS, MODE_PRIVATE).edit().putBoolean(KEY_ENABLED, false).apply()
            stopSelf()
            return START_NOT_STICKY
        }
        if (!Settings.canDrawOverlays(this)) {
            getSharedPreferences(PREFS, MODE_PRIVATE).edit().putBoolean(KEY_ENABLED, false).apply()
            stopSelf()
            return START_NOT_STICKY
        }
        startForegroundCompat()
        if (bubble == null) {
            try {
                showBubble()
            } catch (t: Throwable) {
                Log.e("DinaOverlay", "Unable to create floating bubble", t)
                getSharedPreferences(PREFS, MODE_PRIVATE).edit().putBoolean(KEY_ENABLED, false).apply()
                stopSelf()
                return START_NOT_STICKY
            }
        }
        getSharedPreferences(PREFS, MODE_PRIVATE).edit().putBoolean(KEY_ENABLED, true).apply()
        return START_STICKY
    }

    private fun startForegroundCompat() {
        val notification = NotificationCompat.Builder(this, CHANNEL)
            .setSmallIcon(android.R.drawable.ic_menu_view)
            .setContentTitle("دستیار شناور دینا فعال است")
            .setContentText("با لمس حباب، پنجره گفت‌وگوی شناور باز می‌شود.")
            .setOngoing(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()
        if (Build.VERSION.SDK_INT >= 34) {
            ServiceCompat.startForeground(this, NOTIFICATION_ID, notification, android.content.pm.ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE)
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
    }

    private fun createChannel() {
        if (Build.VERSION.SDK_INT >= 26) {
            getSystemService(NotificationManager::class.java).createNotificationChannel(
                NotificationChannel(CHANNEL, "دستیار شناور", NotificationManager.IMPORTANCE_LOW)
            )
        }
    }

    private fun showBubble() {
        windowManager = getSystemService(WINDOW_SERVICE) as WindowManager
        val prefs = getSharedPreferences(PREFS, MODE_PRIVATE)
        val view = TextView(this).apply {
            text = "🤖"
            textSize = 22f
            setTypeface(typeface, Typeface.BOLD)
            gravity = Gravity.CENTER
            setTextColor(android.graphics.Color.WHITE)
            setBackground(GradientDrawable().apply {
                shape = GradientDrawable.OVAL
                setColor(0xFF4F46E5.toInt())
                setStroke((2 * resources.displayMetrics.density).toInt(), android.graphics.Color.WHITE)
            })
            elevation = 12f
            contentDescription = "باز کردن گفت‌وگوی شناور دینا"
        }
        val size = (52 * resources.displayMetrics.density).toInt()
        val layout = WindowManager.LayoutParams(
            size, size, WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
            PixelFormat.TRANSLUCENT
        ).apply {
            gravity = Gravity.TOP or Gravity.LEFT
            x = BubbleBounds.clampX(prefs.getInt(KEY_X, 0), resources.displayMetrics.widthPixels, size)
            y = BubbleBounds.clampY(prefs.getInt(KEY_Y, 260), resources.displayMetrics.heightPixels, size)
        }
        view.setOnTouchListener { v, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    downX = event.rawX; downY = event.rawY
                    startX = layout.x; startY = layout.y; moved = false
                    true
                }
                MotionEvent.ACTION_MOVE -> {
                    val dx = (event.rawX - downX).toInt()
                    val dy = (event.rawY - downY).toInt()
                    if (abs(dx) + abs(dy) > 8) moved = true
                    layout.x = BubbleBounds.clampX(startX + dx, resources.displayMetrics.widthPixels, size)
                    layout.y = BubbleBounds.clampY(startY + dy, resources.displayMetrics.heightPixels, size)
                    runCatching { windowManager.updateViewLayout(v, layout) }
                    true
                }
                MotionEvent.ACTION_UP -> {
                    if (moved) {
                        val screenWidth = resources.displayMetrics.widthPixels
                        val screenHeight = resources.displayMetrics.heightPixels
                        layout.x = if (layout.x + size / 2 < screenWidth / 2) 0 else BubbleBounds.clampX(screenWidth - size, screenWidth, size)
                        layout.y = BubbleBounds.clampY(layout.y, screenHeight, size)
                        prefs.edit().putInt(KEY_X, layout.x).putInt(KEY_Y, layout.y).apply()
                        runCatching { windowManager.updateViewLayout(v, layout) }
                    } else {
                        if (popup == null) {
                            runCatching { showPopup() }.onFailure { error ->
                                Log.e("DinaOverlay", "Unable to open floating chat popup", error)
                                Toast.makeText(this@FloatingAssistantService, "پنجره گفت‌وگو باز نشد: ${error.message ?: error.javaClass.simpleName}", Toast.LENGTH_LONG).show()
                            }
                        } else closePopup()
                    }
                    true
                }
                else -> true
            }
        }
        windowManager.addView(view, layout)
        bubble = view
        bubbleParams = layout
    }

    private fun showPopup() {
        if (popup != null) return
        val density = resources.displayMetrics.density
        val screenWidth = resources.displayMetrics.widthPixels
        val screenHeight = resources.displayMetrics.heightPixels
        val width = minOf((380 * density).toInt(), (screenWidth - 20 * density).toInt()).coerceAtLeast(1)
        val height = minOf((540 * density).toInt(), (screenHeight - 80 * density).toInt()).coerceAtLeast(1)
        val view = ComposeView(this).apply {
            this.setViewTreeLifecycleOwner(this@FloatingAssistantService)
            this.setViewTreeSavedStateRegistryOwner(this@FloatingAssistantService)
            this.setViewTreeViewModelStoreOwner(this@FloatingAssistantService)
            setViewCompositionStrategy(ViewCompositionStrategy.DisposeOnDetachedFromWindow)
            setContent { FloatingAssistantPopup(onClose = { closePopup() }) }
        }
        val layout = WindowManager.LayoutParams(
            width,
            height,
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN or WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL,
            PixelFormat.TRANSLUCENT,
        ).apply {
            gravity = Gravity.BOTTOM or Gravity.CENTER_HORIZONTAL
            y = (72 * density).toInt()
            softInputMode = WindowManager.LayoutParams.SOFT_INPUT_ADJUST_RESIZE
            setTitle("Dina floating chat")
        }
        try {
            windowManager.addView(view, layout)
            popup = view
            popupParams = layout
        } catch (t: Throwable) {
            view.disposeComposition()
            throw t
        }
    }

    private fun closePopup() {
        popup?.let { view ->
            runCatching { windowManager.removeView(view) }
            view.disposeComposition()
        }
        popup = null
        popupParams = null
    }

    private fun setOverlayAlpha(alpha: Float) {
        val value = alpha.coerceIn(0f, 1f)
        bubble?.post { bubble?.alpha = value; bubble?.visibility = if (value == 0f) View.INVISIBLE else View.VISIBLE }
        popup?.post { popup?.alpha = value; popup?.visibility = if (value == 0f) View.INVISIBLE else View.VISIBLE }
    }

    override fun onDestroy() {
        instance = null
        getSharedPreferences(PREFS, MODE_PRIVATE).edit().putBoolean(KEY_ENABLED, false).apply()
        closePopup()
        bubble?.let { runCatching { windowManager.removeView(it) } }
        bubble = null
        bubbleParams = null
        viewModelStore.clear()
        super.onDestroy()
    }
}
