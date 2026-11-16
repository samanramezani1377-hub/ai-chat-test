package com.samanramezani.aichattest.screen

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.graphics.PixelFormat
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Build
import android.os.IBinder
import android.provider.Settings
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager
import android.widget.TextView
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import com.samanramezani.aichattest.MainActivity
import com.samanramezani.aichattest.R
import kotlin.math.abs

/** User-enabled, draggable edge robot. It does not own or read screen content. */
class FloatingAssistantService : Service() {
    companion object {
        const val ACTION_ENABLE = "com.samanramezani.aichattest.screen.FLOATING_ENABLE"
        const val ACTION_DISABLE = "com.samanramezani.aichattest.screen.FLOATING_DISABLE"
        private const val CHANNEL = "floating_assistant"
        private const val NOTIFICATION_ID = 7410
        private const val PREFS = "screen_assistant"
        private const val KEY_X = "bubble_x"
        private const val KEY_Y = "bubble_y"

        @Volatile private var instance: FloatingAssistantService? = null

        fun setTemporarilyTransparent(transparent: Boolean) {
            instance?.setBubbleAlpha(if (transparent) 0f else 1f)
        }
    }

    private lateinit var windowManager: WindowManager
    private var bubble: TextView? = null
    private var params: WindowManager.LayoutParams? = null
    private var downX = 0f
    private var downY = 0f
    private var startX = 0
    private var startY = 0
    private var moved = false

    override fun onCreate() {
        super.onCreate()
        instance = this
        createChannel()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_DISABLE) {
            stopSelf()
            return START_NOT_STICKY
        }
        if (!Settings.canDrawOverlays(this)) {
            stopSelf()
            return START_NOT_STICKY
        }
        startForegroundCompat()
        if (bubble == null) {
            try { showBubble() } catch (_: Throwable) { stopSelf(); return START_NOT_STICKY }
        }
        return START_STICKY
    }

    private fun startForegroundCompat() {
        val notification = NotificationCompat.Builder(this, CHANNEL)
            .setSmallIcon(android.R.drawable.ic_menu_view)
            .setContentTitle("دستیار شناور فعال است")
            .setContentText("برای باز کردن دستیار، روی حباب دینا بزن.")
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
            setBackground(GradientDrawable().apply { shape = GradientDrawable.OVAL; setColor(0xFF4F46E5.toInt()); setStroke((2 * resources.displayMetrics.density).toInt(), android.graphics.Color.WHITE) })
            elevation = 12f
            contentDescription = "باز کردن دستیار دینا"
        }
        val size = (52 * resources.displayMetrics.density).toInt()
        val layout = WindowManager.LayoutParams(
            size, size, WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
            PixelFormat.TRANSLUCENT
        ).apply {
            gravity = Gravity.TOP or Gravity.LEFT
            x = prefs.getInt(KEY_X, 0)
            y = prefs.getInt(KEY_Y, 260)
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
                    layout.x = (startX + dx).coerceAtLeast(0)
                    layout.y = (startY + dy).coerceAtLeast(0)
                    runCatching { windowManager.updateViewLayout(v, layout) }
                    true
                }
                MotionEvent.ACTION_UP -> {
                    if (moved) {
                        val screenWidth = resources.displayMetrics.widthPixels
                        layout.x = if (layout.x + size / 2 < screenWidth / 2) 0 else (screenWidth - size).coerceAtLeast(0)
                        prefs.edit().putInt(KEY_X, layout.x).putInt(KEY_Y, layout.y).apply()
                        runCatching { windowManager.updateViewLayout(v, layout) }
                    } else {
                        val open = Intent(this, MainActivity::class.java)
                            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_SINGLE_TOP)
                            .putExtra("open_screen_assistant", true)
                        startActivity(open)
                    }
                    true
                }
                else -> true
            }
        }
        windowManager.addView(view, layout)
        bubble = view
        params = layout
    }

    private fun setBubbleAlpha(alpha: Float) {
        val view = bubble ?: return
        view.post {
            view.alpha = alpha.coerceIn(0f, 1f)
            view.visibility = if (alpha <= 0f) View.INVISIBLE else View.VISIBLE
        }
    }

    override fun onDestroy() {
        instance = null
        bubble?.let { runCatching { windowManager.removeView(it) } }
        bubble = null
        params = null
        super.onDestroy()
    }
}
