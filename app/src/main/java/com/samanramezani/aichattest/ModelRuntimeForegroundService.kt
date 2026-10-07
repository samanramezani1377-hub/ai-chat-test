package com.samanramezani.aichattest

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.os.Build
import androidx.core.app.NotificationCompat

/**
 * Keeps the native llama.cpp/OpenCL runtime process important while a model is active.
 * The service does not create another runtime or duplicate model memory.
 */
class ModelRuntimeForegroundService : Service() {
    companion object {
        private const val CHANNEL_ID = "model_runtime"
        private const val NOTIFICATION_ID = 4101
        const val ACTION_START = "com.samanramezani.aichattest.action.START_MODEL_RUNTIME"
        const val ACTION_STOP = "com.samanramezani.aichattest.action.STOP_MODEL_RUNTIME"

        fun start(context: android.content.Context) {
            val intent = Intent(context, ModelRuntimeForegroundService::class.java).setAction(ACTION_START)
            androidx.core.content.ContextCompat.startForegroundService(context, intent)
        }

        fun stop(context: android.content.Context) {
            context.stopService(Intent(context, ModelRuntimeForegroundService::class.java).setAction(ACTION_STOP))
        }
    }

    override fun onCreate() {
        super.onCreate()
        createChannel()
        startForeground(NOTIFICATION_ID, buildNotification())
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelfResult(startId)
            return START_NOT_STICKY
        }
        return START_NOT_STICKY
    }

    override fun onBind(intent: Intent?) = null

    private fun createChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "اجرای مدل محلی",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "حفظ اجرای مدل محلی و شتاب‌دهنده GPU هنگام رفتن برنامه به پس‌زمینه"
                setShowBadge(false)
            }
            getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
        }
    }

    private fun buildNotification(): Notification =
        NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(applicationInfo.icon.takeIf { it != 0 } ?: android.R.drawable.stat_sys_warning)
            .setContentTitle("مدل محلی فعال است")
            .setContentText("اجرای llama.cpp/OpenCL در پس‌زمینه حفظ می‌شود")
            .setOngoing(true)
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()
}
