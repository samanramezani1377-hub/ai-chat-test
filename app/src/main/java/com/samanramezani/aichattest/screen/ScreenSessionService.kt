package com.samanramezani.aichattest.screen

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.PixelFormat
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.os.SystemClock
import android.util.DisplayMetrics
import android.view.WindowManager
import android.view.Display
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import com.samanramezani.aichattest.R
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.max

/**
 * Explicitly consented MediaProjection session. The stream stays active, but only a
 * low-rate latest frame is retained in memory. Vision inference is not run here.
 */
class ScreenSessionService : Service() {
    companion object {
        const val ACTION_START = "com.samanramezani.aichattest.screen.CAPTURE_START"
        const val ACTION_STOP = "com.samanramezani.aichattest.screen.CAPTURE_STOP"
        const val EXTRA_RESULT_CODE = "projection_result_code"
        const val EXTRA_RESULT_DATA = "projection_result_data"
        private const val CHANNEL = "screen_session"
        private const val NOTIFICATION_ID = 7411
        @Volatile var isRunning: Boolean = false
            private set
    }

    private var projection: MediaProjection? = null
    private var virtualDisplay: VirtualDisplay? = null
    private var reader: ImageReader? = null
    private var worker: HandlerThread? = null
    private var handler: Handler? = null
    private val samplePending = AtomicBoolean(false)
    private var lastSampleAt = 0L
    private var displayWidth = 0
    private var displayHeight = 0
    private var densityDpi = 0
    private var displayManager: DisplayManager? = null

    private val displayListener = object : DisplayManager.DisplayListener {
        override fun onDisplayAdded(displayId: Int) = Unit
        override fun onDisplayRemoved(displayId: Int) = Unit
        override fun onDisplayChanged(displayId: Int) {
            if (displayId == Display.DEFAULT_DISPLAY) handler?.post { resizeCaptureForDisplayChange() }
        }
    }

    private val projectionCallback = object : MediaProjection.Callback() {
        override fun onStop() {
            stopCapture()
            stopSelf()
        }
    }

    override fun onCreate() {
        super.onCreate()
        createChannel()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            stopCapture()
            stopSelf()
            return START_NOT_STICKY
        }
        if (intent?.action != ACTION_START || projection != null) return START_NOT_STICKY

        val resultCode = intent.getIntExtra(EXTRA_RESULT_CODE, 0)
        val data = if (Build.VERSION.SDK_INT >= 33) intent.getParcelableExtra(EXTRA_RESULT_DATA, Intent::class.java)
            else @Suppress("DEPRECATION") intent.getParcelableExtra(EXTRA_RESULT_DATA)
        if (resultCode == 0 || data == null) {
            stopSelf()
            return START_NOT_STICKY
        }

        try {
            startForegroundCompat()
            val manager = getSystemService(MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
            projection = manager.getMediaProjection(resultCode, data)
            val wm = getSystemService(WINDOW_SERVICE) as WindowManager
            val metrics = DisplayMetrics()
            @Suppress("DEPRECATION") wm.defaultDisplay.getRealMetrics(metrics)
            displayWidth = metrics.widthPixels
            displayHeight = metrics.heightPixels
            densityDpi = metrics.densityDpi
            worker = HandlerThread("screen-frame-capture").also { it.start() }
            handler = Handler(worker!!.looper)
            reader = ImageReader.newInstance(displayWidth, displayHeight, PixelFormat.RGBA_8888, 3)
            projection?.registerCallback(projectionCallback, handler)
            virtualDisplay = projection?.createVirtualDisplay(
                "DinaScreenSession", displayWidth, displayHeight, densityDpi,
                DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR, reader!!.surface, null, handler
            )
            reader?.setOnImageAvailableListener({ imageReader -> scheduleSample(imageReader) }, handler)
            displayManager = getSystemService(DisplayManager::class.java).also {
                it.registerDisplayListener(displayListener, handler)
            }
            isRunning = true
            getSharedPreferences("screen_assistant", MODE_PRIVATE).edit().putBoolean("capture_running", true).apply()
        } catch (_: Throwable) {
            stopCapture()
            stopSelf()
        }
        return START_NOT_STICKY
    }

    /** Keep the capture surface aligned with rotation and display-size changes. */
    private fun resizeCaptureForDisplayChange() {
        val currentReader = reader ?: return
        if (displayManager == null) return
        val wm = getSystemService(WINDOW_SERVICE) as WindowManager
        val metrics = DisplayMetrics()
        @Suppress("DEPRECATION")
        wm.defaultDisplay.getRealMetrics(metrics)
        if (metrics.widthPixels == displayWidth &&
            metrics.heightPixels == displayHeight &&
            metrics.densityDpi == densityDpi
        ) return

        val nextReader = runCatching {
            ImageReader.newInstance(metrics.widthPixels, metrics.heightPixels, PixelFormat.RGBA_8888, 3)
        }.getOrNull() ?: return
        nextReader.setOnImageAvailableListener({ imageReader -> scheduleSample(imageReader) }, handler)
        try {
            virtualDisplay?.resize(metrics.widthPixels, metrics.heightPixels, metrics.densityDpi)
            virtualDisplay?.setSurface(nextReader.surface)
            reader = nextReader
            displayWidth = metrics.widthPixels
            displayHeight = metrics.heightPixels
            densityDpi = metrics.densityDpi
            currentReader.setOnImageAvailableListener(null, null)
            currentReader.close()
        } catch (_: Throwable) {
            nextReader.setOnImageAvailableListener(null, null)
            nextReader.close()
        }
    }

    private fun scheduleSample(imageReader: ImageReader) {
        val now = SystemClock.elapsedRealtime()
        if (now - lastSampleAt < 1_500L || !samplePending.compareAndSet(false, true)) {
            // Drain stale images so the ImageReader does not stall.
            runCatching { imageReader.acquireLatestImage()?.close() }
            return
        }
        lastSampleAt = now
        // The overlay is made transparent before the next compositor frame is sampled.
        // The app underneath remains visible; the bubble's rectangle is not blacked out.
        FloatingAssistantService.setTemporarilyTransparent(true)
        handler?.postDelayed({
            try {
                val image = imageReader.acquireLatestImage() ?: return@postDelayed
                image.use { ScreenFrameStore.publish(toJpeg(it), displayWidth, displayHeight) }
            } catch (_: Throwable) {
                // A transient frame failure is recoverable; the next frame will be sampled.
            } finally {
                FloatingAssistantService.setTemporarilyTransparent(false)
                samplePending.set(false)
            }
        }, 140L)
    }

    private fun toJpeg(image: android.media.Image): ByteArray {
        val plane = image.planes[0]
        val buffer: ByteBuffer = plane.buffer
        val pixelStride = plane.pixelStride
        val rowStride = plane.rowStride
        val rowPadding = rowStride - pixelStride * image.width
        val paddedWidth = image.width + rowPadding / pixelStride
        val full = Bitmap.createBitmap(paddedWidth, image.height, Bitmap.Config.ARGB_8888)
        full.copyPixelsFromBuffer(buffer)
        val cropped = Bitmap.createBitmap(full, 0, 0, image.width, image.height)
        val maxDimension = 1280
        val scale = minOf(1f, maxDimension.toFloat() / max(image.width, image.height))
        val resized = if (scale < 1f) Bitmap.createScaledBitmap(cropped,
            (image.width * scale).toInt().coerceAtLeast(1),
            (image.height * scale).toInt().coerceAtLeast(1), true) else cropped
        return try {
            ByteArrayOutputStream().use { out ->
                resized.compress(Bitmap.CompressFormat.JPEG, 82, out)
                out.toByteArray()
            }
        } finally {
            if (resized !== cropped && resized !== full) resized.recycle()
            if (cropped !== full) cropped.recycle()
            full.recycle()
        }
    }

    private fun startForegroundCompat() {
        val stopIntent = Intent(this, ScreenSessionService::class.java).setAction(ACTION_STOP)
        val stopPending = PendingIntent.getService(this, 7412, stopIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
        val notification = NotificationCompat.Builder(this, CHANNEL)
            .setSmallIcon(android.R.drawable.ic_menu_view)
            .setContentTitle("مشاهده صفحه فعال است")
            .setContentText("تصویر صفحه فقط در حافظه موقت نگه داشته می‌شود.")
            .setOngoing(true)
            .addAction(0, "توقف مشاهده صفحه", stopPending)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()
        if (Build.VERSION.SDK_INT >= 34) {
            ServiceCompat.startForeground(this, NOTIFICATION_ID, notification,
                android.content.pm.ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION)
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
    }

    private fun createChannel() {
        if (Build.VERSION.SDK_INT >= 26) {
            getSystemService(NotificationManager::class.java).createNotificationChannel(
                NotificationChannel(CHANNEL, "مشاهده صفحه", NotificationManager.IMPORTANCE_LOW)
            )
        }
    }

    private fun stopCapture() {
        FloatingAssistantService.setTemporarilyTransparent(false)
        isRunning = false
        getSharedPreferences("screen_assistant", MODE_PRIVATE).edit().putBoolean("capture_running", false).apply()
        ScreenFrameStore.clear()
        // Release the separately loaded native Vision model when screen observation ends.
        VisionNativeRuntime.unloadAsync()
        displayManager?.let { manager ->
            runCatching { manager.unregisterDisplayListener(displayListener) }
        }
        displayManager = null
        runCatching { virtualDisplay?.release() }
        virtualDisplay = null
        runCatching { reader?.setOnImageAvailableListener(null, null) }
        runCatching { reader?.close() }
        reader = null
        runCatching { projection?.unregisterCallback(projectionCallback) }
        runCatching { projection?.stop() }
        projection = null
        handler?.removeCallbacksAndMessages(null)
        worker?.quitSafely()
        worker = null
        handler = null
    }

    override fun onDestroy() {
        stopCapture()
        ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE)
        super.onDestroy()
    }
}
