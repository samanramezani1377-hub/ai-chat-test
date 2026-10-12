package com.samanramezani.aichattest

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import com.k2fsa.sherpa.onnx.FeatureConfig
import com.k2fsa.sherpa.onnx.KeywordSpotter
import com.k2fsa.sherpa.onnx.KeywordSpotterConfig
import com.k2fsa.sherpa.onnx.OnlineModelConfig
import com.k2fsa.sherpa.onnx.OnlineTransducerModelConfig
import com.k2fsa.sherpa.onnx.OnlineStream
import com.samanramezani.aichattest.voice.WakeWordModelEntry
import com.samanramezani.aichattest.voice.WakeWordModelStore
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Optional, user-enabled microphone foreground service for the local "دینا" wake phrase.
 * The imported KWS package must be compatible with Persian and contain keywords.txt configured
 * for that exact phrase. Android may still restrict automatic activity launches on some devices;
 * a high-priority notification is kept as a reliable user-tap fallback.
 */
class WakeWordForegroundService : Service() {
    companion object {
        private const val TAG = "DinaWakeWord"
        private const val CHANNEL_ID = "dina_wake_word"
        private const val LISTENING_NOTIFICATION_ID = 4201
        private const val DETECTED_NOTIFICATION_ID = 4202
        const val ACTION_START = "com.samanramezani.aichattest.action.START_DINA_WAKE_WORD"
        const val ACTION_STOP = "com.samanramezani.aichattest.action.STOP_DINA_WAKE_WORD"
        const val EXTRA_OPEN_VOICE_FROM_WAKE = "open_voice_from_wake_word"
        const val PREFS = "dina_wake_word"
        const val PREF_MODEL_DIRECTORY = "model_directory"
        const val PREF_ENABLED = "enabled"
        const val PREF_ARMED = "armed"
        private const val SAMPLE_RATE = 16000
        private const val SAMPLES_PER_CHUNK = 1600

        fun start(context: android.content.Context, modelDirectory: String) {
            context.getSharedPreferences(PREFS, android.content.Context.MODE_PRIVATE).edit()
                .putString(PREF_MODEL_DIRECTORY, modelDirectory)
                .putBoolean(PREF_ENABLED, true)
                .putBoolean(PREF_ARMED, true)
                .apply()
            ContextCompat.startForegroundService(
                context,
                Intent(context, WakeWordForegroundService::class.java).setAction(ACTION_START),
            )
        }

        fun pause(context: android.content.Context) {
            context.getSharedPreferences(PREFS, android.content.Context.MODE_PRIVATE).edit().putBoolean(PREF_ENABLED, false).apply()
            context.stopService(Intent(context, WakeWordForegroundService::class.java).setAction(ACTION_STOP))
        }

        fun stop(context: android.content.Context) {
            context.getSharedPreferences(PREFS, android.content.Context.MODE_PRIVATE).edit()
                .putBoolean(PREF_ENABLED, false).putBoolean(PREF_ARMED, false).apply()
            context.stopService(Intent(context, WakeWordForegroundService::class.java).setAction(ACTION_STOP))
            context.getSystemService(NotificationManager::class.java).cancel(DETECTED_NOTIFICATION_ID)
        }
    }

    private val running = AtomicBoolean(false)
    @Volatile private var worker: Thread? = null
    @Volatile private var audioRecord: AudioRecord? = null
    @Volatile private var spotter: KeywordSpotter? = null
    @Volatile private var stream: OnlineStream? = null

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            running.set(false)
            stopSelfResult(startId)
            return START_NOT_STICKY
        }
        val prefs = getSharedPreferences(PREFS, MODE_PRIVATE)
        val modelPath = prefs.getString(PREF_MODEL_DIRECTORY, null)
        if (modelPath.isNullOrBlank() || !prefs.getBoolean(PREF_ENABLED, false)) {
            stopSelfResult(startId)
            return START_NOT_STICKY
        }
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                startForeground(
                    LISTENING_NOTIFICATION_ID,
                    buildListeningNotification(),
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE,
                )
            } else {
                startForeground(LISTENING_NOTIFICATION_ID, buildListeningNotification())
            }
        } catch (t: Throwable) {
            prefs.edit().putBoolean(PREF_ENABLED, false).putBoolean(PREF_ARMED, false).apply()
            Log.e(TAG, "Could not start microphone foreground service", t)
            stopSelfResult(startId)
            return START_NOT_STICKY
        }
        if (running.compareAndSet(false, true)) {
            worker = Thread({ listen(modelPath) }, "DinaWakeWord-KWS").apply { start() }
        }
        return START_NOT_STICKY
    }

    private fun listen(modelPath: String) {
        var detected = false
        try {
            val modelStore = WakeWordModelStore(contentResolver, File(filesDir, "wake-word-models"))
            val entry = modelStore.list().firstOrNull { it.directory.absolutePath == modelPath }
                ?: error("VOICE-KWS-001: مدل بیدارباش انتخاب‌شده پیدا نشد.")
            val config = KeywordSpotterConfig().apply {
                featConfig = FeatureConfig().apply { sampleRate = SAMPLE_RATE; featureDim = 80 }
                modelConfig = OnlineModelConfig().apply {
                    transducer = OnlineTransducerModelConfig().apply {
                        encoder = entry.encoder.absolutePath
                        decoder = entry.decoder.absolutePath
                        joiner = entry.joiner.absolutePath
                    }
                    tokens = entry.tokens.absolutePath
                    numThreads = 1
                    provider = "cpu"
                    modelType = entry.modelType
                    modelingUnit = entry.modelingUnit
                    bpeVocab = entry.bpeVocab?.absolutePath.orEmpty()
                }
                keywordsFile = entry.keywords.absolutePath
                keywordsScore = 1.5f
                keywordsThreshold = 0.25f
                maxActivePaths = 4
                numTrailingBlanks = 2
            }
            spotter = KeywordSpotter(config = config)
            stream = spotter!!.createStream()
            val minBytes = AudioRecord.getMinBufferSize(
                SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
            )
            check(minBytes > 0) { "VOICE-KWS-002: اندازهٔ بافر میکروفون معتبر نیست." }
            val recorder = AudioRecord(
                MediaRecorder.AudioSource.VOICE_RECOGNITION,
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                maxOf(minBytes * 2, SAMPLE_RATE / 5 * 2),
            )
            check(recorder.state == AudioRecord.STATE_INITIALIZED) { "VOICE-KWS-003: راه‌اندازی میکروفون ناموفق بود." }
            audioRecord = recorder
            recorder.startRecording()
            val pcm = ShortArray(SAMPLES_PER_CHUNK)
            while (running.get()) {
                val count = recorder.read(pcm, 0, pcm.size)
                if (count <= 0) continue
                val samples = FloatArray(count) { pcm[it] / 32768.0f }
                val activeStream = stream ?: break
                val activeSpotter = spotter ?: break
                activeStream.acceptWaveform(samples, SAMPLE_RATE)
                while (activeSpotter.isReady(activeStream)) {
                    activeSpotter.decode(activeStream)
                    val result = activeSpotter.getResult(activeStream).keyword
                    if (result.isNotBlank() &&
                        (result.contains("دینا") || result.contains("dina", ignoreCase = true))) {
                        detected = true
                        running.set(false)
                        break
                    }
                }
                if (detected) break
            }
        } catch (t: Throwable) {
            Log.e(TAG, "Wake-word listener stopped", t)
            getSharedPreferences(PREFS, MODE_PRIVATE).edit().putBoolean(PREF_ENABLED, false).putBoolean(PREF_ARMED, false).apply()
            postWakeNotification("شنود بیدارباش متوقف شد؛ برای جزئیات، گزارش برنامه را بررسی کن.", openVoice = false)
        } finally {
            running.set(false)
            runCatching { audioRecord?.stop() }
            runCatching { audioRecord?.release() }
            audioRecord = null
            runCatching { stream?.release() }
            stream = null
            runCatching { spotter?.release() }
            spotter = null
            if (detected) {
                getSharedPreferences(PREFS, MODE_PRIVATE).edit().putBoolean(PREF_ENABLED, false).apply()
                postWakeNotification("عبارت «دینا» شنیده شد؛ در حال باز کردن مکالمهٔ صوتی.", openVoice = true)
                runCatching {
                    startActivity(Intent(this, MainActivity::class.java).apply {
                        flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
                        putExtra(EXTRA_OPEN_VOICE_FROM_WAKE, true)
                    })
                }.onFailure { Log.w(TAG, "Automatic activity launch blocked; notification remains available", it) }
            }
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
        }
    }

    private fun postWakeNotification(message: String, openVoice: Boolean) {
        val intent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
            putExtra(EXTRA_OPEN_VOICE_FROM_WAKE, openVoice)
        }
        val pending = PendingIntent.getActivity(
            this, if (openVoice) 4202 else 4203, intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(applicationInfo.icon.takeIf { it != 0 } ?: android.R.drawable.ic_btn_speak_now)
            .setContentTitle(if (openVoice) "دینا · مکالمهٔ صوتی" else "دینا · بیدارباش")
            .setContentText(message)
            .setStyle(NotificationCompat.BigTextStyle().bigText(message))
            .setContentIntent(pending)
            .setAutoCancel(true)
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .build()
        getSystemService(NotificationManager::class.java).notify(DETECTED_NOTIFICATION_ID, notification)
    }

    private fun buildListeningNotification(): Notification =
        NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(applicationInfo.icon.takeIf { it != 0 } ?: android.R.drawable.ic_btn_speak_now)
            .setContentTitle("بیدارباش «دینا» فعال است")
            .setContentText("میکروفون برای تشخیص محلی عبارت فعال‌کننده در حال استفاده است.")
            .setOngoing(true)
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            getSystemService(NotificationManager::class.java).createNotificationChannel(
                NotificationChannel(CHANNEL_ID, "بیدارباش دینا", NotificationManager.IMPORTANCE_HIGH).apply {
                    description = "وضعیت شنود محلی و اعلان تشخیص عبارت دینا"
                    setShowBadge(false)
                },
            )
        }
    }

    override fun onDestroy() {
        running.set(false)
        // Stop capture to unblock the worker; the worker owns and releases all native handles.
        runCatching { audioRecord?.stop() }
        worker?.interrupt()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

}
