package com.samanramezani.aichattest

import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.media.ToneGenerator
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.content.ContextCompat
import com.k2fsa.sherpa.onnx.FeatureConfig
import com.k2fsa.sherpa.onnx.KeywordSpotter
import com.k2fsa.sherpa.onnx.KeywordSpotterConfig
import com.k2fsa.sherpa.onnx.OnlineModelConfig
import com.k2fsa.sherpa.onnx.OnlineTransducerModelConfig
import com.k2fsa.sherpa.onnx.OnlineStream
import com.samanramezani.aichattest.voice.WakeWordModelEntry
import com.samanramezani.aichattest.voice.WakeWordModelStore
import com.samanramezani.aichattest.voice.isDinaWakeKeywordLabel
import com.samanramezani.aichattest.voice.shouldKeepWakeServiceForConversation
import com.samanramezani.aichattest.voice.VoiceModelStore
import com.samanramezani.aichattest.voice.VoiceModelKind
import com.samanramezani.aichattest.ui.voice.VoiceConversationController
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
        const val ACTION_PAUSE = "com.samanramezani.aichattest.action.PAUSE_DINA_WAKE_WORD"
        const val ACTION_STOP_CONVERSATION = "com.samanramezani.aichattest.action.STOP_DINA_CONVERSATION"
        const val EXTRA_OPEN_VOICE_FROM_WAKE = "open_voice_from_wake_word"
        const val EXTRA_CONVERSATION_STARTED_IN_SERVICE = "dina_conversation_started_in_service"
        const val PREF_STT_MODEL_ID = "voice_stt_model_id"
        const val PREF_TTS_MODEL_ID = "voice_tts_model_id"
        const val PREF_CONVERSATION_STATE = "conversation_state"
        const val PREF_WAKE_STATE = "wake_detection_state"
        const val PREF_UI_STATE = "conversation_ui_state"
        const val PREF_LAST_STATUS = "conversation_last_status"
        const val PREF_LAST_DIAGNOSTIC = "conversation_last_diagnostic"
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
                .putString(PREF_WAKE_STATE, "ARMED")
                .putString(PREF_UI_STATE, "NOT_OPENED")
                .apply()
            ContextCompat.startForegroundService(
                context,
                Intent(context, WakeWordForegroundService::class.java).setAction(ACTION_START),
            )
        }

        fun pause(context: android.content.Context) {
            context.getSharedPreferences(PREFS, android.content.Context.MODE_PRIVATE).edit().putBoolean(PREF_ENABLED, false).apply()
            context.startService(Intent(context, WakeWordForegroundService::class.java).setAction(ACTION_PAUSE))
        }

        fun stop(context: android.content.Context) {
            context.getSharedPreferences(PREFS, android.content.Context.MODE_PRIVATE).edit()
                .putBoolean(PREF_ENABLED, false).putBoolean(PREF_ARMED, false).apply()
            context.stopService(Intent(context, WakeWordForegroundService::class.java).setAction(ACTION_STOP))
            context.getSystemService(NotificationManager::class.java).cancel(DETECTED_NOTIFICATION_ID)
        }
    }

    private val running = AtomicBoolean(false)
    @Volatile private var voiceController: VoiceConversationController? = null
    @Volatile private var worker: Thread? = null
    @Volatile private var audioRecord: AudioRecord? = null
    @Volatile private var spotter: KeywordSpotter? = null
    @Volatile private var stream: OnlineStream? = null
    @Volatile private var wakeLock: PowerManager.WakeLock? = null
    @Volatile private var pauseRequested = false

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP_CONVERSATION) {
            getSharedPreferences(PREFS, MODE_PRIVATE).edit()
                .putString(PREF_CONVERSATION_STATE, "STOP_REQUESTED")
                .putString(PREF_LAST_STATUS, "مکالمهٔ صوتی از رابط کاربری متوقف شد.")
                .apply()
            voiceController?.close()
            voiceController = null
            // This service owns the foreground notification and wake lock too. Do not leave
            // an idle microphone service alive after the conversation controller has stopped.
            running.set(false)
            runCatching { audioRecord?.stop() }
            runCatching { wakeLock?.takeIf { it.isHeld }?.release() }
            wakeLock = null
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelfResult(startId)
            return START_NOT_STICKY
        }
        if (intent?.action == ACTION_STOP) {
            running.set(false)
            voiceController?.close()
            voiceController = null
            stopSelfResult(startId)
            return START_NOT_STICKY
        }
        if (intent?.action == ACTION_PAUSE) {
            if (worker == null) {
                voiceController?.close()
                voiceController = null
                stopSelfResult(startId)
                return START_NOT_STICKY
            }
            getSharedPreferences(PREFS, MODE_PRIVATE).edit().putBoolean(PREF_ENABLED, false).apply()
            pauseRequested = true
            running.set(false)
            runCatching { audioRecord?.stop() }
            return START_NOT_STICKY
        }
        pauseRequested = false
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
            prefs.edit()
                .putString(PREF_CONVERSATION_STATE, "LISTENER_START_FAILED")
                .putString(PREF_WAKE_STATE, "ERROR")
                .putString(PREF_LAST_STATUS, "VOICE-KWS-018: شروع سرویس میکروفون ناموفق بود: ${t.message ?: t.javaClass.simpleName}")
                .putString(PREF_LAST_DIAGNOSTIC, wakeDiagnostic("FOREGROUND_SERVICE_START_FAILED", t.stackTraceToString()))
                .apply()
            Log.e(TAG, "Could not start microphone foreground service", t)
            stopSelfResult(startId)
            return START_NOT_STICKY
        }
        if (running.compareAndSet(false, true)) {
            if (wakeLock?.isHeld != true) {
                runCatching {
                    wakeLock = getSystemService(PowerManager::class.java)
                        .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "$packageName:DinaWakeWord").apply {
                            setReferenceCounted(false)
                            acquire()
                        }
                }.onFailure { Log.w(TAG, "Could not acquire partial wake lock; screen-off listening may be interrupted", it) }
            }
            worker = Thread({ listen(modelPath) }, "DinaWakeWord-KWS").apply { start() }
        }
        return START_NOT_STICKY
    }

    private fun listen(modelPath: String) {
        var detected = false
        var detectedKeyword = ""
        var audioChunksProcessed = 0L
        var rejectedKeywordCandidates = 0
        try {
            val modelStore = WakeWordModelStore(contentResolver, File(filesDir, "wake-word-models")) {
                assets.open("wake-word/zh-en-3m/tokens.txt")
            }
            val entry = modelStore.list().firstOrNull { it.directory.absolutePath == modelPath }
                ?: error("VOICE-KWS-001: مدل بیدارباش انتخاب‌شده پیدا نشد.")
            // Migrate already-imported copies of the official zh-en package too. Updating the
            // importer alone leaves older on-device keywords.txt files unchanged.
            val isKnownZhEnPhoneModel = entry.directory.listFiles().orEmpty().any {
                it.isFile && it.name.equals("en.phone", ignoreCase = true)
            } && entry.encoder.name.contains("epoch-13-avg-2", ignoreCase = true)
            if (isKnownZhEnPhoneModel) {
                val correctedKeywords = "D IY1 N AA1 @دینا\nD IY1 N AH0 @دینا\nD IY0 N AA1 @دینا\nD IH1 N AA1 @دینا\nD IH1 N AH0 @دینا\n"
                if (runCatching { entry.keywords.readText(Charsets.UTF_8) }.getOrNull() != correctedKeywords) {
                    entry.keywords.writeText(correctedKeywords, Charsets.UTF_8)
                    Log.i(TAG, "Migrated existing zh-en wake keywords to canonical phone tokens")
                }
            }
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
                // This model is not Persian-trained; bias the configured phonetic approximation
                // toward recall and keep multiple pronunciation paths alive.
                keywordsScore = 3.0f
                keywordsThreshold = 0.08f
                maxActivePaths = 4
                numTrailingBlanks = 2
            }
            spotter = KeywordSpotter(config = config)
            stream = spotter!!.createStream()
            check(ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
                "VOICE-KWS-001: مجوز میکروفون داده نشده یا لغو شده است."
            }
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
            getSharedPreferences(PREFS, MODE_PRIVATE).edit()
                .putString(PREF_CONVERSATION_STATE, "WAKE_WORD_LISTENING")
                .putString(PREF_WAKE_STATE, "LISTENING")
                .putString(PREF_LAST_STATUS, "شنوندهٔ محلی دینا فعال است.")
                .putString(PREF_LAST_DIAGNOSTIC, wakeDiagnostic("LISTENER_STARTED", "model=${entry.directory.name}; audioRecordState=${recorder.state}; wakeLockHeld=${wakeLock?.isHeld == true}"))
                .apply()
            val pcm = ShortArray(SAMPLES_PER_CHUNK)
            while (running.get()) {
                val count = recorder.read(pcm, 0, pcm.size)
                if (count == AudioRecord.ERROR_DEAD_OBJECT || count == AudioRecord.ERROR_INVALID_OPERATION) {
                    error("VOICE-KWS-019: ضبط صوت از کار افتاد؛ AudioRecord.read=$count")
                }
                if (count < 0) {
                    if (count == AudioRecord.ERROR_BAD_VALUE || count == AudioRecord.ERROR) {
                        error("VOICE-KWS-019: ضبط صوت خطا داد؛ AudioRecord.read=$count")
                    }
                    continue
                }
                if (count == 0) continue
                audioChunksProcessed++
                val samples = FloatArray(count) { pcm[it] / 32768.0f }
                val activeStream = stream ?: break
                val activeSpotter = spotter ?: break
                activeStream.acceptWaveform(samples, SAMPLE_RATE)
                while (activeSpotter.isReady(activeStream)) {
                    activeSpotter.decode(activeStream)
                    val result = activeSpotter.getResult(activeStream)
                    if (result.keyword.isNotBlank()) {
                        if (isDinaWakeKeywordLabel(result.keyword)) {
                            detectedKeyword = result.keyword
                            Log.i(
                                TAG,
                                "Wake phrase accepted: keyword=${result.keyword}; chunks=$audioChunksProcessed; tokens=${result.tokens.joinToString(" ")}",
                            )
                            detected = true
                            running.set(false)
                            break
                        }
                        rejectedKeywordCandidates++
                        Log.w(TAG, "Ignoring non-Dina KWS result: keyword=${result.keyword}; rejected=$rejectedKeywordCandidates")
                    }
                }
                if (detected) break
            }
        } catch (t: Throwable) {
            Log.e(TAG, "Wake-word listener stopped", t)
            val prefs = getSharedPreferences(PREFS, MODE_PRIVATE)
            prefs.edit()
                .putString(PREF_CONVERSATION_STATE, if (pauseRequested) "LISTENER_PAUSED" else "LISTENER_ERROR")
                .putString(PREF_WAKE_STATE, if (pauseRequested) "PAUSED" else "ERROR")
                .putString(PREF_LAST_STATUS, "VOICE-KWS-020: شنونده متوقف شد: ${t.message ?: t.javaClass.simpleName}")
                .putString(PREF_LAST_DIAGNOSTIC, wakeDiagnostic(if (pauseRequested) "LISTENER_PAUSED" else "LISTENER_EXCEPTION", t.stackTraceToString()))
                .apply()
            if (!pauseRequested && running.get()) {
                prefs.edit().putBoolean(PREF_ENABLED, false).putBoolean(PREF_ARMED, false).apply()
                postWakeNotification("شنود بیدارباش متوقف شد؛ برای جزئیات، گزارش برنامه را بررسی کن.", openVoice = false)
            }
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
                val prefs = getSharedPreferences(PREFS, MODE_PRIVATE)
                prefs.edit()
                    .putBoolean(PREF_ENABLED, false)
                    .putString(PREF_WAKE_STATE, "DETECTED")
                    .putString(PREF_CONVERSATION_STATE, "WAKE_WORD_DETECTED")
                    .putString(PREF_LAST_STATUS, "عبارت «دینا» تشخیص داده شد؛ صدای تأیید پخش می‌شود و میکروفون مکالمه فعال خواهد شد.")
                    .putString(PREF_LAST_DIAGNOSTIC, wakeDiagnostic("DETECTED", "keyword=$detectedKeyword; audioChunks=$audioChunksProcessed; rejectedCandidates=$rejectedKeywordCandidates; Activity launch is optional"))
                    .apply()
                playWakeDetectedCue()
                // Audio starts in the already-running microphone foreground service. Opening UI
                // is a best-effort enhancement, not a prerequisite for a voice conversation.
                val conversationStarted = startVoiceConversationInService()
                postWakeNotification(
                    if (conversationStarted) "عبارت «دینا» شنیده شد؛ مکالمهٔ صوتی در پس‌زمینه شروع می‌شود."
                    else "عبارت «دینا» شنیده شد؛ مدل‌های STT و TTS را بررسی کن.",
                    openVoice = true,
                )
                runCatching {
                    startActivity(Intent(this, MainActivity::class.java).apply {
                        flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP
                        putExtra(EXTRA_OPEN_VOICE_FROM_WAKE, true)
                        putExtra(EXTRA_CONVERSATION_STARTED_IN_SERVICE, conversationStarted)
                    })
                    prefs.edit()
                        .putString(PREF_UI_STATE, if (conversationStarted) "ACTIVITY_LAUNCH_ATTEMPTED_AUDIO_RUNNING" else "ACTIVITY_LAUNCH_ATTEMPTED_NO_MODELS")
                        .putString(PREF_LAST_DIAGNOSTIC, wakeDiagnostic("ACTIVITY_LAUNCH_ATTEMPTED", "conversationStartedInService=$conversationStarted"))
                        .apply()
                }.onFailure {
                    Log.w(TAG, "Automatic activity launch blocked; audio conversation remains service-owned", it)
                    prefs.edit()
                        .putString(PREF_UI_STATE, if (conversationStarted) "AUDIO_RUNNING_UI_UNAVAILABLE" else "UI_UNAVAILABLE_NO_MODELS")
                        .putString(PREF_LAST_DIAGNOSTIC, wakeDiagnostic("ACTIVITY_LAUNCH_EXCEPTION", it.javaClass.simpleName + ": " + (it.message ?: "")))
                        .apply()
                }
            }
            val prefs = getSharedPreferences(PREFS, MODE_PRIVATE)
            val keepForegroundForVoice = shouldKeepWakeServiceForConversation(
                armed = prefs.getBoolean(PREF_ARMED, false),
                listeningEnabled = prefs.getBoolean(PREF_ENABLED, false),
                phraseDetected = detected,
                controllerActive = voiceController != null,
            )
            pauseRequested = false
            if (keepForegroundForVoice) {
                val canPostNotification = Build.VERSION.SDK_INT < 33 ||
                    ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) ==
                    PackageManager.PERMISSION_GRANTED
                if (canPostNotification) {
                    notifyIfPermitted(LISTENING_NOTIFICATION_ID, buildConversationNotification())
                } else {
                    // The foreground service keeps running; Android may hide regular notifications
                    // until the user grants notification permission.
                    Log.w(TAG, "POST_NOTIFICATIONS denied; keeping wake-word service active without updating its regular notification")
                }
            } else {
                runCatching { wakeLock?.takeIf { it.isHeld }?.release() }
                wakeLock = null
                stopForeground(STOP_FOREGROUND_REMOVE)
                stopSelf()
            }
            // Clear only our own worker reference; a newer worker may already have replaced it.
            if (worker === Thread.currentThread()) worker = null
        }
    }

    /** Audible confirmation independent of whether the app UI is visible. */
    private fun playWakeDetectedCue() {
        val tone = runCatching { ToneGenerator(android.media.AudioManager.STREAM_NOTIFICATION, 85) }.getOrNull()
        if (tone == null) {
            Log.w(TAG, "Could not create wake-detection confirmation tone")
            return
        }
        try {
            tone.startTone(ToneGenerator.TONE_PROP_ACK, 180)
            // Let the cue finish before starting conversation capture, to avoid feeding the beep to STT.
            Thread.sleep(240)
        } catch (t: InterruptedException) {
            Thread.currentThread().interrupt()
        } finally {
            tone.release()
        }
    }

    private fun startVoiceConversationInService(): Boolean {
        if (voiceController != null) return true
        val prefs = getSharedPreferences(PREFS, MODE_PRIVATE)
        return try {
            val store = VoiceModelStore(contentResolver, File(filesDir, "voice-models"))
            val sttModels = store.list(VoiceModelKind.STT)
            val ttsModels = store.list(VoiceModelKind.TTS)
            val sttId = prefs.getString(PREF_STT_MODEL_ID, null)
            val ttsId = prefs.getString(PREF_TTS_MODEL_ID, null)
            val stt = sttModels.firstOrNull { it.id == sttId }
                ?: sttModels.firstOrNull { it.id.startsWith("stt-shenava-koochik-") }
                ?: sttModels.firstOrNull()
            val tts = ttsModels.firstOrNull { it.id == ttsId } ?: ttsModels.firstOrNull()
            if (stt == null || tts == null) {
                val missing = buildList {
                    if (stt == null) add("STT")
                    if (tts == null) add("TTS")
                }.joinToString("+")
                prefs.edit()
                    .putString(PREF_CONVERSATION_STATE, "DETECTED_MODELS_MISSING")
                    .putString(PREF_LAST_STATUS, "VOICE-KWS-014: مدل‌های لازم برای مکالمه موجود نیستند: $missing")
                    .putString(PREF_LAST_DIAGNOSTIC, wakeDiagnostic("MODELS_MISSING", "stt=${stt != null} tts=${tts != null}"))
                    .apply()
                return false
            }
            prefs.edit()
                .putString(PREF_STT_MODEL_ID, stt.id)
                .putString(PREF_TTS_MODEL_ID, tts.id)
                .putString(PREF_CONVERSATION_STATE, "CONVERSATION_START_REQUESTED")
                .putString(PREF_LAST_STATUS, "دینا تشخیص داده شد؛ موتورهای مکالمه در حال آماده‌سازی هستند.")
                .apply()
            val appContainer = (application as? AIChatApplication)?.container
                ?: error("VOICE-KWS-016: مخزن برنامه در دسترس نیست.")
            val controller = VoiceConversationController(
                context = this,
                container = appContainer,
                onStatus = { status ->
                    val state = when {
                        status.contains("VOICE-START-001") || status.contains("VOICE-CHAT-003") -> "CONVERSATION_ERROR"
                        status.contains("گوش‌دادن") -> "LISTENING_SCREEN_OFF_CAPABLE"
                        status.contains("در حال پخش") || status.contains("در حال پردازش") -> "CONVERSATION_ACTIVE"
                        else -> "CONVERSATION_STARTING"
                    }
                    prefs.edit().putString(PREF_LAST_STATUS, status).putString(PREF_CONVERSATION_STATE, state).apply()
                    notifyIfPermitted(LISTENING_NOTIFICATION_ID, buildConversationNotification(status))
                },
                onDiagnostic = { detail ->
                    prefs.edit().putString(PREF_LAST_DIAGNOSTIC, wakeDiagnostic("VOICE_RUNTIME", detail)).apply()
                    Log.i(TAG, detail)
                },
                onListening = { listening ->
                    prefs.edit().putString(PREF_CONVERSATION_STATE, if (listening) "LISTENING_SCREEN_OFF_CAPABLE" else "CONVERSATION_NOT_LISTENING").apply()
                },
                onLine = { speaker, line ->
                    prefs.edit()
                        .putString("last_voice_speaker", speaker)
                        .putString("last_voice_line", line.take(2000))
                        .putLong("last_voice_line_at", System.currentTimeMillis())
                        .apply()
                },
            )
            voiceController = controller
            controller.start(stt, tts)
            prefs.edit()
                .putString(PREF_CONVERSATION_STATE, "CONVERSATION_START_REQUESTED")
                .putString(PREF_LAST_DIAGNOSTIC, wakeDiagnostic("CONVERSATION_START_REQUESTED", "screenInteractive=${getSystemService(PowerManager::class.java)?.isInteractive}; stt=${stt.id}; tts=${tts.id}"))
                .apply()
            true
        } catch (t: Throwable) {
            prefs.edit()
                .putString(PREF_CONVERSATION_STATE, "CONVERSATION_START_FAILED")
                .putString(PREF_LAST_STATUS, "VOICE-KWS-017: شروع مکالمه ناموفق بود: ${t.message ?: t.javaClass.simpleName}")
                .putString(PREF_LAST_DIAGNOSTIC, wakeDiagnostic("CONVERSATION_START_EXCEPTION", t.stackTraceToString()))
                .apply()
            runCatching { voiceController?.close() }
            voiceController = null
            Log.e(TAG, "Could not start service-owned voice conversation", t)
            false
        }
    }

    private fun wakeDiagnostic(event: String, detail: String): String =
        "DINA WAKE DIAGNOSTICS\n" +
            "event=$event\n" +
            "screenInteractive=${getSystemService(PowerManager::class.java)?.isInteractive}\n" +
            "listenerRunning=${running.get()}\n" +
            "wakeState=${getSharedPreferences(PREFS, MODE_PRIVATE).getString(PREF_WAKE_STATE, "UNKNOWN")}\n" +
            "conversationState=${getSharedPreferences(PREFS, MODE_PRIVATE).getString(PREF_CONVERSATION_STATE, "UNKNOWN")}\n" +
            "uiState=${getSharedPreferences(PREFS, MODE_PRIVATE).getString(PREF_UI_STATE, "UNKNOWN")}\n" +
            "voiceControllerActive=${voiceController != null}\n" +
            "detail=$detail"

    private fun postWakeNotification(message: String, openVoice: Boolean) {
        val intent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
            putExtra(EXTRA_OPEN_VOICE_FROM_WAKE, openVoice)
            putExtra(EXTRA_CONVERSATION_STARTED_IN_SERVICE, voiceController != null)
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
        notifyIfPermitted(DETECTED_NOTIFICATION_ID, notification)
    }

    /**
     * Android 13+ requires runtime notification permission for ordinary notification updates.
     * Keep the microphone foreground service alive even when the user has denied that permission.
     */
    private fun notifyIfPermitted(id: Int, notification: Notification) {
        if (Build.VERSION.SDK_INT >= 33 &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) !=
            PackageManager.PERMISSION_GRANTED
        ) {
            Log.w(TAG, "POST_NOTIFICATIONS denied; skipping notification id=$id")
            return
        }
        NotificationManagerCompat.from(this).notify(id, notification)
    }

    private fun buildConversationNotification(status: String? = null): Notification =
        NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(applicationInfo.icon.takeIf { it != 0 } ?: android.R.drawable.ic_btn_speak_now)
            .setContentTitle("مکالمهٔ صوتی دینا فعال است")
            .setContentText(status?.take(180) ?: "شنود و پردازش گفتار در سرویس پیش‌زمینه ادامه دارد؛ برای توقف از صفحه مکالمه استفاده کن.")
            .setOngoing(true)
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()

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
        runCatching { wakeLock?.takeIf { it.isHeld }?.release() }
        wakeLock = null
        worker?.interrupt()
        voiceController?.close()
        voiceController = null
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

}
