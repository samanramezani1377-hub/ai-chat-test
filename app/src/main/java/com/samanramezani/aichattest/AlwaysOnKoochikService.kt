package com.samanramezani.aichattest

import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.content.ContextCompat
import com.samanramezani.aichattest.ui.voice.VoiceConversationController
import com.samanramezani.aichattest.voice.LocalMicrophone
import com.samanramezani.aichattest.voice.LocalNemoCtcAsr
import com.samanramezani.aichattest.voice.LayaDecisionEngine
import com.samanramezani.aichattest.voice.VoiceModelKind
import com.samanramezani.aichattest.voice.VoiceModelStore
import com.samanramezani.aichattest.voice.extractDinaCommand
import java.io.File
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Wake-word control phrase: stop the current dialogue without disabling always-on wake listening.
 * The command is already extracted after «دینا», so match only the remaining short phrase.
 */
internal fun isDinaSleepCommand(rawCommand: String): Boolean {
    val normalized = rawCommand
        .lowercase(java.util.Locale.ROOT)
        .replace('ي', 'ی')
        .replace('ك', 'ک')
        .replace('‌', ' ')
        .trim()
        .replace(Regex("""[.!؟?،,؛;]+$"""), "")
        .replace(Regex("""\s+"""), " ")
    return normalized in setOf("خاموش شو", "خاموش", "برو بخواب", "فعلا خاموش شو")
}

/**
 * Always-on local Persian speech listener. Shenava Koochik ASR recognizes each bounded utterance;
 * only the text after "دینا" is handed to the assistant. Utterance audio/transcripts are discarded
 * when the utterance ends if the wake phrase was not present; no transcript history is persisted.
 */
class AlwaysOnKoochikService : Service() {
    companion object {
        private const val TAG = "AlwaysOnKoochik"
        private const val CHANNEL_ID = "dina_always_on"
        private const val NOTIFICATION_ID = 4211

        const val ACTION_START = "com.samanramezani.aichattest.action.START_DINA_ALWAYS_ON"
        const val ACTION_STOP = "com.samanramezani.aichattest.action.STOP_DINA_ALWAYS_ON"
        const val ACTION_STOP_CONVERSATION = "com.samanramezani.aichattest.action.STOP_DINA_CONVERSATION"
        const val EXTRA_OPEN_VOICE_FROM_ALWAYS_ON = "open_voice_from_always_on"
        const val EXTRA_CONVERSATION_STARTED_IN_SERVICE = "dina_conversation_started_in_service"

        const val PREFS = "dina_always_on"
        const val PREF_STT_MODEL_ID = "voice_stt_model_id"
        const val PREF_TTS_MODEL_ID = "voice_tts_model_id"
        const val PREF_CONVERSATION_STATE = "conversation_state"
        const val PREF_WAKE_STATE = "wake_detection_state"
        const val PREF_LAST_STATUS = "conversation_last_status"
        const val PREF_LAST_DIAGNOSTIC = "conversation_last_diagnostic"
        const val PREF_MIC_LISTENING = "conversation_mic_listening"
        const val PREF_ENABLED = "enabled"

        fun start(context: Context) {
            context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
                .putBoolean(PREF_ENABLED, true)
                .putString(PREF_WAKE_STATE, "START_REQUESTED")
                .apply()
            ContextCompat.startForegroundService(
                context,
                Intent(context, AlwaysOnKoochikService::class.java).setAction(ACTION_START),
            )
        }

        fun stop(context: Context) {
            context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
                .putBoolean(PREF_ENABLED, false)
                .putString(PREF_WAKE_STATE, "OFF")
                .apply()
            context.startService(Intent(context, AlwaysOnKoochikService::class.java).setAction(ACTION_STOP))
        }
    }

    private val running = AtomicBoolean(false)
    @Volatile private var microphone: LocalMicrophone? = null
    @Volatile private var worker: Thread? = null
    @Volatile private var voiceController: VoiceConversationController? = null
    @Volatile private var transitioningToConversation = false
    @Volatile private var wakeLock: PowerManager.WakeLock? = null
    // Transfer the already-loaded wake recognizer into the conversation controller instead of
    // loading the same large Shenava model a second time immediately after wake detection.
    @Volatile private var pendingConversationRecognizer: LocalNemoCtcAsr? = null

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_STOP -> {
                getSharedPreferences(PREFS, MODE_PRIVATE).edit().putBoolean(PREF_ENABLED, false).apply()
                stopEverything()
                stopSelfResult(startId)
                return START_NOT_STICKY
            }
            ACTION_STOP_CONVERSATION -> {
                runCatching { voiceController?.close() }
                voiceController = null
                transitioningToConversation = false
                if (getSharedPreferences(PREFS, MODE_PRIVATE).getBoolean(PREF_ENABLED, false)) {
                    updateStatus("مکالمه متوقف شد؛ شنود فارسی «دینا» دوباره فعال می‌شود.", "ALWAYS_ON_REARMING")
                    startListenerIfNeeded()
                } else {
                    stopEverything()
                    stopSelfResult(startId)
                    return START_NOT_STICKY
                }
                return START_STICKY
            }
        }

        val prefs = getSharedPreferences(PREFS, MODE_PRIVATE)
        if (!prefs.getBoolean(PREF_ENABLED, false)) {
            stopEverything()
            stopSelfResult(startId)
            return START_NOT_STICKY
        }
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                startForeground(
                    NOTIFICATION_ID,
                    buildNotification("در حال آماده‌سازی مدل فارسی Shenava Koochik…"),
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE,
                )
            } else {
                startForeground(NOTIFICATION_ID, buildNotification("در حال آماده‌سازی مدل فارسی Shenava Koochik…"))
            }
        } catch (t: Throwable) {
            updateStatus("VOICE-ALWAYS-001: شروع سرویس میکروفون ناموفق بود: ${t.message ?: t.javaClass.simpleName}", "ERROR")
            prefs.edit().putBoolean(PREF_ENABLED, false).apply()
            stopSelfResult(startId)
            return START_NOT_STICKY
        }
        if (wakeLock?.isHeld != true) {
            runCatching {
                wakeLock = getSystemService(PowerManager::class.java)
                    .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "$packageName:DinaAlwaysOn").apply {
                        setReferenceCounted(false)
                        acquire()
                    }
            }.onFailure { Log.w(TAG, "Could not acquire partial wake lock", it) }
        }
        startListenerIfNeeded()
        return START_STICKY
    }

    private fun startListenerIfNeeded() {
        if (!getSharedPreferences(PREFS, MODE_PRIVATE).getBoolean(PREF_ENABLED, false)) return
        if (voiceController != null || transitioningToConversation) return
        if (!running.compareAndSet(false, true)) return
        worker = Thread({ listenForDina() }, "Dina-Shenava-Koochik").apply {
            isDaemon = true
            start()
        }
    }

    private fun listenForDina() {
        var recognizer: LocalNemoCtcAsr? = null
        var mic: LocalMicrophone? = null
        var detectedCommand: String? = null
        var lastProbeCommand: String? = null
        var wakeSeenInRollingWindow = false
        var speechSinceProbe = 0
        var failure: Throwable? = null
        val rollingWindow = FloatArray(16000 * 3)
        var rollingWrite = 0
        var rollingCount = 0
        fun appendRolling(samples: FloatArray) {
            for (sample in samples) {
                rollingWindow[rollingWrite] = sample
                rollingWrite = (rollingWrite + 1) % rollingWindow.size
                if (rollingCount < rollingWindow.size) rollingCount++
            }
        }
        fun rollingSnapshot(): FloatArray {
            val result = FloatArray(rollingCount)
            val start = (rollingWrite - rollingCount + rollingWindow.size) % rollingWindow.size
            for (i in result.indices) result[i] = rollingWindow[(start + i) % rollingWindow.size]
            return result
        }
        try {
            check(ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
                "VOICE-ALWAYS-002: مجوز میکروفون داده نشده یا لغو شده است."
            }
            val store = VoiceModelStore(contentResolver, File(filesDir, "voice-models"))
            val sttModels = store.list(VoiceModelKind.STT)
            val prefs = getSharedPreferences(PREFS, MODE_PRIVATE)
            val selectedId = prefs.getString(PREF_STT_MODEL_ID, null)
            val stt = sttModels.firstOrNull {
                it.id == selectedId && it.id.startsWith("stt-shenava-koochik-") && it.asrMode == "nemo-ctc"
            } ?: sttModels.firstOrNull {
                it.id.startsWith("stt-shenava-koochik-") && it.asrMode == "nemo-ctc"
            } ?: error("VOICE-ALWAYS-003: مدل فارسی Shenava Koochik آماده نیست؛ model.onnx و tokens.txt را وارد کنید.")
            // Do not construct a separate ONNX/Piper engine for the wake acknowledgement.
            // The conversation controller speaks the acknowledgement through the exact TTS
            // engine selected in settings (Piper or Pocket TTS), avoiding a hidden fallback.
            val activeRecognizer = pendingConversationRecognizer ?: LocalNemoCtcAsr(stt)
            pendingConversationRecognizer = null
            recognizer = activeRecognizer
            val frames = ArrayBlockingQueue<LocalMicrophone.Frame>(256)
            val activeMic = LocalMicrophone()
            mic = activeMic
            microphone = activeMic
            activeMic.start { frame ->
                if (!frames.offer(frame)) {
                    frames.poll()
                    frames.offer(frame)
                }
            }
            updateStatus(
                "شنود دائمی با مدل فارسی Shenava Koochik فعال است؛ متن مکالمه ذخیره نمی‌شود.",
                "ALWAYS_ON_LISTENING",
                "model=${stt.id}; mode=bounded-utterance-plus-3s-rolling-probe; queueCapacity=256; persistentTranscript=false",
            )
            notifyStatus("در حال شنیدن «دینا» با مدل فارسی محلی…")
            while (running.get() && getSharedPreferences(PREFS, MODE_PRIVATE).getBoolean(PREF_ENABLED, false)) {
                val frame = frames.poll(300, TimeUnit.MILLISECONDS) ?: continue
                appendRolling(frame.samples)
                if (frame.speech) speechSinceProbe += frame.samples.size
                val update = activeRecognizer.accept(frame.samples, 16000, frame.speech)

                // Shenava Koochik is an offline CTC recognizer, not a streaming partial decoder.
                // Probe only the last 3 seconds once per ~1 second of speech; audio history stays
                // bounded, and the current utterance is still accumulated by the same recognizer.
                if (!wakeSeenInRollingWindow && frame.speech &&
                    rollingCount >= 24000 && speechSinceProbe >= 16000
                ) {
                    speechSinceProbe = 0
                    val probe = activeRecognizer.recognizeWindow(rollingSnapshot())
                    val commandCandidate = extractDinaCommand(probe)
                    if (commandCandidate != null) {
                        wakeSeenInRollingWindow = true
                        lastProbeCommand = commandCandidate
                        updateStatus(
                            "«دینا» در گفتار تشخیص داده شد؛ ادامهٔ همان جمله در بافر محدود ثبت می‌شود.",
                            "DINA_DETECTED_RECORDING",
                            "probeWindowMs=${rollingCount * 1000 / 16000}; probeChars=${probe.length}; persistentTranscript=false",
                        )
                        // Acknowledge only after the complete utterance is decoded and handed to the dialogue controller.
                    }
                }

                if (!update.endpoint) continue
                val transcript = update.text
                val command = extractDinaCommand(transcript)
                Log.d(TAG, "ASR endpoint: chars=${transcript.length}; wakeFound=${command != null || wakeSeenInRollingWindow}")
                // Prefer the full utterance decoded at endpoint; use the last rolling result only
                // if endpoint decoding omitted a phrase already detected in the rolling window.
                val finalCommand = command ?: if (wakeSeenInRollingWindow) lastProbeCommand else null
                if (finalCommand != null) {
                    if (isDinaSleepCommand(finalCommand)) {
                        // This is a conversation-level stop command, not a request to disable
                        // always-on mode. Keep the recognizer loaded while re-arming the listener.
                        pendingConversationRecognizer = activeRecognizer
                        detectedCommand = null
                        transitioningToConversation = false
                        running.set(false)
                        updateStatus(
                            "دینا مکالمه را متوقف کرد؛ شنود کلمهٔ بیدارباش همچنان روشن می‌ماند.",
                            "ALWAYS_ON_REARMING",
                            "wakeCommand=STOP_CONVERSATION; enabled=true; transcriptPersisted=false",
                        )
                        break
                    }
                    detectedCommand = finalCommand
                    transitioningToConversation = true
                    running.set(false)
                    // Transfer this loaded recognizer; listenForDina's finally block must not release it.
                    pendingConversationRecognizer = activeRecognizer
                    updateStatus(
                        "«دینا» تأیید شد؛ مدل شنوا بدون بارگذاری مجدد به مکالمه منتقل می‌شود.",
                        "DINA_DETECTED",
                        "utteranceChars=${transcript.length}; commandChars=${finalCommand.length}; sttReused=true; transcriptPersisted=false",
                    )
                    break
                }
                // Drop each completed non-trigger utterance and its rolling audio immediately.
                java.util.Arrays.fill(rollingWindow, 0f)
                rollingWrite = 0
                rollingCount = 0
                speechSinceProbe = 0
                lastProbeCommand = null
            }
        } catch (t: Throwable) {
            if (running.get()) failure = t
            Log.e(TAG, "Always-on Shenava listener stopped", t)
        } finally {
            running.set(false)
            runCatching { mic?.stop() }
            microphone = null
            // A wake-to-conversation handoff transfers ownership to VoiceConversationController.
            if (recognizer !== pendingConversationRecognizer) runCatching { recognizer?.close() }
            if (worker === Thread.currentThread()) worker = null
        }

        if (detectedCommand == null &&
            getSharedPreferences(PREFS, MODE_PRIVATE).getBoolean(PREF_ENABLED, false) &&
            voiceController == null && !transitioningToConversation
        ) {
            // The special «دینا خاموش شو» command stops the current dialogue only; keep the
            // service preference enabled and return to the existing wake-listening loop.
            startListenerIfNeeded()
        }

        if (detectedCommand != null) {
            val command = detectedCommand.orEmpty()
            // Hand the wake acknowledgement to the same controller/engine that speaks the
            // answer. Its speech queue places "متوجه شدم" before the response, so the configured
            // TTS selection is honored and the two utterances cannot overlap.
            val started = startVoiceConversationInService(
                command,
                acknowledgeWake = true,
                speechStartGate = null,
            )
            transitioningToConversation = false
            if (started) {
                notifyStatus("«دینا» شنیده شد؛ جمله به مدل گفت‌وگو ارسال شد و سپس «متوجه شدم» با TTS محلی پخش می‌شود.")
                runCatching {
                    startActivity(Intent(this, MainActivity::class.java).apply {
                        flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP
                        putExtra(EXTRA_OPEN_VOICE_FROM_ALWAYS_ON, true)
                        putExtra(EXTRA_CONVERSATION_STARTED_IN_SERVICE, true)
                    })
                }.onFailure { Log.i(TAG, "UI launch not allowed; background voice session remains active", it) }
            } else {
                getSharedPreferences(PREFS, MODE_PRIVATE).edit()
                    .putString(PREF_LAST_STATUS, "مدل مکالمه آماده نشد؛ شنود دوباره شروع می‌شود.")
                    .apply()
                startListenerIfNeeded()
            }
        } else if (failure != null) {
            val prefs = getSharedPreferences(PREFS, MODE_PRIVATE)
            updateStatus(
                "VOICE-ALWAYS-004: شنود فارسی متوقف شد: ${failure.message ?: failure.javaClass.simpleName}",
                "ERROR",
                failure.stackTraceToString(),
            )
            notifyStatus("شنود متوقف شد؛ تنظیمات مدل‌های محلی و مجوز میکروفون را بررسی کن.")
            // Fail closed rather than creating a tight restart loop for a missing/invalid model.
            prefs.edit().putBoolean(PREF_ENABLED, false).apply()
            stopForeground(STOP_FOREGROUND_REMOVE)
            runCatching { wakeLock?.takeIf { it.isHeld }?.release() }
            wakeLock = null
            stopSelf()
        } else {
            runCatching { acknowledgementTts?.close() }
            acknowledgementTts = null
        }
        if (detectedCommand == null && getSharedPreferences(PREFS, MODE_PRIVATE).getBoolean(PREF_ENABLED, false) &&
            voiceController == null && !transitioningToConversation
        ) {
            // Listener can stop only when the user disables the mode or a detected command is handed off.
            startListenerIfNeeded()
        }
    }

    private fun startVoiceConversationInService(
        initialUtterance: String,
        acknowledgeWake: Boolean,
        speechStartGate: CompletableDeferred<Unit>? = null,
    ): Boolean {
        val prefs = getSharedPreferences(PREFS, MODE_PRIVATE)
        if (voiceController != null) return true
        return try {
            val store = VoiceModelStore(contentResolver, File(filesDir, "voice-models"))
            val stt = store.list(VoiceModelKind.STT).firstOrNull {
                it.id.startsWith("stt-shenava-koochik-") && it.asrMode == "nemo-ctc"
            } ?: error("VOICE-ALWAYS-003: مدل Shenava Koochik پیدا نشد.")
            val ttsModels = store.list(VoiceModelKind.TTS)
            val selectedTtsId = prefs.getString(PREF_TTS_MODEL_ID, null)
            val tts = ttsModels.firstOrNull { it.id == selectedTtsId } ?: ttsModels.firstOrNull()
            val usePocketTts = prefs.getBoolean("pocket_tts_selected", false)
            if (usePocketTts) {
                check(com.samanramezani.aichattest.voice.PocketTtsFarsiBundleStore(
                    contentResolver, File(filesDir, "voice-models/pocket-tts-farsi-v2"),
                ).isReady()) { "POCKET-TTS-SELECT-002: بستهٔ Pocket TTS نصب نیست یا ناقص است." }
            } else {
                check(tts != null) { "VOICE-ALWAYS-005: مدل TTS محلی انتخاب نشده است." }
            }
            val container = (application as? AIChatApplication)?.container
                ?: error("VOICE-ALWAYS-006: مخزن برنامه در دسترس نیست.")
            prefs.edit()
                .putString(PREF_STT_MODEL_ID, stt.id)
                .putString(PREF_TTS_MODEL_ID, tts?.id)
                .putString(PREF_CONVERSATION_STATE, "CONVERSATION_START_REQUESTED")
                .putBoolean(PREF_MIC_LISTENING, false)
                .apply()
            val controller = VoiceConversationController(
                context = this,
                container = container,
                decisionEngine = LayaDecisionEngine(File(filesDir, "voice-models/laya-int4")),
                onStatus = { status ->
                    val state = when {
                        status.contains("VOICE-START-001") || status.contains("VOICE-CHAT-003") ||
                            status.contains("VOICE-TTS-005") || status.contains("VOICE-STT-002") ||
                            status.contains("VOICE-LLM-001") -> "CONVERSATION_ERROR"
                        status.contains("متوقف شد") -> "CONVERSATION_STOPPED"
                        status.contains("گوش‌دادن") -> "CONVERSATION_ACTIVE"
                        status.contains("در حال پخش") || status.contains("در حال پردازش") ||
                            status.contains("در حال شنیدن") || status.contains("پاسخ آماده است") -> "CONVERSATION_ACTIVE"
                        else -> prefs.getString(PREF_CONVERSATION_STATE, "CONVERSATION_STARTING").orEmpty()
                    }
                    prefs.edit().putString(PREF_LAST_STATUS, status).putString(PREF_CONVERSATION_STATE, state).apply()
                    notifyStatus(status)
                    Log.i(TAG, status)
                },
                onDiagnostic = { detail ->
                    prefs.edit().putString(PREF_LAST_DIAGNOSTIC, "ALWAYS_ON_VOICE_RUNTIME: $detail").apply()
                    Log.i(TAG, detail)
                },
                onListening = { active ->
                    prefs.edit()
                        .putBoolean(PREF_MIC_LISTENING, active)
                        .putString(PREF_CONVERSATION_STATE, if (active) "CONVERSATION_ACTIVE" else "CONVERSATION_NOT_LISTENING")
                        .apply()
                    if (active) notifyStatus("مکالمه فعال است؛ مدل محلی آماده دریافت گفتار است.")
                },
                onLine = { speaker, line ->
                    prefs.edit()
                        .putString("last_voice_speaker", speaker)
                        .putString("last_voice_line", line.take(2000))
                        .putLong("last_voice_line_at", System.currentTimeMillis())
                        .apply()
                },
                onWakeShutdown = { loadedAsr ->
                    pendingConversationRecognizer = loadedAsr as? LocalNemoCtcAsr
                    startService(
                        Intent(this, AlwaysOnKoochikService::class.java)
                            .setAction(ACTION_STOP_CONVERSATION),
                    )
                },
            )
            voiceController = controller
            val alreadyLoadedAsr = pendingConversationRecognizer
            pendingConversationRecognizer = null
            controller.start(
                stt,
                tts,
                initialUtterance = initialUtterance,
                acknowledgeWake = acknowledgeWake,
                speechStartGate = speechStartGate,
                preloadedAsr = alreadyLoadedAsr,
            )
            true
        } catch (t: Throwable) {
            Log.e(TAG, "Could not start voice conversation after Dina", t)
            updateStatus("VOICE-ALWAYS-006: شروع مکالمه ناموفق بود: ${t.message ?: t.javaClass.simpleName}", "CONVERSATION_START_FAILED", t.stackTraceToString())
            false
        }
    }

    private fun stopEverything() {
        running.set(false)
        runCatching { microphone?.stop() }
        microphone = null
        runCatching { worker?.interrupt() }
        worker = null
        runCatching { voiceController?.close() }
        voiceController = null
        runCatching { pendingConversationRecognizer?.close() }
        pendingConversationRecognizer = null
        getSharedPreferences(PREFS, MODE_PRIVATE).edit()
            .putBoolean(PREF_MIC_LISTENING, false)
            .apply()
        transitioningToConversation = false
        runCatching { wakeLock?.takeIf { it.isHeld }?.release() }
        wakeLock = null
        stopForeground(STOP_FOREGROUND_REMOVE)
    }

    private fun updateStatus(status: String, state: String, diagnostic: String = "") {
        getSharedPreferences(PREFS, MODE_PRIVATE).edit()
            .putString(PREF_LAST_STATUS, status)
            .putString(PREF_CONVERSATION_STATE, state)
            .putString(PREF_WAKE_STATE, state)
            .apply {
                if (diagnostic.isNotBlank()) putString(PREF_LAST_DIAGNOSTIC, "ALWAYS_ON: $diagnostic")
            }
            .apply()
        Log.i(TAG, status)
        notifyStatus(status)
    }

    private fun notifyStatus(status: String) {
        val permitted = Build.VERSION.SDK_INT < 33 ||
            ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) == PackageManager.PERMISSION_GRANTED
        if (permitted) {
            runCatching { NotificationManagerCompat.from(this).notify(NOTIFICATION_ID, buildNotification(status)) }
        }
    }

    private fun buildNotification(status: String): Notification {
        val openIntent = PendingIntent.getActivity(
            this, 0, Intent(this, MainActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP
                putExtra(EXTRA_OPEN_VOICE_FROM_ALWAYS_ON, true)
                putExtra(EXTRA_CONVERSATION_STARTED_IN_SERVICE, voiceController != null)
            }, PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val stopIntent = PendingIntent.getService(
            this, 1, Intent(this, AlwaysOnKoochikService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setContentTitle("AI Chat Test · شنود فارسی محلی")
            .setContentText(status.take(180))
            .setStyle(NotificationCompat.BigTextStyle().bigText(status))
            .setContentIntent(openIntent)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .addAction(0, "خاموش کردن", stopIntent)
            .build()
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= 26) {
            val manager = getSystemService(NotificationManager::class.java)
            manager.createNotificationChannel(
                NotificationChannel(CHANNEL_ID, "شنود فارسی محلی", NotificationManager.IMPORTANCE_LOW).apply {
                    description = "وضعیت شنود محلی Shenava Koochik برای تشخیص عبارت دینا"
                },
            )
        }
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        stopEverything()
        super.onDestroy()
    }
}
