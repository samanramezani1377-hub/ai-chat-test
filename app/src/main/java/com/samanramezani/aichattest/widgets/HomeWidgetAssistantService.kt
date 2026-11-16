package com.samanramezani.aichattest.widgets

import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.os.SystemClock
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import androidx.core.content.ContextCompat
import com.samanramezani.aichattest.AIChatApplication
import com.samanramezani.aichattest.AlwaysOnKoochikService
import com.samanramezani.aichattest.AppContainer
import com.samanramezani.aichattest.R
import com.samanramezani.aichattest.voice.LocalMicrophone
import com.samanramezani.aichattest.voice.LocalNemoCtcAsr
import com.samanramezani.aichattest.voice.LocalPiperTts
import com.samanramezani.aichattest.voice.VoiceModelKind
import com.samanramezani.aichattest.voice.VoiceModelStore
import com.woogit.aicore.agent.AgentEvent
import com.woogit.aicore.domain.InferenceSettings
import com.woogit.aicore.domain.ModelResult
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

/**
 * Handles home-widget requests without opening MainActivity.
 *
 * Text entry is collected by the compact WidgetComposeActivity; this service performs inference
 * and streams the answer back into RemoteViews. Voice is one bounded utterance per widget tap:
 * Shenava endpoint silence finalizes the turn, the microphone is released, then inference/TTS run.
 */
class HomeWidgetAssistantService : Service() {
    companion object {
        const val ACTION_SEND_TEXT = "com.samanramezani.aichattest.widgets.SEND_TEXT"
        const val ACTION_VOICE_TURN = "com.samanramezani.aichattest.widgets.VOICE_TURN"
        const val EXTRA_TEXT = "widget_text"
        private const val CHANNEL_ID = "home_widget_assistant"
        private const val NOTIFICATION_ID = 7303
        private const val MAX_VOICE_TURN_MS = 25_000L
    }

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val busy = AtomicBoolean(false)
    private val voiceFinished = AtomicBoolean(false)
    @Volatile private var microphone: LocalMicrophone? = null
    @Volatile private var recognizer: LocalNemoCtcAsr? = null
    @Volatile private var tts: LocalPiperTts? = null
    private val container: AppContainer
        get() = (application as AIChatApplication).container

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val action = intent?.action
        if (!busy.compareAndSet(false, true)) {
            if (action == ACTION_VOICE_TURN) {
                HomeWidgetState.updateVoice(this, "درخواست قبلی هنوز در حال اجراست.")
            } else {
                HomeWidgetState.updateChat(this, "در حال پاسخ‌گویی", "یک درخواست دیگر هنوز در حال پردازش است.")
            }
            return START_NOT_STICKY
        }

        try {
            startForegroundFor(action == ACTION_VOICE_TURN)
        } catch (t: Throwable) {
            busy.set(false)
            val message = "سرویس ویجت شروع نشد: ${t.message ?: t.javaClass.simpleName}"
            HomeWidgetState.updateChat(this, "خطا", message)
            HomeWidgetState.updateVoice(this, message)
            stopSelfResult(startId)
            return START_NOT_STICKY
        }

        when (action) {
            ACTION_SEND_TEXT -> {
                val text = intent.getStringExtra(EXTRA_TEXT).orEmpty().trim()
                if (text.isBlank()) {
                    busy.set(false)
                    HomeWidgetState.updateChat(this, "پیام خالی است")
                    stopSelfResult(startId)
                } else {
                    HomeWidgetState.updateChat(this, "در حال آماده‌سازی پاسخ…", "شما: $text\n\nدستیار: در حال پاسخ‌گویی…")
                    scope.launch {
                        try {
                            val reply = generateReply(text)
                            HomeWidgetState.updateChat(this@HomeWidgetAssistantService, "پاسخ آماده است", "شما: $text\n\nدستیار: $reply")
                        } catch (t: Throwable) {
                            HomeWidgetState.updateChat(this@HomeWidgetAssistantService, "خطا", "پاسخ تولید نشد: ${t.message ?: t.javaClass.simpleName}")
                        } finally {
                            busy.set(false)
                            stopSelfResult(startId)
                        }
                    }
                }
            }
            ACTION_VOICE_TURN -> {
                HomeWidgetState.updateVoice(this, "در حال آماده‌سازی شنوا…")
                scope.launch { startOneShotVoiceTurn(startId) }
            }
            else -> {
                busy.set(false)
                stopSelfResult(startId)
            }
        }
        return START_NOT_STICKY
    }

    private fun startForegroundFor(microphoneRequired: Boolean) {
        val notification = buildNotification(
            if (microphoneRequired) "در حال آماده‌سازی ضبط یک نوبت گفتار" else "در حال آماده‌سازی پاسخ دستیار",
        )
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val type = when {
                microphoneRequired -> ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE
                Build.VERSION.SDK_INT >= 34 -> ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE
                else -> 0
            }
            ServiceCompat.startForeground(this, NOTIFICATION_ID, notification, type)
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
    }

    private fun switchToInferenceForeground() {
        val notification = buildNotification("گفتار دریافت شد؛ در حال تولید و پخش پاسخ")
        if (Build.VERSION.SDK_INT >= 34) {
            ServiceCompat.startForeground(
                this,
                NOTIFICATION_ID,
                notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE,
            )
        } else {
            ServiceCompat.startForeground(this, NOTIFICATION_ID, notification, 0)
        }
    }

    private suspend fun startOneShotVoiceTurn(startId: Int) {
        var localMic: LocalMicrophone? = null
        var localAsr: LocalNemoCtcAsr? = null
        try {
            check(ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
                "مجوز میکروفون قبلاً از داخل برنامه داده نشده است؛ یک بار برنامه را باز کن و مجوز صدا را فعال کن."
            }
            val store = VoiceModelStore(contentResolver, File(filesDir, "voice-models"))
            val prefs = getSharedPreferences(AlwaysOnKoochikService.PREFS, MODE_PRIVATE)
            val sttModels = store.list(VoiceModelKind.STT)
            val ttsModels = store.list(VoiceModelKind.TTS)
            val selectedStt = prefs.getString(AlwaysOnKoochikService.PREF_STT_MODEL_ID, null)
            val selectedTts = prefs.getString(AlwaysOnKoochikService.PREF_TTS_MODEL_ID, null)
            val stt = sttModels.firstOrNull {
                it.id == selectedStt && it.id.startsWith("stt-shenava-koochik-") && it.asrMode == "nemo-ctc"
            } ?: sttModels.firstOrNull {
                it.id.startsWith("stt-shenava-koochik-") && it.asrMode == "nemo-ctc"
            } ?: error("مدل Shenava Koochik وارد نشده است؛ ابتدا آن را در تنظیمات صدا وارد کن.")
            val ttsEntry = ttsModels.firstOrNull { it.id == selectedTts } ?: ttsModels.firstOrNull()
                ?: error("مدل Piper برای پاسخ صوتی وارد نشده است؛ ابتدا مدل TTS را در تنظیمات صدا وارد کن.")

            HomeWidgetState.updateVoice(this, "در حال شنیدن؛ پس از مکث کوتاه ضبط خودکار تمام می‌شود.")
            val asr = withContext(Dispatchers.IO) { LocalNemoCtcAsr(stt) }
            val mic = LocalMicrophone()
            localAsr = asr
            localMic = mic
            recognizer = asr
            microphone = mic
            voiceFinished.set(false)
            mic.start { frame ->
                if (voiceFinished.get()) return@start
                val update = runCatching { asr.accept(frame.samples, 16000, frame.speech) }.getOrElse { error ->
                    if (voiceFinished.compareAndSet(false, true)) {
                        scope.launch { finishVoiceFailure(mic, asr, error) }
                    }
                    return@start
                }
                if (update.endpoint && update.text.isNotBlank() && voiceFinished.compareAndSet(false, true)) {
                    scope.launch {
                        mic.stop()
                        microphone = null
                        runCatching { asr.close() }
                        recognizer = null
                        switchToInferenceForeground()
                        HomeWidgetState.updateVoice(this@HomeWidgetAssistantService, "گفتار دریافت شد؛ در حال پاسخ‌گویی…")
                        try {
                            val reply = generateReply(update.text)
                            HomeWidgetState.updateChat(
                                this@HomeWidgetAssistantService,
                                "پاسخ صوتی آماده است",
                                "شما: ${update.text}\n\nدستیار: $reply",
                            )
                            val engine = withContext(Dispatchers.IO) { LocalPiperTts(ttsEntry) }
                            tts = engine
                            engine.speak(reply, AtomicBoolean(false))
                            HomeWidgetState.updateVoice(this@HomeWidgetAssistantService, "پاسخ تمام شد؛ برای نوبت بعد دوباره لمس کن.")
                        } catch (t: Throwable) {
                            HomeWidgetState.updateVoice(this@HomeWidgetAssistantService, "خطا در پاسخ صوتی: ${t.message ?: t.javaClass.simpleName}")
                            HomeWidgetState.updateChat(this@HomeWidgetAssistantService, "خطا", "گفتار دریافت شد، اما پاسخ کامل نشد: ${t.message ?: t.javaClass.simpleName}")
                        } finally {
                            runCatching { tts?.close() }
                            tts = null
                            busy.set(false)
                            stopSelfResult(startId)
                        }
                    }
                }
            }

            scope.launch {
                delay(MAX_VOICE_TURN_MS)
                if (voiceFinished.compareAndSet(false, true)) {
                    mic.stop()
                    microphone = null
                    runCatching { asr.close() }
                    recognizer = null
                    HomeWidgetState.updateVoice(this@HomeWidgetAssistantService, "زمان ضبط تمام شد؛ اگر آماده‌ای دوباره لمس کن.")
                    busy.set(false)
                    stopSelfResult(startId)
                }
            }
        } catch (t: Throwable) {
            if (voiceFinished.compareAndSet(false, true)) {
                runCatching { localMic?.stop() }
                runCatching { localAsr?.close() }
                microphone = null
                recognizer = null
                HomeWidgetState.updateVoice(this, "خطا: ${t.message ?: t.javaClass.simpleName}")
                busy.set(false)
                stopSelfResult(startId)
            }
        }
    }

    private suspend fun finishVoiceFailure(mic: LocalMicrophone, asr: LocalNemoCtcAsr, error: Throwable) {
        runCatching { mic.stop() }
        runCatching { asr.close() }
        microphone = null
        recognizer = null
        HomeWidgetState.updateVoice(this, "خطا در ضبط یا تشخیص: ${error.message ?: error.javaClass.simpleName}")
        busy.set(false)
        stopSelf()
    }

    private suspend fun generateReply(userText: String): String {
        ensureInferenceReady()
        val history = container.conversationHistory
        val preferences = getSharedPreferences("home_assistant_widgets", MODE_PRIVATE)
        val storedId = preferences.getString("conversation_id", null)
        val existing = if (storedId != null) history.recent(5000).firstOrNull { it.id == storedId } else null
        val record = existing ?: history.create("گفت‌وگوی ویجت")
        if (existing == null) preferences.edit().putString("conversation_id", record.id).apply()

        val lastUpdate = AtomicLong(0L)
        val streamedText = StringBuilder()
        val session = container.createAgentSession(record.id, includeAgentTools = false) { event ->
            when (event) {
                is AgentEvent.Token -> {
                    val now = SystemClock.elapsedRealtime()
                    synchronized(streamedText) { streamedText.append(event.value) }
                    if (now - lastUpdate.get() >= 300L) {
                        lastUpdate.set(now)
                        val partial = synchronized(streamedText) { streamedText.toString() }
                        HomeWidgetState.updateChat(
                            this@HomeWidgetAssistantService,
                            "در حال تولید پاسخ…",
                            "شما: $userText\n\nدستیار: $partial",
                        )
                    }
                }
                is AgentEvent.Failed -> HomeWidgetState.updateChat(
                    this@HomeWidgetAssistantService, "خطا در تولید پاسخ", event.message,
                )
                else -> Unit
            }
        } ?: error("Agent در دسترس نیست.")
        val result = session.send(
            userText,
            InferenceSettings(maxNewTokens = 1024, enableThinking = false),
            requestedRecentMessages = 30,
        )
        return result.generation.text.ifBlank { "پاسخی از مدل دریافت نشد." }
    }

    private suspend fun ensureInferenceReady() {
        // Provider mode is honored as configured; this widget never silently switches sources.
        if (container.isCloudflareConfigured()) return
        val manager = container.modelManager ?: error("مدیر مدل محلی در دسترس نیست.")
        val active = manager.activeModel()
        if (active !is ModelResult.Success || active.value == null) {
            val restored = manager.restoreActive()
            check(restored is ModelResult.Success && restored.value != null) {
                "مدل محلی فعالی پیدا نشد؛ ابتدا یک مدل را در برنامه فعال کن."
            }
        }
    }

    private fun createNotificationChannel() {
        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(
            NotificationChannel(CHANNEL_ID, "دستیار ویجت", NotificationManager.IMPORTANCE_LOW).apply {
                description = "وضعیت پردازش درخواست‌های متنی و صوتی ویجت"
                setShowBadge(false)
            },
        )
    }

    private fun buildNotification(status: String): Notification =
        NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(applicationInfo.icon.takeIf { it != 0 } ?: android.R.drawable.ic_btn_speak_now)
            .setContentTitle("دستیار هوشمند")
            .setContentText(status)
            .setOngoing(true)
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()

    override fun onDestroy() {
        runCatching { microphone?.stop() }
        microphone = null
        runCatching { recognizer?.close() }
        recognizer = null
        runCatching { tts?.close() }
        tts = null
        scope.cancel()
        super.onDestroy()
    }
}
