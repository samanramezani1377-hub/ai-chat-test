package com.samanramezani.aichattest.ui.voice

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.app.Activity
import android.media.AudioManager
import android.net.Uri
import android.os.SystemClock
import android.os.Build
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.*
import androidx.compose.ui.platform.LocalContext
import com.samanramezani.aichattest.AppContainer
import com.samanramezani.aichattest.AlwaysOnKoochikService
import com.samanramezani.aichattest.ui.voice.VoiceConversationPage
import com.samanramezani.aichattest.voice.LocalMicrophone
import com.samanramezani.aichattest.voice.VoiceAudioInputGuard
import com.samanramezani.aichattest.voice.LocalPiperTts
import com.samanramezani.aichattest.voice.LocalStreamingAsr
import com.samanramezani.aichattest.voice.LocalAsr
import com.samanramezani.aichattest.voice.LocalQwen3Asr
import com.samanramezani.aichattest.voice.VoiceModelEntry
import com.samanramezani.aichattest.voice.VoiceModelKind
import com.samanramezani.aichattest.voice.VoiceModelStore
import com.samanramezani.aichattest.voice.PiperComponent
import com.samanramezani.aichattest.voice.SmallSttComponent
import com.samanramezani.aichattest.voice.LocalNemoCtcAsr
import com.woogit.aicore.agent.AgentEvent
import com.woogit.aicore.domain.InferenceSettings
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CompletableDeferred
import androidx.core.content.ContextCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.joinAll
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.ConcurrentLinkedQueue

@Composable
internal fun VoiceConversationHost(container: AppContainer, onBack: () -> Unit) {
    val context = LocalContext.current
    val uiScope = rememberCoroutineScope()
    val store = remember(context) { VoiceModelStore(context.contentResolver, File(context.filesDir, "voice-models")) }
    var sttModels by remember { mutableStateOf(store.list(VoiceModelKind.STT)) }
    var ttsModels by remember { mutableStateOf(store.list(VoiceModelKind.TTS)) }
    var missingKoochikComponents by remember { mutableStateOf(store.missingKoochikComponents()) }
    val alwaysOnPrefs = remember(context) { context.getSharedPreferences(AlwaysOnKoochikService.PREFS, Context.MODE_PRIVATE) }
    var alwaysOnEnabled by remember { mutableStateOf(alwaysOnPrefs.getBoolean(AlwaysOnKoochikService.PREF_ENABLED, false)) }
    var missingTtsComponents by remember { mutableStateOf(store.missingPiperComponents()) }
    var activeSttId by remember { mutableStateOf(sttModels.firstOrNull { it.id.startsWith("stt-shenava-koochik-") }?.id ?: sttModels.firstOrNull()?.id) }
    var activeTtsId by remember { mutableStateOf(ttsModels.firstOrNull()?.id) }
    var lines by remember { mutableStateOf(emptyList<VoiceLine>()) }
    var status by remember { mutableStateOf("برای شروع، مدل‌های محلی STT و TTS را انتخاب یا وارد کنید.") }
    var diagnosticLogs by remember {
        mutableStateOf(
            buildList {
                add("[${System.currentTimeMillis()}] صفحه مکالمه صوتی باز شد.")
                alwaysOnPrefs.getString(AlwaysOnKoochikService.PREF_CONVERSATION_STATE, null)?.let { add("ALWAYS-ON STATE: $it") }
                alwaysOnPrefs.getString(AlwaysOnKoochikService.PREF_LAST_STATUS, null)?.let { add("ALWAYS-ON LAST STATUS: $it") }
                alwaysOnPrefs.getString(AlwaysOnKoochikService.PREF_LAST_DIAGNOSTIC, null)?.let { add(it) }
            }.takeLast(200)
        )
    }
    fun updateVoiceStatus(value: String) {
        status = value
        diagnosticLogs = (diagnosticLogs + "[${System.currentTimeMillis()}] $value").takeLast(200)
    }
    var listening by remember { mutableStateOf(false) }
    var busy by remember { mutableStateOf(false) }
    var pendingAlwaysOnEnable by remember { mutableStateOf(false) }
    val notificationPermissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted ->
        if (pendingAlwaysOnEnable) {
            pendingAlwaysOnEnable = false
            try {
                val stt = sttModels.firstOrNull { it.id == activeSttId && it.id.startsWith("stt-shenava-koochik-") && it.asrMode == "nemo-ctc" }
                    ?: sttModels.firstOrNull { it.id.startsWith("stt-shenava-koochik-") && it.asrMode == "nemo-ctc" }
                val tts = ttsModels.firstOrNull { it.id == activeTtsId } ?: ttsModels.firstOrNull()
                check(stt != null && tts != null) { "ابتدا مدل Shenava Koochik و مدل TTS محلی را وارد کنید." }
                alwaysOnPrefs.edit().putString(AlwaysOnKoochikService.PREF_STT_MODEL_ID, stt.id)
                    .putString(AlwaysOnKoochikService.PREF_TTS_MODEL_ID, tts.id).apply()
                AlwaysOnKoochikService.start(context)
                alwaysOnEnabled = true
                updateVoiceStatus(if (granted)
                    "شنود دائمی فارسی فعال شد؛ Shenava Koochik عبارت «دینا» را در گفتار تشخیص می‌دهد."
                    else "شنود دائمی فارسی فعال شد؛ مجوز اعلان رد شد و ممکن است اعلان در کشو پنهان باشد.")
            } catch (t: Throwable) {
                alwaysOnEnabled = false
                updateVoiceStatus("VOICE-ALWAYS-001: شروع شنود فارسی ناموفق بود: ${t.message ?: t.javaClass.simpleName}")
            }
        } else if (!granted) {
            updateVoiceStatus("مجوز اعلان داده نشد؛ اعلان‌های معمولی ممکن است در اندروید ۱۳ به بالا نمایش داده نشوند.")
        }
    }

    val controller = remember(container, context) {
        VoiceConversationController(context, container,
            onStatus = { value -> uiScope.launch(Dispatchers.Main.immediate) { updateVoiceStatus(value) } },
            onDiagnostic = { value -> uiScope.launch(Dispatchers.Main.immediate) { diagnosticLogs = (diagnosticLogs + "[${System.currentTimeMillis()}] $value").takeLast(200) } },
            onListening = { value -> uiScope.launch(Dispatchers.Main.immediate) { listening = value; busy = false } },
            onLine = { speaker, text -> uiScope.launch(Dispatchers.Main.immediate) {
                if (text.isNotBlank()) lines = lines + VoiceLine(System.nanoTime(), speaker, text)
            } },
        )
    }
    DisposableEffect(controller) {
        onDispose {
            controller.close()
        }
    }

    LaunchedEffect(Unit) {
        val activity = context as? Activity
        val openedByAlwaysOn = activity?.intent?.getBooleanExtra(AlwaysOnKoochikService.EXTRA_OPEN_VOICE_FROM_ALWAYS_ON, false) == true
        val audioStartedInService = activity?.intent?.getBooleanExtra(AlwaysOnKoochikService.EXTRA_CONVERSATION_STARTED_IN_SERVICE, false) == true
        if (openedByAlwaysOn) {
            activity?.intent?.removeExtra(AlwaysOnKoochikService.EXTRA_OPEN_VOICE_FROM_ALWAYS_ON)
            activity?.intent?.removeExtra(AlwaysOnKoochikService.EXTRA_CONVERSATION_STARTED_IN_SERVICE)
            if (audioStartedInService) {
                updateVoiceStatus("دینا تشخیص داده شد؛ درخواست شروع مکالمه به سرویس پس‌زمینه داده شده است. فعال‌شدن واقعی پس از آماده‌شدن موتور و میکروفون در وضعیت تشخیص‌ها تأیید می‌شود.")
            } else {
                val stt = sttModels.firstOrNull { it.id == activeSttId }
                val ttsModel = ttsModels.firstOrNull { it.id == activeTtsId }
                if (ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
                    updateVoiceStatus("VOICE-ALWAYS-002: دینا تشخیص داده شد، اما مجوز میکروفون برای شروع مکالمه در دسترس نیست.")
                } else if (stt == null || ttsModel == null) {
                    updateVoiceStatus("VOICE-ALWAYS-003: دینا تشخیص داده شد؛ برای شروع مکالمه ابتدا مدل اصلی Shenava Koochik و مدل TTS را وارد و انتخاب کن.")
                } else {
                    busy = true
                    updateVoiceStatus("عبارت «دینا» تشخیص داده شد؛ در حال شروع مکالمهٔ صوتی…")
                    controller.start(stt, ttsModel)
                }
            }
        }
    }

    LaunchedEffect(activeSttId, activeTtsId) {
        alwaysOnPrefs.edit()
            .putString(AlwaysOnKoochikService.PREF_STT_MODEL_ID, activeSttId)
            .putString(AlwaysOnKoochikService.PREF_TTS_MODEL_ID, activeTtsId)
            .apply()
    }

    fun refreshModels() {
        sttModels = store.list(VoiceModelKind.STT)
        ttsModels = store.list(VoiceModelKind.TTS)
        if (activeSttId !in sttModels.map { it.id }) activeSttId = sttModels.firstOrNull { it.id.startsWith("stt-shenava-koochik-") }?.id ?: sttModels.firstOrNull()?.id
        if (activeTtsId !in ttsModels.map { it.id }) activeTtsId = ttsModels.firstOrNull()?.id
    }

    fun refreshImportProgress() {
        missingKoochikComponents = store.missingKoochikComponents()
        missingTtsComponents = store.missingPiperComponents()
    }

    fun importKoochikComponent(uri: Uri, component: SmallSttComponent, label: String) {
        uiScope.launch {
            busy = true
            updateVoiceStatus("در حال وارد کردن ${label} مدل اصلی Shenava Koochik…")
            try {
                val ready = withContext(Dispatchers.IO) { store.importKoochikComponent(uri, component) }
                refreshModels()
                refreshImportProgress()
                if (ready != null) activeSttId = ready.id
                val missing = if (ready == null) store.missingKoochikComponents() else emptyList()
                updateVoiceStatus(if (ready != null) "مدل اصلی Shenava Koochik آماده و انتخاب شد."
                    else "فایل ${label} ذخیره شد؛ برای تکمیل Koochik هنوز لازم است: ${missing.joinToString(" و ")}.")
            } catch (t: Throwable) {
                updateVoiceStatus("VOICE-STT-IMPORT: ${t.message ?: t.javaClass.simpleName}")
            } finally { refreshImportProgress(); busy = false }
        }
    }

    val koochikModelPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) importKoochikComponent(uri, SmallSttComponent.MODEL, "model.onnx")
    }
    val koochikTokensPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) importKoochikComponent(uri, SmallSttComponent.TOKENS, "tokens.txt")
    }

    val sttPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
        if (uri != null) uiScope.launch {
            busy = true
            updateVoiceStatus("در حال وارد کردن و اعتبارسنجی مدل تشخیص گفتار…")
            try {
                withContext(Dispatchers.IO) { store.importArchive(uri, VoiceModelKind.STT) }
                refreshModels()
                updateVoiceStatus("مدل STT وارد شد. مدل سازگار با زبان فارسی باید انتخاب شود.")
            } catch (t: Throwable) {
                updateVoiceStatus("VOICE-STT-IMPORT: ${t.message ?: t.javaClass.simpleName}")
            } finally { busy = false }
        }
    }
    fun importPiperComponent(uri: Uri, component: PiperComponent, label: String) {
        uiScope.launch {
            busy = true
            updateVoiceStatus("در حال وارد کردن ${label} و آماده‌سازی مدل محلی…")
            try {
                val ready = withContext(Dispatchers.IO) {
                    store.importPiperComponent(uri, component) { progress ->
                        uiScope.launch { updateVoiceStatus(progress) }
                    }
                }
                refreshModels()
                refreshImportProgress()
                if (ready != null) activeTtsId = ready.id
                val missing = if (ready == null) store.missingPiperComponents() else emptyList()
                updateVoiceStatus(if (ready != null) "مدل گفتار ${ready.title} آماده و انتخاب شد."
                    else "فایل ${label} ذخیره شد؛ برای تکمیل مدل Piper هنوز لازم است: ${missing.joinToString(" و ")}.")
            } catch (t: Throwable) {
                updateVoiceStatus("VOICE-PIPER-IMPORT: ${t.message ?: t.javaClass.simpleName}")
            } finally { refreshImportProgress(); busy = false }
        }
    }

    val ttsModelPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) importPiperComponent(uri, PiperComponent.MODEL, "مدل ONNX")
    }
    val ttsConfigPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) importPiperComponent(uri, PiperComponent.CONFIG, "تنظیمات Piper")
    }
    val espeakPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) importPiperComponent(uri, PiperComponent.ESPEAK_DATA, "داده‌های آواشناسی")
    }

    val ttsPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
        if (uri != null) uiScope.launch {
            busy = true
            updateVoiceStatus("در حال وارد کردن و اعتبارسنجی مدل گفتار…")
            try {
                withContext(Dispatchers.IO) { store.importArchive(uri, VoiceModelKind.TTS) }
                refreshModels()
                updateVoiceStatus("مدل TTS وارد شد.")
            } catch (t: Throwable) {
                updateVoiceStatus("VOICE-TTS-IMPORT: ${t.message ?: t.javaClass.simpleName}")
            } finally { busy = false }
        }
    }

    VoiceConversationPage(
        lines = lines,
        status = status,
        diagnosticLogs = diagnosticLogs,
        listening = listening,
        busy = busy,
        sttModels = sttModels,
        ttsModels = ttsModels,
        missingKoochikComponents = missingKoochikComponents,
        alwaysOnEnabled = alwaysOnEnabled,
        missingTtsComponents = missingTtsComponents,
        activeSttId = activeSttId,
        activeTtsId = activeTtsId,
        onBack = onBack,
        onImportStt = { sttPicker.launch(arrayOf("application/zip", "application/x-zip-compressed", "application/x-bzip2", "application/octet-stream")) },
        onImportKoochikModel = { koochikModelPicker.launch(arrayOf("application/onnx", "application/octet-stream", "*/*")) },
        onImportKoochikTokens = { koochikTokensPicker.launch(arrayOf("text/plain", "application/octet-stream", "*/*")) },
        onToggleAlwaysOn = { enabled ->
            if (enabled && ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
                updateVoiceStatus("VOICE-ALWAYS-002: برای شنود دائمی ابتدا مجوز میکروفون را بده.")
            } else {
                val stt = sttModels.firstOrNull { it.id == activeSttId && it.id.startsWith("stt-shenava-koochik-") && it.asrMode == "nemo-ctc" }
                    ?: sttModels.firstOrNull { it.id.startsWith("stt-shenava-koochik-") && it.asrMode == "nemo-ctc" }
                val tts = ttsModels.firstOrNull { it.id == activeTtsId } ?: ttsModels.firstOrNull()
                if (enabled && stt == null) {
                    updateVoiceStatus("VOICE-ALWAYS-003: برای شنود دائمی باید مدل Shenava Koochik فارسی را وارد کنی.")
                } else if (enabled && tts == null) {
                    updateVoiceStatus("VOICE-ALWAYS-005: برای گفتن «متوجه شدم» و پاسخ صوتی، مدل TTS محلی را وارد کن.")
                } else if (enabled && Build.VERSION.SDK_INT >= 33 &&
                    ContextCompat.checkSelfPermission(context, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
                ) {
                    pendingAlwaysOnEnable = true
                    notificationPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
                } else {
                    try {
                        if (enabled) {
                            alwaysOnPrefs.edit()
                                .putString(AlwaysOnKoochikService.PREF_STT_MODEL_ID, stt!!.id)
                                .putString(AlwaysOnKoochikService.PREF_TTS_MODEL_ID, tts!!.id)
                                .apply()
                            AlwaysOnKoochikService.start(context)
                        } else AlwaysOnKoochikService.stop(context)
                        alwaysOnEnabled = enabled
                        updateVoiceStatus(if (enabled)
                            "شنود دائمی فارسی فعال شد؛ مدل Shenava Koochik به‌صورت محلی به دنبال «دینا» می‌گردد."
                            else "شنود دائمی فارسی خاموش شد.")
                    } catch (t: Throwable) {
                        alwaysOnEnabled = false
                        updateVoiceStatus("VOICE-ALWAYS-001: شروع/توقف شنود ناموفق بود: ${t.message ?: t.javaClass.simpleName}")
                    }
                }
            }
        },
        onImportTts = { ttsPicker.launch(arrayOf("application/zip", "application/x-zip-compressed", "application/x-bzip2", "application/octet-stream")) },
        onImportTtsModel = { ttsModelPicker.launch(arrayOf("application/onnx", "application/octet-stream", "*/*")) },
        onImportTtsConfig = { ttsConfigPicker.launch(arrayOf("application/json", "text/json", "text/plain", "*/*")) },
        onImportEspeakData = { espeakPicker.launch(arrayOf("application/x-bzip2", "application/zip", "application/octet-stream", "*/*")) },
        onSelectStt = { activeSttId = it; alwaysOnPrefs.edit().putString(AlwaysOnKoochikService.PREF_STT_MODEL_ID, it).apply() },
        onSelectTts = { activeTtsId = it; alwaysOnPrefs.edit().putString(AlwaysOnKoochikService.PREF_TTS_MODEL_ID, it).apply() },
        onStart = {
            val stt = sttModels.firstOrNull { it.id == activeSttId }
            val tts = ttsModels.firstOrNull { it.id == activeTtsId }
            if (stt == null || tts == null) updateVoiceStatus("VOICE-MODEL-001: ابتدا مدل‌های STT و TTS را وارد و انتخاب کنید.")
            else {
                alwaysOnPrefs.edit()
                    .putString(AlwaysOnKoochikService.PREF_STT_MODEL_ID, stt.id)
                    .putString(AlwaysOnKoochikService.PREF_TTS_MODEL_ID, tts.id)
                    .apply()
                if (alwaysOnEnabled) { AlwaysOnKoochikService.stop(context); alwaysOnEnabled = false }
                busy = true
                controller.start(stt, tts)
            }
        },
        onStop = {
            controller.stop()
            val serviceOwnedState = alwaysOnPrefs.getBoolean(AlwaysOnKoochikService.PREF_ENABLED, false) &&
                alwaysOnPrefs.getString(AlwaysOnKoochikService.PREF_CONVERSATION_STATE, "").orEmpty() in
                setOf("CONVERSATION_ACTIVE", "CONVERSATION_STARTING", "CONVERSATION_START_REQUESTED", "DINA_DETECTED")
            if (serviceOwnedState) {
                runCatching {
                    context.startService(Intent(context, AlwaysOnKoochikService::class.java).setAction(AlwaysOnKoochikService.ACTION_STOP_CONVERSATION))
                }
            }
        },
        onInterrupt = { controller.interrupt("گفتار مدل با درخواست کاربر قطع شد.") },
    )
}


/**
 * Strip private reasoning and presentation markup before displaying or speaking an answer.
 * TTS receives only the final sanitized response, never partial generation tokens.
 */
private fun sanitizeAssistantVoiceText(raw: String): String {
    var text = raw
        .replace(Regex("(?is)<think\\b(?!\\s*/)[^>]*>.*?(?:</think\\s*>|<think\\s*/\\s*>|$)"), " ")
        .replace(Regex("(?is)<think\\s*/\\s*>"), " ")
        .replace(Regex("(?is)</?think\\b[^>]*>"), " ")
        .replace(Regex("(?is)<(analysis|reasoning|scratchpad)\\b[^>]*>.*?</\\1\\s*>"), " ")
        .replace(Regex("(?m)^\\s*`{3,}[^\\n]*"), " ")
        .replace(Regex("(?m)^\\s*#{1,6}\\s*"), "")
        .replace(Regex("(?m)^\\s*(?:[-*_]\\s*){3,}$"), " ")
        .replace(Regex("(?m)^\\s*[-*+]\\s+"), "")
        .replace(Regex("(?m)^\\s*\\d+[.)]\\s+"), "")
        .replace(Regex("(?<!\\w)(?:\\*\\*|__)(?=\\S)|(?<=\\S)(?:\\*\\*|__)"), "")
        .replace(Regex("(?<!\\w)[*_~](?=\\S)|(?<=\\S)[*_~](?!\\w)"), "")
        .replace(Regex("(?is)<[^>]+>"), " ")
        .replace(Regex("[ \\t]+"), " ")
        .replace(Regex(" *\\n *")) { "\n" }
        .replace(Regex("\\n{3,}")) { "\n\n" }
        .trim()
    // Also discard orphaned reasoning markers from malformed model output.
    val marker = Regex("(?i)<think\\s*/?>")
    while (marker.containsMatchIn(text)) {
        val match = marker.find(text) ?: break
        text = text.removeRange(match.range).trimStart()
    }
    return text
}

internal class VoiceConversationController(
    private val context: Context,
    private val container: AppContainer,
    private val onStatus: (String) -> Unit,
    private val onDiagnostic: (String) -> Unit,
    private val onListening: (Boolean) -> Unit,
    private val onLine: (String, String) -> Unit,
) {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val cancelledSpeech = AtomicReference<AtomicBoolean?>(null)
    private val speaking = AtomicBoolean(false)
    private val turnRunning = AtomicBoolean(false)
    private val bargeInTriggered = AtomicBoolean(false)
    private val bargeInCapturePending = AtomicBoolean(false)
    private val bargeInCaptureStartedAtMs = AtomicLong(0L)
    // Playback guard is also applied to ASR input, not only to barge-in detection.
    // Loudspeaker echo can remain in the microphone after AudioTrack stops.
    private val lastTtsPlaybackAtMs = AtomicLong(0L)
    // Bounded queue preserves short consecutive user turns without unbounded memory growth.
    private val pendingUtterances = ConcurrentLinkedQueue<String>()
    private val startInProgress = AtomicBoolean(false)
    private val lifecycleEpoch = AtomicLong(0L)
    @Volatile private var microphone: LocalMicrophone? = null
    private var recognizer: LocalAsr? = null
    private var tts: LocalPiperTts? = null
    private var frameJob: Job? = null
    private var turnJob: Job? = null
    private var speechJob: Job? = null
    private var conversationId: String? = null
    private var previousAudioMode: Int? = null
    @Volatile private var closed = false

    private fun enterCommunicationAudioMode() {
        val manager = context.getSystemService(AudioManager::class.java) ?: return
        if (previousAudioMode == null) previousAudioMode = manager.mode
        manager.mode = AudioManager.MODE_NORMAL
        // Media usage follows the device's main speaker and media-volume stream, not the earpiece/call route.
    }

    private fun restoreAudioMode() {
        val previous = previousAudioMode ?: return
        previousAudioMode = null
        runCatching {
            context.getSystemService(AudioManager::class.java)?.mode = previous
        }
    }

    fun start(
        stt: VoiceModelEntry,
        ttsModel: VoiceModelEntry,
        initialUtterance: String? = null,
        acknowledgeWake: Boolean = false,
        speechStartGate: CompletableDeferred<Unit>? = null,
    ) {
        if (closed || microphone != null || !startInProgress.compareAndSet(false, true)) return
        val startEpoch = lifecycleEpoch.get()
        scope.launch {
            try {
                onStatus("در حال آماده‌سازی موتورهای محلی صوت…")
                val engines = withContext(Dispatchers.IO) {
                    val asr = when (stt.asrMode) {
                        "qwen3-asr" -> LocalQwen3Asr(stt)
                        "nemo-ctc" -> LocalNemoCtcAsr(stt)
                        else -> LocalStreamingAsr(stt)
                    }
                    try {
                        asr to LocalPiperTts(ttsModel)
                    } catch (t: Throwable) {
                        runCatching { asr.close() }
                        throw t
                    }
                }
                if (closed || lifecycleEpoch.get() != startEpoch) {
                    runCatching { engines.first.close() }
                    runCatching { engines.second.close() }
                    return@launch
                }
                recognizer = engines.first
                tts = engines.second
                enterCommunicationAudioMode()
                if (conversationId == null) conversationId = container.conversationHistory.create("مکالمه صوتی").id
                // Keep capture independent from ASR. The queue is bounded to avoid seconds of
                // stale microphone audio accumulating while an offline recognizer is decoding.
                val frames = Channel<LocalMicrophone.Frame>(capacity = 128)
                val mic = LocalMicrophone()
                val loudSpeechFrames = AtomicInteger(0)
                val audioDropReported = AtomicBoolean(false)
                val echoSuppressionReported = AtomicBoolean(false)
                if (closed || lifecycleEpoch.get() != startEpoch) {
                    runCatching { mic.stop() }
                    cleanupEngines()
                    restoreAudioMode()
                    return@launch
                }
                microphone = mic
                mic.start { frame ->
                    val nowMs = SystemClock.elapsedRealtime()
                    if (VoiceAudioInputGuard.shouldSuppress(nowMs)) {
                        loudSpeechFrames.set(0)
                        return@start
                    }
                    val aecActive = mic.isEchoCancellationActive
                    val sincePlaybackMs = nowMs - lastTtsPlaybackAtMs.get()
                    // AEC shortens the safe echo tail; devices without it use a conservative
                    // guard because speaker leakage cannot be distinguished from speech by RMS.
                    val echoTailMs = if (aecActive) 450L else 1200L
                    val assistantAudioActive = speaking.get() || sincePlaybackMs in 0L..echoTailMs
                    val threshold = if (assistantAudioActive) 0.060f else 0.035f
                    val requiredFrames = if (assistantAudioActive) 6 else 3
                    // Barge-in during speaker playback is enabled only when platform AEC is
                    // actually active. Without AEC, loudspeaker echo is indistinguishable from
                    // a user voice using RMS alone, so fail closed instead of spawning false turns.
                    val candidate = turnRunning.get() && frame.speech && frame.rms > threshold &&
                        (!assistantAudioActive || aecActive)
                    if (candidate) {
                        if (loudSpeechFrames.incrementAndGet() >= requiredFrames &&
                            bargeInTriggered.compareAndSet(false, true)) {
                            bargeInCapturePending.set(aecActive)
                            bargeInCaptureStartedAtMs.set(nowMs)
                            interrupt("صدای کاربر تشخیص داده شد؛ تولید یا پخش پاسخ قبلی متوقف شد.")
                            loudSpeechFrames.set(0)
                            if (aecActive) frames.trySend(frame)
                        }
                    } else {
                        loudSpeechFrames.set(0)
                    }

                    // Never let assistant output or its acoustic tail enter the recognizer.
                    // If AEC is unavailable, barge-in during playback is disabled by design.
                    if (bargeInCapturePending.get() && nowMs - bargeInCaptureStartedAtMs.get() > 5000L) {
                        bargeInCapturePending.set(false)
                    }
                    val preserveUserAudio = bargeInCapturePending.get() && aecActive
                    if ((speaking.get() && !preserveUserAudio) ||
                        (assistantAudioActive && !bargeInCapturePending.get())) {
                        if (echoSuppressionReported.compareAndSet(false, true)) {
                            scope.launch {
                                onDiagnostic("VOICE-ECHO-001: ورودی ASR هنگام پخش و تا " + echoTailMs + "ms پس از آن مسدود شد؛ AEC=" +
                                    if (aecActive) "فعال" else "غیرفعال")
                            }
                        }
                    } else {
                        echoSuppressionReported.set(false)
                    val sent = frames.trySend(frame)
                    if (sent.isFailure) {
                        // Prefer the newest live audio over stale buffered audio. The consumer
                        // remains bounded and latency does not grow by several seconds.
                        frames.tryReceive()
                        frames.trySend(frame)
                        if (audioDropReported.compareAndSet(false, true)) {
                            scope.launch {
                                onDiagnostic("VOICE-AUDIO-001: صف صوتی پر شد؛ قدیمی‌ترین فریم حذف شد تا تأخیر زنده محدود بماند.")
                                onStatus("VOICE-AUDIO-001: تشخیص گفتار زیر بار پردازش عقب افتاده است؛ ممکن است بخشی از جمله از دست برود.")
                            }
                        }
                    } else {
                        audioDropReported.set(false)
                    }
                    }
                }
                onDiagnostic("VOICE-AUDIO-SETUP: AEC=" + (if (mic.isEchoCancellationActive) "active" else "unavailable") +
                    " NS=" + (if (mic.isNoiseSuppressionActive) "active" else "unavailable") + " sampleRate=16000 frameMs=40")
                frameJob = scope.launch {
                    onListening(true)
                    onStatus("در حال گوش‌دادن؛ گفتار و پردازش کاملاً روی دستگاه است.")
                    for (frame in frames) {
                        val asrStartedAtMs = SystemClock.elapsedRealtime()
                        val update = try { recognizer?.accept(frame.samples, 16000, frame.speech) } catch (t: Throwable) {
                            onDiagnostic("VOICE-STT-002\n${t.stackTraceToString()}")
                            onStatus("VOICE-STT-002: خطا در تشخیص گفتار: ${t.message ?: t.javaClass.simpleName}")
                            null
                        } ?: continue
                        val asrDurationMs = SystemClock.elapsedRealtime() - asrStartedAtMs
                        if (update.endpoint || asrDurationMs >= 100L) {
                            onDiagnostic("VOICE-STT-PERF: acceptMs=$asrDurationMs endpoint=${update.endpoint} textChars=${update.text.length}")
                        }
                        if (update.text.isNotBlank()) onStatus("در حال شنیدن: ${update.text}")
                        if (update.endpoint) bargeInCapturePending.set(false)
                        if (update.endpoint && update.text.isNotBlank()) {
                            val utterance = update.text.trim()
                            if (utterance.isNotBlank()) {
                                if (turnRunning.compareAndSet(false, true)) {
                                    bargeInTriggered.set(false)
                                    bargeInCapturePending.set(false)
                                    onLine("شما", utterance)
                                    turnJob = scope.launch { processTurn(utterance) }
                                } else {
                                    // Keep the latest finalized utterance instead of silently dropping it
                                    // while the previous generation is being cancelled.
                                    if (pendingUtterances.size >= 4) {
                                        pendingUtterances.poll()
                                        onDiagnostic("VOICE-TURN-001: صف گفتارهای منتظر پر بود؛ قدیمی‌ترین درخواست حذف شد.")
                                    }
                                    pendingUtterances.offer(utterance)
                                    onLine("شما", utterance)
                                    onStatus("گفتار جدید دریافت شد؛ پس از توقف امن پاسخ قبلی پردازش می‌شود.")
                                }
                            }
                        }
                    }
                }
                val initial = initialUtterance?.trim().orEmpty()
                if (initial.isNotBlank()) {
                    onLine("شما", initial)
                    turnRunning.set(true)
                    turnJob = scope.launch {
                        processTurn(initial, acknowledgeWake = acknowledgeWake, speechStartGate = speechStartGate)
                    }
                } else if (acknowledgeWake) {
                    scope.launch { speakWakeAcknowledgement() }
                }
            } catch (t: Throwable) {
                startInProgress.set(false)
                onDiagnostic("VOICE-START-001\n${t.stackTraceToString()}")
                onStatus("VOICE-START-001: شروع مکالمه ناموفق بود: ${t.message ?: t.javaClass.simpleName}")
                runCatching { microphone?.stop() }
                microphone = null
                cleanupEngines()
                restoreAudioMode()
                onListening(false)
            } finally {
                startInProgress.set(false)
            }
        }
    }

    private suspend fun processTurn(
        utterance: String,
        acknowledgeWake: Boolean = false,
        speechStartGate: CompletableDeferred<Unit>? = null,
    ) {
        // A cancelled native TTS call may finish its current ONNX inference before returning.
        // Never enter the same native TTS object concurrently for a barge-in replacement turn.
        speechJob?.takeIf { !it.isCompleted }?.join()
        val turnCancel = AtomicBoolean(false)
        cancelledSpeech.set(turnCancel)
        val queue = Channel<String>(Channel.UNLIMITED)
        val activeSpeechJob = scope.launch {
            try {
                for (segment in queue) {
                    if (turnCancel.get()) break
                    // The local wake acknowledgement can play while the LLM starts decoding.
                    // Gate only response playback so assistant speech never overlaps the acknowledgement.
                    speechStartGate?.await()
                    if (turnCancel.get()) break
                    val engine = tts ?: break
                    val segmentPlaybackStarted = AtomicBoolean(false)
                    try {
                        // Mark speaking only when audio playback actually starts, not during synthesis.
                        engine.speak(
                            segment,
                            turnCancel,
                            onStarted = {
                                val playbackStart = SystemClock.elapsedRealtime()
                                segmentPlaybackStarted.set(true)
                                lastTtsPlaybackAtMs.set(playbackStart)
                                speaking.set(true)
                            },
                            onSynthesisComplete = { synthesisMs ->
                                onDiagnostic("VOICE-TTS-PERF: synthesisMs=$synthesisMs chars=${segment.length}")
                            },
                        )
                    } finally {
                        // Record the actual end even when cancellation interrupts AudioTrack.write().
                        if (segmentPlaybackStarted.get()) lastTtsPlaybackAtMs.set(SystemClock.elapsedRealtime())
                        speaking.set(false)
                    }
                }
            } catch (_: CancellationException) {
                speaking.set(false)
            } catch (t: Throwable) {
                speaking.set(false)
                onDiagnostic("VOICE-TTS-005\n${t.stackTraceToString()}")
                onStatus("VOICE-TTS-005: تولید یا پخش گفتار ناموفق بود: ${t.message ?: t.javaClass.simpleName}")
            }
        }
        speechJob = activeSpeechJob
        if (acknowledgeWake) {
            queue.trySend("متوجه شدم")
            onLine("دستیار", "متوجه شدم")
        }
        val pending = StringBuilder()
        fun enqueueCompleteSentences(token: String, flush: Boolean = false) {
            for (char in token) {
                pending.append(char)
                if (char in charArrayOf('.', '!', '?', '؟', '؛', '\n') || pending.length >= 120 && char == ' ') {
                    val segment = pending.toString().trim()
                    pending.clear()
                    if (segment.isNotBlank()) queue.trySend(segment)
                }
            }
            if (flush) {
                val segment = pending.toString().trim()
                pending.clear()
                if (segment.isNotBlank()) queue.trySend(segment)
            }
        }

        try {
            onStatus("در حال پردازش گفتار و تولید پاسخ محلی…")
            val id = conversationId ?: error("VOICE-CHAT-001: شناسه مکالمه ایجاد نشده است.")
            val modelFailure = AtomicReference<String?>(null)
            val session = container.createAgentSession(id, eventSink = { event ->
                // Avoid speaking streamed fragments: <think> and Markdown may be split across callbacks.
                if (event is AgentEvent.Failed) {
                    modelFailure.compareAndSet(null, event.message)
                    onDiagnostic("VOICE-LLM-001: ${event.message}")
                    onStatus("VOICE-LLM-001: ${event.message}")
                }
            }, includeAgentTools = false) ?: error("VOICE-CHAT-002: نشست مدل محلی در دسترس نیست.")
            // Voice is latency-sensitive: skip Qwen reasoning, cap response length, and keep
            // a short rolling history so each decode step attends to less accumulated KV state.
            val result = session.send(
                utterance,
                InferenceSettings(maxNewTokens = 384, recentMessages = 12, enableThinking = false),
                requestedRecentMessages = 12,
            )
            if (!turnCancel.get()) {
                val generation = result.generation
                val answer = sanitizeAssistantVoiceText(generation.text)
                if (answer.isNotBlank()) enqueueCompleteSentences(answer, flush = true)
                queue.close()
                activeSpeechJob.join()
                val outputTokens = generation.outputTokens
                val generationMs = generation.generationTimeMs
                val firstTokenMs = generation.firstTokenTimeMs
                val totalTps = if (outputTokens != null && generationMs != null && generationMs > 0) {
                    outputTokens * 1000.0 / generationMs
                } else null
                val decodeWindowMs = if (generationMs != null && firstTokenMs != null) generationMs - firstTokenMs else null
                val postFirstTokenTps = if (outputTokens != null && decodeWindowMs != null && decodeWindowMs > 0) {
                    outputTokens * 1000.0 / decodeWindowMs
                } else null
                onDiagnostic(
                    "VOICE-PERF: outputTokens=${outputTokens ?: "n/a"} " +
                        "TTFTMs=${firstTokenMs ?: "n/a"} generationMs=${generationMs ?: "n/a"} " +
                        "totalTokensPerSec=${totalTps?.let { "%.2f".format(java.util.Locale.US, it) } ?: "n/a"} " +
                        "postFirstTokenEstimateTokensPerSec=${postFirstTokenTps?.let { "%.2f".format(java.util.Locale.US, it) } ?: "n/a"} " +
                        "enableThinking=false recentMessages=12 maxNewTokens=384"
                )
                if (answer.isNotBlank()) {
                    onLine("دستیار", answer)
                    onStatus(if (speaking.get()) "در حال پخش پاسخ…" else "پاسخ آماده است؛ می‌توانید صحبت کنید.")
                } else {
                    val failure = modelFailure.get()
                    val detail = failure ?: "موتور مدل خروجی متنی تولید نکرد؛ گزارش اجرای مدل را بررسی کنید."
                    onDiagnostic("VOICE-LLM-002: empty_response outputTokens=${result.generation.outputTokens ?: "n/a"} generationMs=${result.generation.generationTimeMs ?: "n/a"} detail=$detail")
                    onStatus(if (failure != null) "VOICE-LLM-001: $failure" else "VOICE-LLM-002: پاسخ خالی دریافت شد؛ تولید متن مدل ناموفق بود.")
                }
            } else {
                queue.close()
                activeSpeechJob.cancel()
            }
        } catch (t: CancellationException) {
            queue.close()
            activeSpeechJob.cancel()
        } catch (t: Throwable) {
            queue.close()
            activeSpeechJob.cancel()
            onDiagnostic("VOICE-CHAT-003\n${t.stackTraceToString()}")
            onStatus("VOICE-CHAT-003: اجرای مکالمه ناموفق بود: ${t.message ?: t.javaClass.simpleName}")
        } finally {
            speaking.set(false)
            cancelledSpeech.compareAndSet(turnCancel, null)
            // Keep a cancelled TTS job referenced until it actually completes; the next turn joins it
            // before entering the same native Piper engine again.
            if (speechJob === activeSpeechJob && activeSpeechJob.isCompleted) speechJob = null
            turnRunning.set(false)
            bargeInTriggered.set(false)
            val pending = pendingUtterances.poll()
            if (!closed && pending != null && turnRunning.compareAndSet(false, true)) {
                onStatus("در حال پردازش گفتار جدید…")
                turnJob = scope.launch { processTurn(pending) }
            }
        }
    }

    private suspend fun speakWakeAcknowledgement() {
        val engine = tts ?: return
        val cancelled = AtomicBoolean(false)
        try {
            engine.speak(
                "متوجه شدم",
                cancelled,
                onStarted = {
                    lastTtsPlaybackAtMs.set(SystemClock.elapsedRealtime())
                    speaking.set(true)
                },
                onSynthesisComplete = { synthesisMs ->
                    onDiagnostic("VOICE-TTS-ACK: synthesisMs=$synthesisMs")
                },
            )
            onLine("دستیار", "متوجه شدم")
        } catch (t: Throwable) {
            onDiagnostic("VOICE-TTS-ACK-001: ${t.stackTraceToString()}")
            onStatus("VOICE-TTS-ACK-001: پخش تأیید بیدارباش ناموفق بود: ${t.message ?: t.javaClass.simpleName}")
        } finally {
            speaking.set(false)
            lastTtsPlaybackAtMs.set(SystemClock.elapsedRealtime())
        }
    }

    fun interrupt(message: String) {
        // Start the echo-tail window from interruption only if speaker output was actually active.
        // Interrupting LLM generation before TTS must not mute the next user utterance.
        if (speaking.get()) lastTtsPlaybackAtMs.set(SystemClock.elapsedRealtime())
        cancelledSpeech.get()?.set(true)
        speaking.set(false)
        scope.launch { container.modelRuntime.stopGeneration() }
        turnJob?.cancel(CancellationException("barge-in"))
        turnJob = null
        speechJob?.cancel()
        // Let processTurn.finally release the active turn and dispatch any newly finalized utterance.
        onStatus(message)
    }

    fun stop(): Job {
        lifecycleEpoch.incrementAndGet()
        val oldFrameJob = frameJob
        frameJob = null
        oldFrameJob?.cancel()
        val oldTurnJob = turnJob
        turnJob = null
        oldTurnJob?.cancel()
        val oldSpeechJob = speechJob
        oldSpeechJob?.cancel()
        cancelledSpeech.getAndSet(null)?.set(true)
        bargeInCapturePending.set(false)
        speaking.set(false)
        val oldMicrophone = microphone
        microphone = null
        oldMicrophone?.stop()
        return scope.launch {
            listOfNotNull(oldFrameJob, oldTurnJob, oldSpeechJob).joinAll()
            cleanupEngines()
            restoreAudioMode()
            onListening(false)
            onStatus("مکالمه متوقف شد.")
        }
    }

    private fun cleanupEngines() {
        runCatching { recognizer?.close() }; recognizer = null
        runCatching { tts?.close() }; tts = null
    }

    fun close() {
        closed = true
        stop().invokeOnCompletion { scope.cancel() }
    }
}
