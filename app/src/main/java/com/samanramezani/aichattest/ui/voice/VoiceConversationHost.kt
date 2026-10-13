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
import com.samanramezani.aichattest.isDinaSleepCommand
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
import com.samanramezani.aichattest.voice.LayaDecisionModelStore
import com.samanramezani.aichattest.voice.PocketTtsFarsiBundleStore
import com.samanramezani.aichattest.voice.LayaDecisionEngine
import com.samanramezani.aichattest.voice.DinaDeviceCommandExecutor
import com.samanramezani.aichattest.voice.extractDinaCommand
import com.woogit.aicore.agent.AgentEvent
import com.woogit.aicore.conversation.ConversationMessage
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
    val layaStore = remember(context) {
        LayaDecisionModelStore(context.contentResolver, File(context.filesDir, "voice-models/laya-int4"))
    }
    var layaReady by remember { mutableStateOf(layaStore.isReady()) }
    var layaStatus by remember { mutableStateOf(layaStore.status()) }
    val pocketTtsStore = remember(context) {
        PocketTtsFarsiBundleStore(context.contentResolver, File(context.filesDir, "voice-models/pocket-tts-farsi-v2"))
    }
    var pocketTtsReady by remember { mutableStateOf(pocketTtsStore.isReady()) }
    var pocketTtsStatus by remember { mutableStateOf(pocketTtsStore.status()) }
    var pocketTtsSelected by remember { mutableStateOf(context.getSharedPreferences(AlwaysOnKoochikService.PREFS, Context.MODE_PRIVATE).getBoolean("pocket_tts_selected", false)) }
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
    var serviceOwnsConversation by remember { mutableStateOf(false) }
    var lastServiceStatus by remember { mutableStateOf("") }
    var lastServiceDiagnostic by remember { mutableStateOf("") }
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
                val pocketCanSpeak = pocketTtsSelected && pocketTtsStore.isReady()
                check(stt != null && (tts != null || pocketCanSpeak)) {
                    "ابتدا Shenava Koochik و یک موتور TTS محلی معتبر را انتخاب کنید."
                }
                alwaysOnPrefs.edit().putString(AlwaysOnKoochikService.PREF_STT_MODEL_ID, stt.id)
                    .putString(AlwaysOnKoochikService.PREF_TTS_MODEL_ID, tts?.id).apply()
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
            LayaDecisionEngine(File(context.filesDir, "voice-models/laya-int4")),
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

    DisposableEffect(alwaysOnPrefs, serviceOwnsConversation) {
        if (serviceOwnsConversation) {
            fun syncServiceState() {
                listening = alwaysOnPrefs.getBoolean(AlwaysOnKoochikService.PREF_MIC_LISTENING, false)
                val latestStatus = alwaysOnPrefs.getString(AlwaysOnKoochikService.PREF_LAST_STATUS, "").orEmpty()
                if (latestStatus.isNotBlank() && latestStatus != lastServiceStatus) {
                    lastServiceStatus = latestStatus
                    status = latestStatus
                    diagnosticLogs = (diagnosticLogs + "[${System.currentTimeMillis()}] SERVICE: $latestStatus").takeLast(200)
                }
                val latestDiagnostic = alwaysOnPrefs.getString(AlwaysOnKoochikService.PREF_LAST_DIAGNOSTIC, "").orEmpty()
                if (latestDiagnostic.isNotBlank() && latestDiagnostic != lastServiceDiagnostic) {
                    lastServiceDiagnostic = latestDiagnostic
                    diagnosticLogs = (diagnosticLogs + latestDiagnostic).takeLast(200)
                }
            }
            val listener = android.content.SharedPreferences.OnSharedPreferenceChangeListener { _, key ->
                if (key == null || key == AlwaysOnKoochikService.PREF_MIC_LISTENING ||
                    key == AlwaysOnKoochikService.PREF_CONVERSATION_STATE ||
                    key == AlwaysOnKoochikService.PREF_LAST_STATUS ||
                    key == AlwaysOnKoochikService.PREF_LAST_DIAGNOSTIC
                ) syncServiceState()
            }
            syncServiceState()
            alwaysOnPrefs.registerOnSharedPreferenceChangeListener(listener)
            onDispose { alwaysOnPrefs.unregisterOnSharedPreferenceChangeListener(listener) }
        } else {
            onDispose { }
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
                serviceOwnsConversation = true
                updateVoiceStatus("دینا تشخیص داده شد؛ وضعیت واقعی میکروفون و موتور مکالمه از سرویس پس‌زمینه دنبال می‌شود.")
            } else {
                val stt = sttModels.firstOrNull { it.id == activeSttId }
                val ttsModel = ttsModels.firstOrNull { it.id == activeTtsId }
                val pocketCanSpeak = pocketTtsSelected && pocketTtsStore.isReady()
                if (ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
                    updateVoiceStatus("VOICE-ALWAYS-002: دینا تشخیص داده شد، اما مجوز میکروفون برای شروع مکالمه در دسترس نیست.")
                } else if (stt == null || (ttsModel == null && !pocketCanSpeak)) {
                    updateVoiceStatus("VOICE-ALWAYS-003: دینا تشخیص داده شد؛ ابتدا Shenava Koochik و یک موتور TTS محلی معتبر را انتخاب کن.")
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

    val layaPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
        if (uri != null) uiScope.launch {
            busy = true
            layaStatus = "در حال وارد کردن و اعتبارسنجی مدل محلی Laya INT4…"
            try {
                layaStatus = withContext(Dispatchers.IO) { layaStore.importBundle(uri) }
                layaReady = layaStore.isReady()
                updateVoiceStatus(layaStatus)
            } catch (t: Throwable) {
                layaStatus = "LAYA-IMPORT: ${t.message ?: t.javaClass.simpleName}"
                updateVoiceStatus(layaStatus)
            } finally { busy = false }
        }
    }


    val pocketTtsPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
        if (uri != null) uiScope.launch {
            busy = true
            pocketTtsStatus = "در حال وارد کردن و اعتبارسنجی بستهٔ Pocket TTS…"
            try {
                pocketTtsStatus = withContext(Dispatchers.IO) { pocketTtsStore.importArchive(uri) }
                pocketTtsReady = pocketTtsStore.isReady()
                if (pocketTtsReady) {
                    // A successfully validated Pocket bundle is a complete TTS engine option.
                    // Select it immediately so users do not need a separate hidden/disabled action.
                    pocketTtsSelected = true
                    alwaysOnPrefs.edit().putBoolean("pocket_tts_selected", true).apply()
                    pocketTtsStatus = "بستهٔ Pocket TTS معتبر است و برای پاسخ‌های مکالمه انتخاب شد."
                }
                updateVoiceStatus(pocketTtsStatus)
            } catch (t: Throwable) {
                pocketTtsStatus = "POCKET-TTS-IMPORT: ${t.message ?: t.javaClass.simpleName}"
                updateVoiceStatus(pocketTtsStatus)
            } finally { busy = false }
        }
    }

    fun downloadPocketTtsBundle() {
        uiScope.launch {
            busy = true
            val finished = AtomicBoolean(false)
            pocketTtsStatus = "در حال دریافت بستهٔ کامل Pocket TTS Farsi v2 از Hugging Face…"
            updateVoiceStatus(pocketTtsStatus)
            try {
                val result = withContext(Dispatchers.IO) {
                    pocketTtsStore.downloadArchive { progress ->
                        uiScope.launch(Dispatchers.Main.immediate) {
                            if (!finished.get()) pocketTtsStatus = progress
                        }
                    }
                }
                finished.set(true)
                pocketTtsStatus = result
                updateVoiceStatus(result)
            } catch (t: Throwable) {
                finished.set(true)
                pocketTtsStatus = "POCKET-TTS-DOWNLOAD: ${t.message ?: t.javaClass.simpleName}"
                updateVoiceStatus(pocketTtsStatus)
            } finally { finished.set(true); busy = false }
        }
    }

    fun downloadLayaModel() {
        uiScope.launch {
            busy = true
            val downloadFinished = AtomicBoolean(false)
            layaStatus = "در حال اتصال به Hugging Face و دریافت بستهٔ رسمی Laya INT4…"
            updateVoiceStatus(layaStatus)
            try {
                val result = withContext(Dispatchers.IO) {
                    layaStore.downloadOfficialBundle { progress ->
                        uiScope.launch(Dispatchers.Main.immediate) {
                            if (!downloadFinished.get()) layaStatus = progress
                        }
                    }
                }
                downloadFinished.set(true)
                layaReady = layaStore.isReady()
                layaStatus = result
                updateVoiceStatus(result)
            } catch (t: Throwable) {
                downloadFinished.set(true)
                layaReady = layaStore.isReady()
                layaStatus = "LAYA-DOWNLOAD: ${t.message ?: t.javaClass.simpleName}"
                updateVoiceStatus(layaStatus)
            } finally {
                downloadFinished.set(true)
                busy = false
            }
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
        layaModelReady = layaReady,
        layaStatus = layaStatus,
        pocketTtsReady = pocketTtsReady,
        pocketTtsStatus = pocketTtsStatus,
        pocketTtsSelected = pocketTtsSelected,
        onPocketTtsSelectedChange = { selected ->
            if (selected && !pocketTtsStore.isReady()) {
                updateVoiceStatus("POCKET-TTS-SELECT-001: ابتدا بستهٔ Pocket TTS را دانلود و وارد کن.")
            } else {
                pocketTtsSelected = selected
                context.getSharedPreferences(AlwaysOnKoochikService.PREFS, Context.MODE_PRIVATE).edit().putBoolean("pocket_tts_selected", selected).apply()
                updateVoiceStatus(if (selected) "موتور Pocket TTS برای پاسخ‌های مکالمه انتخاب شد؛ در شروع یا شروع مجدد مکالمه اجرا می‌شود." else "موتور Piper برای پاسخ‌های مکالمه انتخاب شد.")
            }
        },
        onDownloadPocketTts = { downloadPocketTtsBundle() },
        onImportPocketTts = { pocketTtsPicker.launch(arrayOf("application/zip", "application/octet-stream", "*/*")) },
        onImportLaya = { layaPicker.launch(arrayOf("application/zip", "application/octet-stream", "*/*")) },
        onDownloadLaya = { downloadLayaModel() },
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
                val pocketCanSpeak = pocketTtsSelected && pocketTtsStore.isReady()
                if (enabled && stt == null) {
                    updateVoiceStatus("VOICE-ALWAYS-003: برای شنود دائمی باید مدل Shenava Koochik فارسی را وارد کنی.")
                } else if (enabled && tts == null && !pocketCanSpeak) {
                    updateVoiceStatus("VOICE-ALWAYS-005: برای گفتن «متوجه شدم» و پاسخ صوتی، یک موتور TTS محلی معتبر انتخاب کن.")
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
                                .putString(AlwaysOnKoochikService.PREF_TTS_MODEL_ID, tts?.id)
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
            if (stt == null || (tts == null && !(pocketTtsSelected && pocketTtsReady))) updateVoiceStatus("VOICE-MODEL-001: ابتدا مدل‌های STT و TTS را وارد و انتخاب کنید.")
            else {
                alwaysOnPrefs.edit()
                    .putString(AlwaysOnKoochikService.PREF_STT_MODEL_ID, stt.id)
                    .putString(AlwaysOnKoochikService.PREF_TTS_MODEL_ID, tts?.id)
                    .apply()
                if (alwaysOnEnabled && alwaysOnPrefs.getBoolean(AlwaysOnKoochikService.PREF_ENABLED, false)) {
                    updateVoiceStatus("شنود دائمی از قبل روشن است؛ برای شروع مکالمه فقط «دینا» را بگو. مدل شنوا دوباره بارگذاری نمی‌شود.")
                } else {
                    busy = true
                    controller.start(stt, tts)
                }
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
 * Only a final/answer channel may reach the transcript UI or TTS.
 * Raw Kotlin strings keep regex escapes exact (not double-escaped into literal backslashes).
 */
internal fun sanitizeAssistantVoiceText(raw: String): String {
    val channelMarker = Regex("""(?i)<\|channel\|?>\s*(analysis|thought|reasoning|final|answer|commentary)\b""")
    val channels = channelMarker.findAll(raw).toList()
    var source = raw
    if (channels.isNotEmpty()) {
        val finalChannel = channels.lastOrNull {
            it.groupValues[1].equals("final", true) || it.groupValues[1].equals("answer", true)
        } ?: return ""
        source = raw.substring(finalChannel.range.last + 1)
        source = source.substring(0, channelMarker.find(source)?.range?.first ?: source.length)
    }
    return source
        .replace(Regex("""(?is)<think\b(?!\s*/)[^>]*>.*?(?:</think\s*>|<think\s*/\s*>|$)"""), " ")
        .replace(Regex("""(?is)<think\s*/\s*>"""), " ")
        .replace(Regex("""(?is)</?think\b[^>]*>"""), " ")
        .replace(Regex("""(?is)<(analysis|reasoning|scratchpad)\b[^>]*>.*?</\1\s*>"""), " ")
        .replace(Regex("""(?is)<\|(?:turn|end_of_turn|endofturn|im_end|fim_suffix)\|?>.*$"""), " ")
        .replace(Regex("""(?m)^\s*`{3,}[^\n]*"""), " ")
        .replace(Regex("""(?m)^\s*#{1,6}\s*"""), "")
        .replace(Regex("""(?m)^\s*(?:[-*_]\s*){3,}$"""), " ")
        .replace(Regex("""(?m)^\s*[-*+]\s+"""), "")
        .replace(Regex("""(?m)^\s*\d+[.)]\s+"""), "")
        .replace(Regex("""(?<!\w)(?:\*\*|__)(?=\S)|(?<=\S)(?:\*\*|__)"""), "")
        .replace(Regex("""(?<!\w)[*_~](?=\S)|(?<=\S)[*_~](?!\w)"""), "")
        .replace(Regex("""(?is)<[^>]+>"""), " ")
        .replace(Regex("""[ \t]+"""), " ")
        .replace(Regex(""" *\n *"""), "\n")
        .replace(Regex("""\n{3,}"""), "\n\n")
        .trim()
}

internal fun shouldInterruptForVoiceBargeIn(
    turnRunning: Boolean,
    assistantSpeaking: Boolean,
    speech: Boolean,
    rms: Float,
    threshold: Float,
    aecActive: Boolean,
): Boolean = turnRunning && assistantSpeaking && speech && rms > threshold && aecActive

internal class VoiceConversationController(
    private val context: Context,
    private val container: AppContainer,
    private val decisionEngine: LayaDecisionEngine,
    private val onStatus: (String) -> Unit,
    private val onDiagnostic: (String) -> Unit,
    private val onListening: (Boolean) -> Unit,
    private val onLine: (String, String) -> Unit,
    private val onWakeShutdown: (LocalAsr?) -> Unit = {},
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
    private val lastFinalizedUtterance = AtomicReference("")
    private val lastFinalizedUtteranceAtMs = AtomicLong(0L)
    private val startInProgress = AtomicBoolean(false)
    private val lifecycleEpoch = AtomicLong(0L)
    @Volatile private var microphone: LocalMicrophone? = null
    private var recognizer: LocalAsr? = null
    private var tts: com.samanramezani.aichattest.voice.LocalTtsEngine? = null
    private var frameJob: Job? = null
    private var turnJob: Job? = null
    private var speechJob: Job? = null
    private var conversationId: String? = null
    @Volatile private var lastLoggedIntentModelError: String? = null
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
        ttsModel: VoiceModelEntry?,
        initialUtterance: String? = null,
        acknowledgeWake: Boolean = false,
        speechStartGate: CompletableDeferred<Unit>? = null,
        preloadedAsr: LocalAsr? = null,
    ) {
        if (closed || microphone != null || !startInProgress.compareAndSet(false, true)) {
            if (preloadedAsr != null) runCatching { preloadedAsr.close() }
            return
        }
        val startEpoch = lifecycleEpoch.get()
        scope.launch {
            try {
                onStatus("در حال آماده‌سازی موتورهای محلی صوت…")
                val engines = withContext(Dispatchers.IO) {
                    // Reuse the always-on listener's loaded Shenava instance after wake detection.
                    // Manual conversation starts still create their own recognizer as before.
                    val asr = preloadedAsr ?: when (stt.asrMode) {
                        "qwen3-asr" -> LocalQwen3Asr(stt)
                        "nemo-ctc" -> LocalNemoCtcAsr(stt)
                        else -> LocalStreamingAsr(stt)
                    }
                    try {
                        val usePocket = context.getSharedPreferences(AlwaysOnKoochikService.PREFS, Context.MODE_PRIVATE)
                            .getBoolean("pocket_tts_selected", false)
                        val ttsEngine = if (usePocket) {
                            val modelDir = File(context.filesDir, "voice-models/pocket-tts-farsi-v2")
                            check(PocketTtsFarsiBundleStore(context.contentResolver, modelDir).isReady()) {
                                "POCKET-TTS-SELECT-002: بستهٔ Pocket TTS نصب نیست یا ناقص است."
                            }
                            com.samanramezani.aichattest.voice.LocalPocketTts(modelDir)
                        } else LocalPiperTts(requireNotNull(ttsModel) { "VOICE-TTS-001: مدل Piper انتخاب نشده است." })
                        asr to ttsEngine
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
                if (conversationId == null) {
                    conversationId = container.conversationHistory.create("مکالمه صوتی").id
                    onDiagnostic("VOICE-SESSION-START: conversationId=$conversationId; one session is retained across turns until «دینا خاموش شو».")
                }
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
                    // Never cancel LLM generation because of ambient noise. Barge-in is only
                    // meaningful while assistant audio is actually playing, and only when Android
                    // confirms AEC is active so speaker leakage is less likely to be mistaken for speech.
                    val candidate = shouldInterruptForVoiceBargeIn(
                        turnRunning = turnRunning.get(),
                        assistantSpeaking = speaking.get(),
                        speech = frame.speech,
                        rms = frame.rms,
                        threshold = threshold,
                        aecActive = aecActive,
                    )
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
                                val finalizedAtMs = SystemClock.elapsedRealtime()
                                val previousFinalized = lastFinalizedUtterance.get()
                                if (utterance.equals(previousFinalized, ignoreCase = true) &&
                                    finalizedAtMs - lastFinalizedUtteranceAtMs.get() in 0L..1800L
                                ) {
                                    onDiagnostic("VOICE-STT-DUPLICATE: duplicate finalized utterance suppressed; chars=${utterance.length}")
                                    continue
                                }
                                lastFinalizedUtterance.set(utterance)
                                lastFinalizedUtteranceAtMs.set(finalizedAtMs)
                                // In an active dialogue, «دینا خاموش شو» returns control to the
                                // always-on wake listener instead of being sent to the chat model.
                                val wakeControl = extractDinaCommand(utterance)
                                if (wakeControl != null && isDinaSleepCommand(wakeControl)) {
                                    onDiagnostic("VOICE-SESSION-END: conversationId=$conversationId; reason=DINA_SLEEP; next wake starts a fresh conversation.")
                                    onStatus("فرمان «دینا خاموش شو» دریافت شد؛ مکالمه متوقف و شنود بیدارباش دوباره فعال می‌شود.")
                                    // Transfer ownership back to the always-on service so the same
                                    // Shenava instance remains loaded across conversation boundaries.
                                    val loadedAsr = recognizer
                                    recognizer = null
                                    onWakeShutdown(loadedAsr)
                                    return@launch
                                }
                                if (turnRunning.compareAndSet(false, true)) {
                                    bargeInTriggered.set(false)
                                    bargeInCapturePending.set(false)
                                    onLine("شما", utterance)
                                    turnJob = scope.launch { processTurn(utterance) }
                                } else {
                                    // The newest finalized user turn supersedes stale generation immediately.
                                    // Keep only the latest utterance to prevent a long decode from building
                                    // a backlog and answering obsolete requests out of order.
                                    pendingUtterances.clear()
                                    pendingUtterances.offer(utterance)
                                    onLine("شما", utterance)
                                    onDiagnostic("VOICE-TURN-INTERRUPT: turnRunning=true; pending=1; previousGeneration=stop_requested")
                                    interrupt("گفتار تازه دریافت شد؛ پاسخ قبلی متوقف و درخواست جدید پردازش می‌شود.")
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
                        processTurn(
                            initial,
                            acknowledgeWake = acknowledgeWake,
                            speechStartGate = speechStartGate,
                        )
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
            onStatus("در حال بررسی درخواست فارسی روی دستگاه…")
            val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
            val layaDecision = runCatching {
                decisionEngine.classify(
                    utterance,
                    audioManager.getStreamVolume(AudioManager.STREAM_MUSIC),
                    audioManager.getStreamMaxVolume(AudioManager.STREAM_MUSIC),
                )
            }.getOrNull()
            decisionEngine.lastError?.let { modelError ->
                if (lastLoggedIntentModelError != modelError) {
                    lastLoggedIntentModelError = modelError
                    onDiagnostic(modelError + "؛ هیچ فرمانی بدون تصمیم معتبر Laya اجرا نمی‌شود و متن اصلی به مدل گفت‌وگو می‌رود.")
                }
            }
            layaDecision?.trace?.forEach { onDiagnostic("LAYA-DECISION: $it") }
            // Laya is the sole decision-maker for local voice actions.
            // Never infer an intent or extract a number with a parallel rule/regex parser.
            // If Laya is unavailable or abstains, forward the exact original transcript to chat.
            val command = layaDecision?.command
                ?: com.samanramezani.aichattest.voice.DinaCommand(
                    com.samanramezani.aichattest.voice.DinaIntent.CHAT
                )
            if (command.intent != com.samanramezani.aichattest.voice.DinaIntent.CHAT) {
                val localReply = runCatching { DinaDeviceCommandExecutor(context).execute(command) }
                    .getOrElse { "انجام فرمان ناموفق بود: ${it.message ?: it.javaClass.simpleName}" }
                    .orEmpty()
                if (localReply.isNotBlank()) {
                    onDiagnostic("DINA-LOCAL-COMMAND: intent=${command.intent.name} parameter=${command.value ?: "none"} classifier=${if (layaDecision != null && layaDecision.command.intent != com.samanramezani.aichattest.voice.DinaIntent.CHAT) "laya-multilingual-int4" else "safe-rules"} model=local-device-executor")
                    queue.trySend(localReply)
                    queue.close()
                    activeSpeechJob.join()
                    if (!turnCancel.get()) {
                        persistAssistantReplyIfMissing(localReply)
                        onLine("دستیار", localReply)
                        onStatus("فرمان محلی اجرا شد؛ مدل گفت‌وگو فراخوانی نشد.")
                    }
                    return
                }
            }
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
                persistAssistantReplyIfMissing(generation.text)
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
                    val raw = generation.text
                    val rawHasFinalChannel = raw.contains("<|channel|>final", ignoreCase = true) ||
                        raw.contains("<|channel|>answer", ignoreCase = true)
                    val rawHasReasoningMarker = raw.contains("<think", ignoreCase = true) ||
                        raw.contains("<|channel|>analysis", ignoreCase = true) ||
                        raw.contains("<|channel|>reasoning", ignoreCase = true) ||
                        raw.contains("<|channel|>thought", ignoreCase = true)
                    val detail = failure ?: "پس از پاک‌سازی، متن قابل‌گفتار باقی نماند."
                    onDiagnostic(
                        "VOICE-LLM-002: empty_response rawChars=${raw.length} " +
                            "outputTokens=${generation.outputTokens ?: "n/a"} " +
                            "generationMs=${generation.generationTimeMs ?: "n/a"} " +
                            "stopped=${generation.stopped} finalChannel=$rawHasFinalChannel " +
                            "reasoningMarker=$rawHasReasoningMarker detail=$detail"
                    )
                    onStatus(if (failure != null) "VOICE-LLM-001: $failure" else "VOICE-LLM-002: مدل متن نهایی قابل‌گفتار تولید نکرد؛ جزئیات کانال خروجی ثبت شد.")
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

    /**
     * AgentSession normally persists the final answer. Verify that invariant before showing
     * it in the live transcript; also covers local-device replies, which bypass AgentSession.
     * The last-message check prevents duplicate rows when the normal persistence path succeeded.
     */
    private suspend fun persistAssistantReplyIfMissing(content: String) {
        val id = conversationId ?: error("VOICE-CHAT-001: شناسه مکالمه ایجاد نشده است.")
        if (content.isBlank()) return
        val last = container.conversationHistory.messages(id).lastOrNull()
        if (last?.role == ConversationMessage.Role.ASSISTANT && last.content == content) return
        val now = System.currentTimeMillis()
        val saved = container.conversationHistory.append(
            id,
            ConversationMessage(
                id = java.util.UUID.randomUUID().toString(),
                role = ConversationMessage.Role.ASSISTANT,
                content = content,
                timestampEpochMs = now,
            ),
            now,
        )
        check(saved) { "VOICE-HISTORY-001: پاسخ دستیار در تاریخچهٔ گفتگو ذخیره نشد." }
        onDiagnostic("VOICE-HISTORY-SAVED: conversationId=$id; chars=${content.length}")
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
        speechJob?.cancel()
        // Signal native cancellation before cancelling the coroutine waiting on native decode.
        // This lets the runtime release its generation mutex before the next turn starts.
        val interruptedJob = turnJob
        scope.launch {
            runCatching { container.modelRuntime.stopGeneration() }
            interruptedJob?.cancel(CancellationException("barge-in"))
            if (turnJob === interruptedJob) turnJob = null
        }
        // processTurn.finally releases the active turn and dispatches the newest pending utterance.
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
        stop().invokeOnCompletion {
            decisionEngine.close()
            scope.cancel()
        }
    }
}
