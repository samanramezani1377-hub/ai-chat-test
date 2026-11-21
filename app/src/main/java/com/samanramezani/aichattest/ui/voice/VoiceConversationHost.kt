package com.samanramezani.aichattest.ui.voice

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.app.Activity
import android.net.Uri
import android.os.Build
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.*
import androidx.compose.ui.platform.LocalContext
import com.samanramezani.aichattest.AppContainer
import com.samanramezani.aichattest.AlwaysOnKoochikService
import com.samanramezani.aichattest.ui.voice.VoiceConversationPage
import com.samanramezani.aichattest.voice.VoiceModelKind
import com.samanramezani.aichattest.voice.VoiceModelStore
import com.samanramezani.aichattest.voice.PiperComponent
import com.samanramezani.aichattest.voice.SmallSttComponent
import com.samanramezani.aichattest.voice.LayaDecisionModelStore
import com.samanramezani.aichattest.voice.LayaDecisionEngine
import kotlinx.coroutines.CancellationException
import androidx.core.content.ContextCompat
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean

@Composable
internal fun VoiceConversationHost(
    container: AppContainer,
    onBack: () -> Unit,
    settingsOnly: Boolean = false,
    onHistoryChanged: () -> Unit = {},
    onOpenSettings: () -> Unit = {},
) {
    val context = LocalContext.current
    val uiScope = rememberCoroutineScope()
    val store = remember(context) { VoiceModelStore(context.contentResolver, File(context.filesDir, "voice-models")) }
    val layaStore = remember(context) {
        LayaDecisionModelStore(context.contentResolver, File(context.filesDir, "voice-models/laya-int4"))
    }
    var layaReady by remember { mutableStateOf(layaStore.isReady()) }
    var layaStatus by remember { mutableStateOf(layaStore.status()) }
    var sttModels by remember { mutableStateOf(store.list(VoiceModelKind.STT)) }
    var ttsModels by remember { mutableStateOf(store.list(VoiceModelKind.TTS)) }
    var missingKoochikComponents by remember { mutableStateOf(store.missingKoochikComponents()) }
    val alwaysOnPrefs = remember(context) { context.getSharedPreferences(AlwaysOnKoochikService.PREFS, Context.MODE_PRIVATE) }
    var alwaysOnEnabled by remember { mutableStateOf(alwaysOnPrefs.getBoolean(AlwaysOnKoochikService.PREF_ENABLED, false)) }
    var missingTtsComponents by remember { mutableStateOf(store.missingPiperComponents()) }
    var activeSttId by remember { mutableStateOf(sttModels.firstOrNull { it.id.startsWith("stt-shenava-koochik-") }?.id ?: sttModels.firstOrNull()?.id) }
    var activeTtsId by remember { mutableStateOf(ttsModels.firstOrNull()?.id) }
    var lines by remember { mutableStateOf(emptyList<VoiceLine>()) }
    var streamingAssistantLineId by remember { mutableStateOf<Long?>(null) }
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
                check(stt != null && tts != null) {
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
                if (text.isNotBlank()) {
                    if (speaker == "شما") streamingAssistantLineId = null
                    val partialId = if (speaker == "دستیار") streamingAssistantLineId else null
                    if (partialId != null) {
                        lines = lines.map { if (it.id == partialId) it.copy(text = text) else it }
                        streamingAssistantLineId = null
                    } else {
                        lines = lines + VoiceLine(System.nanoTime(), speaker, text)
                    }
                    if (speaker == "دستیار") onHistoryChanged()
                }
            } },
            onPartialAssistant = { text -> uiScope.launch(Dispatchers.Main.immediate) {
                if (text.isNotBlank()) {
                    val id = streamingAssistantLineId ?: System.nanoTime().also { streamingAssistantLineId = it }
                    lines = if (lines.any { it.id == id }) {
                        lines.map { if (it.id == id) it.copy(text = text) else it }
                    } else {
                        lines + VoiceLine(id, "دستیار", text)
                    }
                }
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
                if (ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
                    updateVoiceStatus("VOICE-ALWAYS-002: دینا تشخیص داده شد، اما مجوز میکروفون برای شروع مکالمه در دسترس نیست.")
                } else if (stt == null || ttsModel == null) {
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
            } catch (cancelled: CancellationException) {
                throw cancelled
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
            } catch (cancelled: CancellationException) {
                throw cancelled
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
            } catch (cancelled: CancellationException) {
                throw cancelled
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
            } catch (cancelled: CancellationException) {
                throw cancelled
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
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (t: Throwable) {
                layaStatus = "LAYA-IMPORT: ${t.message ?: t.javaClass.simpleName}"
                updateVoiceStatus(layaStatus)
            } finally { busy = false }
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
            } catch (cancelled: CancellationException) {
                downloadFinished.set(true)
                throw cancelled
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
        settingsOnly = settingsOnly,
        onOpenSettings = onOpenSettings,
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
                if (enabled && stt == null) {
                    updateVoiceStatus("VOICE-ALWAYS-003: برای شنود دائمی باید مدل Shenava Koochik فارسی را وارد کنی.")
                } else if (enabled && tts == null) {
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
            if (stt == null || tts == null) updateVoiceStatus("VOICE-MODEL-001: ابتدا مدل‌های STT و TTS را وارد و انتخاب کنید.")
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

