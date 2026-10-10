package com.samanramezani.aichattest.ui.voice

import android.content.Context
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.*
import androidx.compose.ui.platform.LocalContext
import com.samanramezani.aichattest.AppContainer
import com.samanramezani.aichattest.ui.voice.VoiceConversationPage
import com.samanramezani.aichattest.voice.LocalMicrophone
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

@Composable
internal fun VoiceConversationHost(container: AppContainer, onBack: () -> Unit) {
    val context = LocalContext.current
    val uiScope = rememberCoroutineScope()
    val store = remember(context) { VoiceModelStore(context.contentResolver, File(context.filesDir, "voice-models")) }
    var sttModels by remember { mutableStateOf(store.list(VoiceModelKind.STT)) }
    var ttsModels by remember { mutableStateOf(store.list(VoiceModelKind.TTS)) }
    var activeSttId by remember { mutableStateOf(sttModels.firstOrNull()?.id) }
    var activeTtsId by remember { mutableStateOf(ttsModels.firstOrNull()?.id) }
    var lines by remember { mutableStateOf(emptyList<VoiceLine>()) }
    var status by remember { mutableStateOf("برای شروع، مدل‌های محلی STT و TTS را انتخاب یا وارد کنید.") }
    var diagnosticLogs by remember { mutableStateOf(listOf("[${System.currentTimeMillis()}] صفحه مکالمه صوتی باز شد.")) }
    fun updateVoiceStatus(value: String) {
        status = value
        diagnosticLogs = (diagnosticLogs + "[${System.currentTimeMillis()}] $value").takeLast(200)
    }
    var listening by remember { mutableStateOf(false) }
    var busy by remember { mutableStateOf(false) }

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
    DisposableEffect(controller) { onDispose { controller.close() } }

    fun refreshModels() {
        sttModels = store.list(VoiceModelKind.STT)
        ttsModels = store.list(VoiceModelKind.TTS)
        if (activeSttId !in sttModels.map { it.id }) activeSttId = sttModels.firstOrNull()?.id
        if (activeTtsId !in ttsModels.map { it.id }) activeTtsId = ttsModels.firstOrNull()?.id
    }

    fun importSmallSttComponent(uri: Uri, component: SmallSttComponent, label: String) {
        uiScope.launch {
            busy = true
            updateVoiceStatus("در حال وارد کردن ${label} مدل فارسی کوچک…")
            try {
                val ready = withContext(Dispatchers.IO) { store.importSmallPersianSttComponent(uri, component) }
                refreshModels()
                if (ready != null) activeSttId = ready.id
                val missing = if (ready == null) store.missingSmallSttComponents() else emptyList()
                updateVoiceStatus(if (ready != null) "مدل کوچک Shenava Rizeh-Pizeh آماده و انتخاب شد؛ تشخیص نهایی پس از مکث کوتاه گفتار انجام می‌شود."
                    else "فایل ${label} ذخیره شد؛ برای تکمیل مدل Shenava هنوز وارد کنید: ${missing.joinToString(" و ")}.")
            } catch (t: Throwable) {
                updateVoiceStatus("VOICE-STT-IMPORT: ${t.message ?: t.javaClass.simpleName}")
            } finally { busy = false }
        }
    }

    val smallSttModelPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) importSmallSttComponent(uri, SmallSttComponent.MODEL, "model.onnx")
    }
    val smallSttTokensPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) importSmallSttComponent(uri, SmallSttComponent.TOKENS, "tokens.txt")
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
                val ready = withContext(Dispatchers.IO) { store.importPiperComponent(uri, component) }
                refreshModels()
                if (ready != null) activeTtsId = ready.id
                val missing = if (ready == null) store.missingPiperComponents() else emptyList()
                updateVoiceStatus(if (ready != null) "مدل گفتار ${ready.title} آماده و انتخاب شد."
                    else "فایل ${label} ذخیره شد؛ برای تکمیل مدل Piper هنوز لازم است: ${missing.joinToString(" و ")}.")
            } catch (t: Throwable) {
                updateVoiceStatus("VOICE-PIPER-IMPORT: ${t.message ?: t.javaClass.simpleName}")
            } finally { busy = false }
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
        activeSttId = activeSttId,
        activeTtsId = activeTtsId,
        onBack = onBack,
        onImportStt = { sttPicker.launch(arrayOf("application/zip", "application/x-zip-compressed", "application/x-bzip2", "application/octet-stream")) },
        onImportSmallSttModel = { smallSttModelPicker.launch(arrayOf("application/onnx", "application/octet-stream", "*/*")) },
        onImportSmallSttTokens = { smallSttTokensPicker.launch(arrayOf("text/plain", "application/octet-stream", "*/*")) },
        onImportTts = { ttsPicker.launch(arrayOf("application/zip", "application/x-zip-compressed", "application/x-bzip2", "application/octet-stream")) },
        onImportTtsModel = { ttsModelPicker.launch(arrayOf("application/onnx", "application/octet-stream", "*/*")) },
        onImportTtsConfig = { ttsConfigPicker.launch(arrayOf("application/json", "text/json", "text/plain", "*/*")) },
        onImportEspeakData = { espeakPicker.launch(arrayOf("application/x-bzip2", "application/zip", "application/octet-stream", "*/*")) },
        onSelectStt = { activeSttId = it },
        onSelectTts = { activeTtsId = it },
        onStart = {
            val stt = sttModels.firstOrNull { it.id == activeSttId }
            val tts = ttsModels.firstOrNull { it.id == activeTtsId }
            if (stt == null || tts == null) updateVoiceStatus("VOICE-MODEL-001: ابتدا مدل‌های STT و TTS را وارد و انتخاب کنید.")
            else {
                busy = true
                controller.start(stt, tts)
            }
        },
        onStop = { controller.stop() },
        onInterrupt = { controller.interrupt("گفتار مدل با درخواست کاربر قطع شد.") },
    )
}

private class VoiceConversationController(
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
    private var microphone: LocalMicrophone? = null
    private var recognizer: LocalAsr? = null
    private var tts: LocalPiperTts? = null
    private var frameJob: Job? = null
    private var turnJob: Job? = null
    private var speechJob: Job? = null
    private var conversationId: String? = null
    private var closed = false

    fun start(stt: VoiceModelEntry, ttsModel: VoiceModelEntry) {
        if (microphone != null || closed) return
        scope.launch {
            try {
                onStatus("در حال آماده‌سازی موتورهای محلی صوت…")
                val engines = withContext(Dispatchers.IO) {
                    (when (stt.asrMode) {
                        "qwen3-asr" -> LocalQwen3Asr(stt)
                        "nemo-ctc" -> LocalNemoCtcAsr(stt)
                        else -> LocalStreamingAsr(stt)
                    }) to LocalPiperTts(ttsModel)
                }
                recognizer = engines.first
                tts = engines.second
                if (conversationId == null) conversationId = container.conversationHistory.create("مکالمه صوتی").id
                val frames = Channel<LocalMicrophone.Frame>(capacity = 64)
                val mic = LocalMicrophone()
                val loudSpeechFrames = AtomicInteger(0)
                microphone = mic
                mic.start { frame ->
                    // Run barge-in detection on the capture thread, not the ASR consumer.
                    // Offline ASR may need hundreds of milliseconds to decode an utterance.
                    if (speaking.get() && frame.rms > 0.035f && frame.speech) {
                        if (loudSpeechFrames.incrementAndGet() >= 3) {
                            interrupt("صدای کاربر تشخیص داده شد؛ پاسخ قبلی متوقف شد.")
                            loudSpeechFrames.set(0)
                        }
                    } else {
                        loudSpeechFrames.set(0)
                    }
                    frames.trySend(frame)
                }
                frameJob = scope.launch {
                    onListening(true)
                    onStatus("در حال گوش‌دادن؛ گفتار و پردازش کاملاً روی دستگاه است.")
                    for (frame in frames) {
                        val update = try { recognizer?.accept(frame.samples, 16000, frame.speech) } catch (t: Throwable) {
                            onDiagnostic("VOICE-STT-002\n${t.stackTraceToString()}")
                            onStatus("VOICE-STT-002: خطا در تشخیص گفتار: ${t.message ?: t.javaClass.simpleName}")
                            null
                        } ?: continue
                        if (update.text.isNotBlank()) onStatus("در حال شنیدن: ${update.text}")
                        if (update.endpoint && update.text.isNotBlank()) {
                            val utterance = update.text.trim()
                            if (utterance.isNotBlank() && turnRunning.compareAndSet(false, true)) {
                                onLine("شما", utterance)
                                turnJob = scope.launch { processTurn(utterance) }
                            }
                        }
                    }
                }
            } catch (t: Throwable) {
                onDiagnostic("VOICE-START-001\n${t.stackTraceToString()}")
                onStatus("VOICE-START-001: شروع مکالمه ناموفق بود: ${t.message ?: t.javaClass.simpleName}")
                cleanupEngines()
                onListening(false)
            }
        }
    }

    private suspend fun processTurn(utterance: String) {
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
                    val engine = tts ?: break
                    speaking.set(true)
                    try {
                        engine.speak(segment, turnCancel)
                    } finally {
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
            val session = container.createAgentSession(id) { event ->
                if (event is AgentEvent.Token && !turnCancel.get()) enqueueCompleteSentences(event.value)
                if (event is AgentEvent.Failed) { onDiagnostic("VOICE-LLM-001: ${event.message}"); onStatus("VOICE-LLM-001: ${event.message}") }
            } ?: error("VOICE-CHAT-002: نشست مدل محلی در دسترس نیست.")
            val result = session.send(utterance, InferenceSettings(maxNewTokens = 1024), requestedRecentMessages = Int.MAX_VALUE)
            if (!turnCancel.get()) {
                enqueueCompleteSentences("", flush = true)
                queue.close()
                activeSpeechJob.join()
                if (result.generation.text.isNotBlank()) onLine("دستیار", result.generation.text)
                onStatus(if (speaking.get()) "در حال پخش پاسخ…" else "پاسخ آماده است؛ می‌توانید صحبت کنید.")
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
            if (speechJob === activeSpeechJob) speechJob = null
            turnRunning.set(false)
        }
    }

    fun interrupt(message: String) {
        cancelledSpeech.get()?.set(true)
        speaking.set(false)
        scope.launch { container.modelManager?.stopGeneration() }
        turnJob?.cancel(CancellationException("barge-in"))
        turnJob = null
        speechJob?.cancel()
        turnRunning.set(false)
        onStatus(message)
    }

    fun stop(): Job {
        val oldFrameJob = frameJob
        frameJob = null
        oldFrameJob?.cancel()
        val oldTurnJob = turnJob
        turnJob = null
        oldTurnJob?.cancel()
        val oldSpeechJob = speechJob
        oldSpeechJob?.cancel()
        cancelledSpeech.getAndSet(null)?.set(true)
        speaking.set(false)
        val oldMicrophone = microphone
        microphone = null
        oldMicrophone?.stop()
        return scope.launch {
            listOfNotNull(oldFrameJob, oldTurnJob, oldSpeechJob).joinAll()
            cleanupEngines()
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
