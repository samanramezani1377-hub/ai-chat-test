package com.samanramezani.aichattest.ui.voice

import android.content.Context
import android.net.Uri
import android.os.SystemClock
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
import com.samanramezani.aichattest.voice.SmallSttModel
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
import java.util.concurrent.atomic.AtomicLong

@Composable
internal fun VoiceConversationHost(container: AppContainer, onBack: () -> Unit) {
    val context = LocalContext.current
    val uiScope = rememberCoroutineScope()
    val store = remember(context) { VoiceModelStore(context.contentResolver, File(context.filesDir, "voice-models")) }
    var sttModels by remember { mutableStateOf(store.list(VoiceModelKind.STT)) }
    var ttsModels by remember { mutableStateOf(store.list(VoiceModelKind.TTS)) }
    var missingSttComponents by remember { mutableStateOf(store.missingSmallSttComponents()) }
    var missingKoochikComponents by remember { mutableStateOf(store.missingSmallSttComponents(SmallSttModel.KOOCHIK)) }
    var missingTtsComponents by remember { mutableStateOf(store.missingPiperComponents()) }
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

    fun refreshImportProgress() {
        missingSttComponents = store.missingSmallSttComponents()
        missingKoochikComponents = store.missingSmallSttComponents(SmallSttModel.KOOCHIK)
        missingTtsComponents = store.missingPiperComponents()
    }

    fun importSmallSttComponent(uri: Uri, component: SmallSttComponent, label: String, modelType: SmallSttModel = SmallSttModel.RIZEH_PIZEH) {
        uiScope.launch {
            busy = true
            updateVoiceStatus("در حال وارد کردن ${label} مدل فارسی کوچک…")
            try {
                val ready = withContext(Dispatchers.IO) { store.importSmallPersianSttComponent(uri, component, modelType) }
                refreshModels()
                refreshImportProgress()
                if (ready != null) activeSttId = ready.id
                val missing = if (ready == null) store.missingSmallSttComponents(modelType) else emptyList()
                val modelName = if (modelType == SmallSttModel.KOOCHIK) "Shenava Koochik" else "Shenava Rizeh-Pizeh"
                updateVoiceStatus(if (ready != null) "مدل $modelName آماده و انتخاب شد؛ تشخیص نهایی پس از مکث کوتاه گفتار انجام می‌شود."
                    else "فایل ${label} مدل $modelName ذخیره شد؛ برای تکمیل آن هنوز وارد کنید: ${missing.joinToString(" و ")}.")
            } catch (t: Throwable) {
                updateVoiceStatus("VOICE-STT-IMPORT: ${t.message ?: t.javaClass.simpleName}")
            } finally { refreshImportProgress(); busy = false }
        }
    }

    val smallSttModelPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) importSmallSttComponent(uri, SmallSttComponent.MODEL, "model.onnx")
    }
    val smallSttTokensPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) importSmallSttComponent(uri, SmallSttComponent.TOKENS, "tokens.txt")
    }
    val koochikModelPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) importSmallSttComponent(uri, SmallSttComponent.MODEL, "model.onnx", SmallSttModel.KOOCHIK)
    }
    val koochikTokensPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) importSmallSttComponent(uri, SmallSttComponent.TOKENS, "tokens.txt", SmallSttModel.KOOCHIK)
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
        missingSttComponents = missingSttComponents,
        missingKoochikComponents = missingKoochikComponents,
        missingTtsComponents = missingTtsComponents,
        activeSttId = activeSttId,
        activeTtsId = activeTtsId,
        onBack = onBack,
        onImportStt = { sttPicker.launch(arrayOf("application/zip", "application/x-zip-compressed", "application/x-bzip2", "application/octet-stream")) },
        onImportSmallSttModel = { smallSttModelPicker.launch(arrayOf("application/onnx", "application/octet-stream", "*/*")) },
        onImportSmallSttTokens = { smallSttTokensPicker.launch(arrayOf("text/plain", "application/octet-stream", "*/*")) },
        onImportKoochikModel = { koochikModelPicker.launch(arrayOf("application/onnx", "application/octet-stream", "*/*")) },
        onImportKoochikTokens = { koochikTokensPicker.launch(arrayOf("text/plain", "application/octet-stream", "*/*")) },
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
        // Some local model outputs leak one or two Latin `n` characters directly before Persian text (e.g. `nnسلام`).\n        // Drop only this narrow leading artifact; preserve legitimate Latin text elsewhere.\n        .replace(Regex("(?i)^\\s*n{1,2}(?=[\\u0600-\\u06FF])"), "")\n        .replace(Regex("[ \\t]+"), " ")
        .replace(Regex(" *\\n *"), "\\n")
        .replace(Regex("\\n{3,}"), "\\n\\n")
        .trim()
    // Also discard orphaned reasoning markers from malformed model output.
    val marker = Regex("(?i)<think\\s*/?>")
    while (marker.containsMatchIn(text)) {
        val match = marker.find(text) ?: break
        text = text.removeRange(match.range).trimStart()
    }
    return text
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
    private val bargeInTriggered = AtomicBoolean(false)
    // Audio from the loudspeaker can leak into the mic briefly after AudioTrack stops.
    private val lastTtsPlaybackAtMs = AtomicLong(0L)
    private val pendingUtterance = AtomicReference<String?>(null)
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
                // 256 x 40 ms frames gives the ASR consumer more room during offline decoding.
                val frames = Channel<LocalMicrophone.Frame>(capacity = 256)
                val mic = LocalMicrophone()
                val loudSpeechFrames = AtomicInteger(0)
                val audioDropReported = AtomicBoolean(false)
                microphone = mic
                mic.start { frame ->
                    // Run barge-in detection on the capture thread, not the ASR consumer.
                    // Offline ASR may need hundreds of milliseconds to decode an utterance.
                    // Barge-in must also work while the LLM is generating text, before TTS starts.
                    // Never interpret the assistant's own loudspeaker output as barge-in.
                    // Keep a short tail guard because acoustic echo can remain in captured frames
                    // after AudioTrack stops. Echo cancellation is device-dependent, so RMS/VAD
                    // alone must not be allowed to interrupt active TTS.
                    val nowMs = SystemClock.elapsedRealtime()
                    val assistantAudioActive = speaking.get() || nowMs - lastTtsPlaybackAtMs.get() < 700L
                    if (turnRunning.get() && !assistantAudioActive && frame.rms > 0.035f && frame.speech) {
                        if (loudSpeechFrames.incrementAndGet() >= 3 && bargeInTriggered.compareAndSet(false, true)) {
                            interrupt("صدای کاربر تشخیص داده شد؛ تولید یا پخش پاسخ قبلی متوقف شد.")
                            loudSpeechFrames.set(0)
                        }
                    } else {
                        loudSpeechFrames.set(0)
                    }
                    if (frames.trySend(frame).isFailure && audioDropReported.compareAndSet(false, true)) {
                        scope.launch {
                            onDiagnostic("VOICE-AUDIO-001: صف دریافت صدا پر شد؛ بخشی از فریم‌های میکروفون ممکن است از دست رفته باشد.")
                            onStatus("VOICE-AUDIO-001: پردازش گفتار عقب افتاده است؛ برای جلوگیری از تشخیص ناقص، مکالمه را دوباره شروع کنید.")
                        }
                    }
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
                            if (utterance.isNotBlank()) {
                                if (turnRunning.compareAndSet(false, true)) {
                                    bargeInTriggered.set(false)
                                    onLine("شما", utterance)
                                    turnJob = scope.launch { processTurn(utterance) }
                                } else {
                                    // Keep the latest finalized utterance instead of silently dropping it
                                    // while the previous generation is being cancelled.
                                    pendingUtterance.set(utterance)
                                    onLine("شما", utterance)
                                    onStatus("گفتار جدید دریافت شد؛ پس از توقف امن پاسخ قبلی پردازش می‌شود.")
                                }
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
                    try {
                        // Mark speaking only when audio playback actually starts, not during synthesis.
                        engine.speak(segment, turnCancel) {
                            lastTtsPlaybackAtMs.set(SystemClock.elapsedRealtime())
                            speaking.set(true)
                        }
                    } finally {
                        speaking.set(false)
                        lastTtsPlaybackAtMs.set(SystemClock.elapsedRealtime())
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
            val pending = pendingUtterance.getAndSet(null)
            if (!closed && pending != null && turnRunning.compareAndSet(false, true)) {
                onStatus("در حال پردازش گفتار جدید…")
                turnJob = scope.launch { processTurn(pending) }
            }
        }
    }

    fun interrupt(message: String) {
        cancelledSpeech.get()?.set(true)
        speaking.set(false)
        scope.launch { container.modelManager?.stopGeneration() }
        turnJob?.cancel(CancellationException("barge-in"))
        turnJob = null
        speechJob?.cancel()
        // Let processTurn.finally release the active turn and dispatch any newly finalized utterance.
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
