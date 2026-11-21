package com.samanramezani.aichattest.ui.voice

import android.content.Context
import android.media.AudioManager
import android.os.SystemClock
import androidx.compose.runtime.*
import com.samanramezani.aichattest.AppContainer
import com.samanramezani.aichattest.voice.LocalMicrophone
import com.samanramezani.aichattest.voice.VoiceAudioInputGuard
import com.samanramezani.aichattest.voice.LocalPiperTts
import com.samanramezani.aichattest.voice.SpeechSegmenter
import com.samanramezani.aichattest.voice.matchedSpeechPrefixLength
import com.samanramezani.aichattest.voice.LocalStreamingAsr
import com.samanramezani.aichattest.voice.LocalAsr
import com.samanramezani.aichattest.voice.LocalQwen3Asr
import com.samanramezani.aichattest.voice.VoiceModelEntry
import com.samanramezani.aichattest.voice.LocalNemoCtcAsr
import com.samanramezani.aichattest.voice.LayaDecisionEngine
import com.samanramezani.aichattest.voice.DinaDeviceCommandExecutor
import com.woogit.aicore.agent.AgentEvent
import com.woogit.aicore.conversation.ConversationMessage
import com.woogit.aicore.domain.InferenceSettings
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.joinAll
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong

internal class VoiceConversationController(
    private val context: Context,
    private val container: AppContainer,
    private val decisionEngine: LayaDecisionEngine,
    private val onStatus: (String) -> Unit,
    private val onDiagnostic: (String) -> Unit,
    private val onListening: (Boolean) -> Unit,
    private val onLine: (String, String) -> Unit,
    private val onPartialAssistant: (String) -> Unit = {},
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
    // Keep only the newest pending utterance; stale requests never accumulate.
    private val pendingUtterances = AtomicReference<String?>(null)
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
                        val ttsEngine = LocalPiperTts(requireNotNull(ttsModel) { "VOICE-TTS-001: مدل Piper انتخاب نشده است." })
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
                        val update = try { recognizer?.accept(frame.samples, 16000, frame.speech) } catch (cancelled: CancellationException) {
                            throw cancelled
                        } catch (t: Throwable) {
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
                                // Inside an active dialogue a bare «خاموش شو» is enough (wake word optional).
                                if (com.samanramezani.aichattest.voice.DinaSleepIntent.matchesUtterance(utterance, requireWake = false)) {
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
                                    pendingUtterances.set(utterance)
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
            } catch (cancelled: CancellationException) {
                runCatching { microphone?.stop() }
                microphone = null
                cleanupEngines()
                restoreAudioMode()
                onListening(false)
                throw cancelled
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
        // stop() advances the epoch; barge-in cancellation deliberately does not.
        // This lets a replacement utterance survive cancellation of the previous turn.
        val turnEpoch = lifecycleEpoch.get()
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
                    } catch (cancelled: CancellationException) {
                        throw cancelled
                    } catch (t: Throwable) {
                        // One malformed/unsupported sentence must not discard the rest of the
                        // answer that is already waiting in the channel.
                        onDiagnostic("VOICE-TTS-SEGMENT-FAILED: chars=${segment.length} ${t.stackTraceToString()}")
                        onStatus("VOICE-TTS-SEGMENT-FAILED: یک بخش از پاسخ خوانده نشد؛ ادامهٔ پاسخ پخش می‌شود.")
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
        var segmenter = SpeechSegmenter()
        val queuedSpeechSegments = ArrayList<String>()
        fun enqueueCompleteSentences(token: String, flush: Boolean = false) {
            val ready = segmenter.feed(token) + if (flush) segmenter.flush() else emptyList()
            for (segment in ready) {
                if (queue.trySend(segment).isSuccess) queuedSpeechSegments += segment
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
                    persistUserMessageIfMissing(utterance)
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
            val voiceStreamRaw = StringBuilder()
            val voiceStreamSafeFed = StringBuilder()
            val session = container.createAgentSession(id, includeAgentTools = false, eventSink = { event ->
                when (event) {
                    is AgentEvent.Token -> {
                        // Voice has its own streaming consumer: sanitize the accumulated output,
                        // then send only newly available complete sentences to TTS. This avoids
                        // speaking split Markdown/reasoning markers and does not wait for the
                        // entire model response before starting audio playback.
                        voiceStreamRaw.append(event.value)
                        val safe = sanitizeAssistantVoiceText(voiceStreamRaw.toString())
                        if (safe.isNotBlank()) onPartialAssistant(safe)
                        val alreadyFed = voiceStreamSafeFed.toString()
                        if (safe.startsWith(alreadyFed)) {
                            val delta = safe.substring(alreadyFed.length)
                            if (delta.isNotEmpty()) {
                                enqueueCompleteSentences(delta)
                                voiceStreamSafeFed.append(delta)
                            }
                        }
                    }
                    is AgentEvent.Failed -> {
                        modelFailure.compareAndSet(null, event.message)
                        onDiagnostic("VOICE-LLM-001: ${event.message}")
                        onStatus("VOICE-LLM-001: ${event.message}")
                    }
                    else -> Unit
                }
            }) ?: error("VOICE-CHAT-002: نشست مدل محلی در دسترس نیست.")
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
                // Flush only text not already handed to the sentence queue while streaming.
                val streamedSafe = voiceStreamSafeFed.toString()
                if (answer.startsWith(streamedSafe)) {
                    enqueueCompleteSentences(answer.substring(streamedSafe.length), flush = true)
                } else if (answer.isNotBlank()) {
                    // A streamed prefix can differ from the final response after sanitization.
                    // Keep every already-queued matching sentence, discard only the unspoken
                    // partial buffer, then continue from the first changed final sentence.
                    val matchedPrefix = matchedSpeechPrefixLength(answer, queuedSpeechSegments)
                    segmenter.reset()
                    enqueueCompleteSentences(answer.substring(matchedPrefix), flush = true)
                    onDiagnostic(
                        "VOICE-TTS-STREAM-RECONCILE: streamedPrefixChars=${streamedSafe.length} " +
                            "queuedSegments=${queuedSpeechSegments.size} finalChars=${answer.length} " +
                            "resumeAt=$matchedPrefix"
                    )
                } else {
                    enqueueCompleteSentences("", flush = true)
                }
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
            throw t
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
            val canContinueSession = !closed && lifecycleEpoch.get() == turnEpoch
            val pending = if (canContinueSession) pendingUtterances.getAndSet(null) else null
            if (canContinueSession && pending != null && turnRunning.compareAndSet(false, true)) {
                onStatus("در حال پردازش گفتار جدید…")
                turnJob = scope.launch { processTurn(pending) }
            }
        }
    }

    private suspend fun persistUserMessageIfMissing(content: String) {
        val id = conversationId ?: error("VOICE-CHAT-001: شناسه مکالمه ایجاد نشده است.")
        if (content.isBlank()) return
        val last = container.conversationHistory.messages(id).lastOrNull()
        if (last?.role == ConversationMessage.Role.USER && last.content == content) return
        val now = System.currentTimeMillis()
        val saved = container.conversationHistory.append(
            id,
            ConversationMessage(
                id = java.util.UUID.randomUUID().toString(),
                role = ConversationMessage.Role.USER,
                content = content,
                timestampEpochMs = now,
            ),
            now,
        )
        check(saved) { "VOICE-HISTORY-002: پیام کاربر در تاریخچهٔ گفتگو ذخیره نشد." }
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
        } catch (cancelled: CancellationException) {
            throw cancelled
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
        pendingUtterances.set(null)
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
        return scope.launch {
            // AudioRecord.stop() and joining its capture thread can block; never do
            // that on the UI caller of stop().
            withContext(Dispatchers.IO) { oldMicrophone?.stop() }
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

