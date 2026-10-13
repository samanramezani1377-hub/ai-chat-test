package com.samanramezani.aichattest.voice

import android.annotation.SuppressLint
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.MediaRecorder
import android.os.SystemClock
import android.media.audiofx.AcousticEchoCanceler
import android.media.audiofx.NoiseSuppressor
import com.k2fsa.sherpa.onnx.FeatureConfig
import com.k2fsa.sherpa.onnx.OfflineTts
import com.k2fsa.sherpa.onnx.OfflineTtsConfig
import com.k2fsa.sherpa.onnx.OfflineTtsModelConfig
import com.k2fsa.sherpa.onnx.OfflineTtsVitsModelConfig
import com.k2fsa.sherpa.onnx.OnlineModelConfig
import com.k2fsa.sherpa.onnx.OnlineRecognizer
import com.k2fsa.sherpa.onnx.OnlineRecognizerConfig
import com.k2fsa.sherpa.onnx.OnlineTransducerModelConfig
import com.k2fsa.sherpa.onnx.OfflineModelConfig
import com.k2fsa.sherpa.onnx.OfflineNemoEncDecCtcModelConfig
import com.k2fsa.sherpa.onnx.OfflineQwen3AsrModelConfig
import com.k2fsa.sherpa.onnx.OfflineRecognizer
import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig
import com.k2fsa.sherpa.onnx.OnlineStream
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import org.json.JSONObject
import java.util.concurrent.atomic.AtomicBoolean

private fun normalizePersianTtsText(value: String): String =
    java.text.Normalizer.normalize(value, java.text.Normalizer.Form.NFC)
        .replace('ي', 'ی')
        .replace('ى', 'ی')
        .replace('ك', 'ک')
        .replace("ۀ", "هٔ")
        .replace('ة', 'ه')
        .replace('\u0640'.toString(), "")
        .replace(Regex("[\\t\\x0B\\f\\r ]+"), " ")
        .trim()

class LocalPiperTts(private val entry: VoiceModelEntry) {
    private val engine: OfflineTts

    init {
        val tokensFile = entry.tokensFile
            ?: error("VOICE-TTS-006: مدل Piper فاقد tokens.txt است.")
        val dataDirectory = entry.dataDirectory
            ?: error("VOICE-TTS-006: مدل Piper فاقد پوشه espeak-ng-data است.")
        require(entry.modelFile.isFile && entry.modelFile.length() > 1024L) {
            "VOICE-TTS-006: فایل مدل ONNX وجود ندارد یا ناقص است: ${entry.modelFile.name}."
        }
        require(tokensFile.isFile && tokensFile.length() > 0L) {
            "VOICE-TTS-006: فایل tokens.txt وجود ندارد یا خالی است."
        }
        val requiredEspeakFiles = listOf("phontab", "phonindex", "phondata", "intonations")
        val missingEspeakFiles = requiredEspeakFiles.filter {
            val file = File(dataDirectory, it)
            !file.isFile || file.length() == 0L
        }
        require(missingEspeakFiles.isEmpty()) {
            "VOICE-TTS-006: داده‌های espeak-ng ناقص‌اند (${missingEspeakFiles.joinToString()})؛ مدل را با بستهٔ رسمی espeak-ng-data.tar.bz2 دوباره وارد کنید."
        }
        val tokenLines = tokensFile.useLines { lines -> lines.count { it.isNotBlank() } }
        require(tokenLines >= 2) {
            "VOICE-TTS-006: tokens.txt قالب معتبر Piper ندارد (تعداد ردیف‌ها: $tokenLines)."
        }

        // Piper publishes the voice's recommended inference values in its companion
        // *.onnx.json file. Honor those official per-voice settings instead of silently
        // relying on sherpa-onnx defaults, which can alter prosody/phoneme timing.
        val piperConfigFile = sequenceOf(
            File(entry.modelFile.parentFile, "\${entry.modelFile.name}.json"),
            File(entry.directory, "model.onnx.json"),
            File(entry.modelFile.parentFile, "model.onnx.json"),
        ).firstOrNull { it.isFile && it.length() in 1L..(8L * 1024L * 1024L) }
        val piperInference = runCatching {
            piperConfigFile?.let { JSONObject(it.readText(Charsets.UTF_8)).optJSONObject("inference") }
        }.getOrNull()
        val piperNoiseScale = piperInference?.optDouble("noise_scale", 0.667)?.toFloat()
            ?.takeIf { it.isFinite() && it in 0.0f..1.5f } ?: 0.667f
        val piperNoiseScaleW = piperInference?.optDouble("noise_w", 0.8)?.toFloat()
            ?.takeIf { it.isFinite() && it in 0.0f..1.5f } ?: 0.8f
        val piperLengthScale = piperInference?.optDouble("length_scale", 1.0)?.toFloat()
            ?.takeIf { it.isFinite() && it in 0.5f..1.5f } ?: 1.0f

        val vits = OfflineTtsVitsModelConfig().apply {
            model = entry.modelFile.absolutePath
            tokens = tokensFile.absolutePath
            dataDir = dataDirectory.absolutePath
            noiseScale = piperNoiseScale
            noiseScaleW = piperNoiseScaleW
            lengthScale = piperLengthScale
        }
        val modelConfig = OfflineTtsModelConfig().apply {
            this.vits = vits
            numThreads = 2
            debug = false
            provider = "cpu"
        }
        engine = try {
            OfflineTts(config = OfflineTtsConfig().apply {
                model = modelConfig
                maxNumSentences = 1
                // Keep Piper's normal inter-word silence; aggressively shrinking it can make
                // short Persian phrases sound clipped or rushed.
                silenceScale = 0.2f
            })
        } catch (t: Throwable) {
            throw IllegalArgumentException(
                "VOICE-TTS-007: ساخت موتور sherpa-onnx برای مدل «${entry.title}» ناموفق بود. " +
                    "مدل=${entry.modelFile.name} (${entry.modelFile.length()} بایت)، " +
                    "توکن‌ها=$tokenLines ردیف، espeak=${dataDirectory.absolutePath}. " +
                    "فایل‌های مدل یا metadata ممکن است با Piper/VITS سازگار نباشند. علت بومی: ${t.message ?: t.javaClass.simpleName}",
                t,
            )
        }
    }

    suspend fun speak(
        text: String,
        cancelled: AtomicBoolean,
        onStarted: () -> Unit = {},
        onSynthesisComplete: (Long) -> Unit = {},
        speed: Float = 1.0f,
        playbackTailMs: Long = 0L,
    ) = withContext(Dispatchers.Default) {
        require(text.isNotBlank()) { "VOICE-TTS-002: متن گفتار خالی است." }
        require(speed in 0.5f..1.5f) { "VOICE-TTS-008: سرعت گفتار خارج از محدودهٔ مجاز است." }
        require(playbackTailMs in 0L..2000L) { "VOICE-TTS-009: مکث انتهایی خارج از محدودهٔ مجاز است." }
        val synthesisStartedAtMs = SystemClock.elapsedRealtime()
        val normalizedText = normalizePersianTtsText(text)
        // Normalize common Arabic-script variants before espeak phonemization.
        val audio = engine.generate(text = normalizedText, sid = 0, speed = speed)
        onSynthesisComplete(SystemClock.elapsedRealtime() - synthesisStartedAtMs)
        if (cancelled.get()) throw CancellationException("TTS interrupted")
        val samples = audio.samples
        require(samples.isNotEmpty() && samples.any { it > 1e-5f || it < -1e-5f }) {
            "VOICE-TTS-011: مدل «${entry.title}» برای متن «${normalizedText.take(48)}» صدای قابل پخشی تولید نکرد؛ samples=${samples.size}."
        }
        val rate = audio.sampleRate
        require(rate > 0) { "VOICE-TTS-012: نرخ نمونه‌برداری خروجی TTS نامعتبر است: $rate." }
        val minBuffer = AudioTrack.getMinBufferSize(rate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT)
        require(minBuffer > 0) { "VOICE-TTS-003: پیکربندی خروجی صوتی پشتیبانی نمی‌شود." }
        val track = AudioTrack.Builder()
            .setAudioAttributes(AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA).setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
            .setAudioFormat(AudioFormat.Builder().setSampleRate(rate).setChannelMask(AudioFormat.CHANNEL_OUT_MONO).setEncoding(AudioFormat.ENCODING_PCM_16BIT).build())
            .setBufferSizeInBytes(maxOf(minBuffer, rate / 5 * 2))
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()
        try {
            track.play()
            onStarted()
            var offset = 0
            val pcm = ShortArray(4096)
            while (offset < samples.size && !cancelled.get()) {
                val count = minOf(pcm.size, samples.size - offset)
                for (i in 0 until count) pcm[i] = (samples[offset + i].coerceIn(-1f, 1f) * Short.MAX_VALUE).toInt().toShort()
                var written = 0
                while (written < count && !cancelled.get()) {
                    val n = track.write(pcm, written, count - written, AudioTrack.WRITE_BLOCKING)
                    if (n < 0) error("VOICE-TTS-004: پخش صوت متوقف شد (AudioTrack=$n).")
                    written += n
                }
                offset += count
            }
            if (cancelled.get()) {
                track.pause()
            } else {
                // Drain queued PCM before releasing AudioTrack; otherwise the final syllable can be clipped.
                val framesToPlay = samples.size
                val expectedPlaybackMs = (framesToPlay * 1000L / rate).coerceAtLeast(1L)
                val drainDeadlineMs = SystemClock.elapsedRealtime() + expectedPlaybackMs + 1500L
                while (!cancelled.get() && track.playbackHeadPosition.toLong() < framesToPlay &&
                    SystemClock.elapsedRealtime() < drainDeadlineMs
                ) {
                    kotlinx.coroutines.delay(20)
                }
                if (!cancelled.get() && track.playbackHeadPosition.toLong() < framesToPlay) {
                    error("VOICE-TTS-010: پخش صوت پیش از پایان بافر متوقف شد؛ playbackHead=" +
                        track.playbackHeadPosition + "/" + framesToPlay)
                }
                if (!cancelled.get() && playbackTailMs > 0L) kotlinx.coroutines.delay(playbackTailMs)
                if (!cancelled.get()) track.stop()
            }
        } finally {
            runCatching { track.flush() }
            track.release()
        }
        if (cancelled.get()) throw CancellationException("TTS interrupted")
    }

    fun close() = engine.release()
}

/** Streaming local ASR for sherpa-onnx online transducer packages. */
data class AsrUpdate(val text: String, val endpoint: Boolean)

interface LocalAsr {
    fun accept(samples: FloatArray, sampleRate: Int = 16000, speech: Boolean = true): AsrUpdate
    fun close()
}

class LocalStreamingAsr(private val entry: VoiceModelEntry) : LocalAsr {
    private val recognizer: OnlineRecognizer
    private val stream: OnlineStream

    init {
        val model = OnlineModelConfig().apply {
            transducer = OnlineTransducerModelConfig().apply {
                encoder = requireNotNull(entry.encoder).absolutePath
                decoder = requireNotNull(entry.decoder).absolutePath
                joiner = requireNotNull(entry.joiner).absolutePath
            }
            tokens = requireNotNull(entry.tokensFile).absolutePath
            numThreads = 2
            provider = "cpu"
            modelType = entry.modelType
        }
        recognizer = OnlineRecognizer(config = OnlineRecognizerConfig().apply {
            featConfig = FeatureConfig().apply { sampleRate = 16000; featureDim = 80 }
            modelConfig = model
            enableEndpoint = true
        })
        stream = recognizer.createStream()
    }

    override fun accept(samples: FloatArray, sampleRate: Int, speech: Boolean): AsrUpdate {
        stream.acceptWaveform(samples, sampleRate)
        while (recognizer.isReady(stream)) recognizer.decode(stream)
        val text = recognizer.getResult(stream).text
        val endpoint = recognizer.isEndpoint(stream)
        if (endpoint) recognizer.reset(stream)
        return AsrUpdate(text, endpoint)
    }

    override fun close() {
        runCatching { stream.release() }
        recognizer.release()
    }
}

/**
 * Offline Qwen3-ASR adapter for multilingual Persian speech. It accumulates the current
 * utterance while microphone capture stays live, then decodes after 600 ms of silence.
 * VAD-based barge-in remains immediate even though transcript finalization is offline.
 */
/**
 * Tiny Persian Shenava Rizeh-Pizeh FastConformer CTC model (6.9M parameters).
 * This export is non-streaming: keep the microphone live, buffer the utterance,
 * and decode locally after endpoint silence.
 */
class LocalNemoCtcAsr(private val entry: VoiceModelEntry) : LocalAsr {
    private val recognizer: OfflineRecognizer
    private val utterance = ArrayList<Float>(16000 * 4)
    private var hasSpeech = false
    private var silenceSamples = 0
    private val maxUtteranceSamples = 16000 * 20
    private val endSilenceSamples = 16000 * 3 / 5

    init {
        val modelPath = entry.modelFile.absolutePath
        val tokensPath = requireNotNull(entry.tokensFile).absolutePath
        require(entry.modelFile.isFile && entry.modelFile.length() > 1024) {
            "VOICE-STT-CTC-001: فایل model.onnx وجود ندارد یا خالی است."
        }
        require(File(tokensPath).isFile) { "VOICE-STT-CTC-002: فایل tokens.txt پیدا نشد." }
        recognizer = OfflineRecognizer(config = OfflineRecognizerConfig().apply {
            featConfig = FeatureConfig().apply { sampleRate = 16000; featureDim = 80 }
            modelConfig = OfflineModelConfig().apply {
                nemo = OfflineNemoEncDecCtcModelConfig().apply { model = modelPath }
                tokens = tokensPath
                numThreads = 2
                debug = false
                provider = "cpu"
            }
            decodingMethod = "greedy_search"
        })
    }

    /**
     * Bounded rolling-window probe used only to spot "دینا" before the current utterance ends.
     * It creates a temporary stream and never appends probe samples to the persistent utterance buffer.
     */
    fun recognizeWindow(samples: FloatArray, sampleRate: Int = 16000): String {
        require(sampleRate == 16000) { "VOICE-STT-CTC-003: مدل Shenava به صدای ۱۶ کیلوهرتز نیاز دارد." }
        if (samples.size < 8000) return ""
        val stream = recognizer.createStream()
        return try {
            stream.acceptWaveform(samples, sampleRate)
            recognizer.decode(stream)
            recognizer.getResult(stream).text.trim()
        } finally {
            stream.release()
        }
    }

    override fun accept(samples: FloatArray, sampleRate: Int, speech: Boolean): AsrUpdate {
        require(sampleRate == 16000) { "VOICE-STT-CTC-003: مدل Shenava به صدای ۱۶ کیلوهرتز نیاز دارد." }
        if (speech) {
            hasSpeech = true
            silenceSamples = 0
        } else if (hasSpeech) {
            silenceSamples += samples.size
        }
        if (hasSpeech) {
            val remaining = maxUtteranceSamples - utterance.size
            val count = minOf(remaining, samples.size)
            for (i in 0 until count) utterance.add(samples[i])
        }
        if (!hasSpeech || (silenceSamples < endSilenceSamples && utterance.size < maxUtteranceSamples)) {
            return AsrUpdate("", false)
        }
        val audio = FloatArray(utterance.size) { utterance[it] }
        utterance.clear()
        hasSpeech = false
        silenceSamples = 0
        val stream = recognizer.createStream()
        return try {
            stream.acceptWaveform(audio, sampleRate)
            recognizer.decode(stream)
            AsrUpdate(recognizer.getResult(stream).text.trim(), true)
        } finally {
            stream.release()
        }
    }

    override fun close() {
        utterance.clear()
        recognizer.release()
    }
}

class LocalQwen3Asr(private val entry: VoiceModelEntry) : LocalAsr {
    private val recognizer: OfflineRecognizer
    private val utterance = ArrayList<Float>(16000 * 8)
    private var hasSpeech = false
    private var silenceSamples = 0
    private val maxUtteranceSamples = 16000 * 30
    private val endSilenceSamples = 16000 * 3 / 5

    init {
        val qwen = OfflineQwen3AsrModelConfig().apply {
            convFrontend = requireNotNull(entry.convFrontend).absolutePath
            encoder = requireNotNull(entry.encoder).absolutePath
            decoder = requireNotNull(entry.decoder).absolutePath
            tokenizer = requireNotNull(entry.tokenizerDirectory).absolutePath
            maxTotalLen = 512
            maxNewTokens = 256
            temperature = 1e-6f
            topP = 0.8f
        }
        recognizer = OfflineRecognizer(config = OfflineRecognizerConfig().apply {
            featConfig = FeatureConfig().apply { sampleRate = 16000; featureDim = 80 }
            modelConfig = OfflineModelConfig().apply {
                qwen3Asr = qwen
                numThreads = 2
                debug = false
                provider = "cpu"
            }
        })
    }

    override fun accept(samples: FloatArray, sampleRate: Int, speech: Boolean): AsrUpdate {
        require(sampleRate == 16000) { "VOICE-STT-QWEN-001: مدل Qwen3-ASR به صدای ۱۶ کیلوهرتز نیاز دارد." }
        if (speech) {
            hasSpeech = true
            silenceSamples = 0
        } else if (hasSpeech) {
            silenceSamples += samples.size
        }
        if (hasSpeech) {
            // Bound memory even if an upstream caller sends unusually large frames.
            val remaining = maxUtteranceSamples - utterance.size
            val count = minOf(remaining, samples.size)
            for (i in 0 until count) utterance.add(samples[i])
        }
        if (!hasSpeech || (silenceSamples < endSilenceSamples && utterance.size < maxUtteranceSamples)) {
            return AsrUpdate("", false)
        }
        val audio = FloatArray(utterance.size) { utterance[it] }
        utterance.clear()
        hasSpeech = false
        silenceSamples = 0
        val stream = recognizer.createStream()
        return try {
            stream.acceptWaveform(audio, sampleRate)
            recognizer.decode(stream)
            AsrUpdate(recognizer.getResult(stream).text.trim(), true)
        } finally {
            stream.release()
        }
    }

    override fun close() {
        utterance.clear()
        recognizer.release()
    }
}

class LocalMicrophone {
    data class Frame(val samples: FloatArray, val rms: Float, val speech: Boolean)
    private var recorder: AudioRecord? = null
    private var echoCanceler: AcousticEchoCanceler? = null
    private var noiseSuppressor: NoiseSuppressor? = null
    val isEchoCancellationActive: Boolean get() = runCatching { echoCanceler?.enabled == true }.getOrDefault(false)
    val isNoiseSuppressionActive: Boolean get() = runCatching { noiseSuppressor?.enabled == true }.getOrDefault(false)
    private val running = AtomicBoolean(false)
    @Volatile private var captureThread: Thread? = null

    // AudioRecord construction/startRecording are permission-gated by Android. Both calls below
    // explicitly translate SecurityException into VOICE-MIC-004, including permission revocation races.
    @SuppressLint("MissingPermission")
    fun start(onFrame: (Frame) -> Unit) {
        check(running.compareAndSet(false, true)) { "VOICE-MIC-001: میکروفون از قبل فعال است." }
        val sampleRate = 16000
        val min = try {
            AudioRecord.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        } catch (t: Throwable) {
            running.set(false)
            throw IllegalStateException("VOICE-MIC-002: بررسی بافر میکروفون ناموفق بود: ${t.message ?: t.javaClass.simpleName}", t)
        }
        if (min <= 0) {
            running.set(false)
            error("VOICE-MIC-002: اندازه بافر میکروفون نامعتبر است.")
        }
        val record = try {
            AudioRecord(
                MediaRecorder.AudioSource.VOICE_COMMUNICATION,
                sampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                maxOf(min * 2, sampleRate / 2),
            )
        } catch (t: Throwable) {
            running.set(false)
            if (t is SecurityException) {
                throw IllegalStateException("VOICE-MIC-004: مجوز ضبط صدا داده نشده است.", t)
            }
            throw IllegalStateException("VOICE-MIC-005: ساخت AudioRecord ناموفق بود: ${t.message ?: t.javaClass.simpleName}", t)
        }
        if (record.state != AudioRecord.STATE_INITIALIZED) {
            running.set(false); record.release()
            error("VOICE-MIC-003: راه‌اندازی میکروفون ناموفق بود.")
        }
        recorder = record
        // Audio effects are optional and OEM implementations can throw despite isAvailable().
        // Never fail microphone startup solely because an effect could not be instantiated.
        echoCanceler = runCatching {
            if (AcousticEchoCanceler.isAvailable()) {
                AcousticEchoCanceler.create(record.audioSessionId)?.apply { enabled = true }
            } else null
        }.getOrNull()
        noiseSuppressor = runCatching {
            if (NoiseSuppressor.isAvailable()) {
                NoiseSuppressor.create(record.audioSessionId)?.apply { enabled = true }
            } else null
        }.getOrNull()
        try {
            record.startRecording()
        } catch (t: Throwable) {
            running.set(false)
            recorder = null
            runCatching { noiseSuppressor?.release() }; noiseSuppressor = null
            runCatching { echoCanceler?.release() }; echoCanceler = null
            runCatching { record.release() }
            if (t is SecurityException) {
                throw IllegalStateException("VOICE-MIC-004: مجوز ضبط صدا داده نشده است.", t)
            }
            throw IllegalStateException("VOICE-MIC-005: شروع ضبط میکروفون ناموفق بود: ${t.message ?: t.javaClass.simpleName}", t)
        }
        captureThread = Thread({
            val buffer = ShortArray(640) // 40 ms at 16 kHz
            while (running.get()) {
                val n = try {
                    record.read(buffer, 0, buffer.size, AudioRecord.READ_BLOCKING)
                } catch (_: Throwable) {
                    if (running.get()) continue else break
                }
                if (n <= 0) {
                    if (!running.get()) break
                    continue
                }
                var energy = 0.0
                val samples = FloatArray(n)
                for (i in 0 until n) {
                    val v = buffer[i].toFloat() / Short.MAX_VALUE
                    samples[i] = v
                    energy += v * v
                }
                val rms = kotlin.math.sqrt(energy / n).toFloat()
                runCatching { onFrame(Frame(samples, rms, rms >= 0.018f)) }
            }
        }, "local-voice-microphone").apply { isDaemon = true; start() }
    }

    fun stop() {
        running.set(false)
        val oldRecorder = recorder
        runCatching { oldRecorder?.stop() }
        val oldThread = captureThread
        if (oldThread != null && oldThread !== Thread.currentThread()) {
            runCatching { oldThread.join(400) }
        }
        runCatching { echoCanceler?.release() }; echoCanceler = null
        runCatching { noiseSuppressor?.release() }; noiseSuppressor = null
        runCatching { oldRecorder?.release() }
        recorder = null
        captureThread = null
    }
}


/** Briefly suppresses microphone frames after Android volume-key presses to avoid treating key feedback as speech. */
object VoiceAudioInputGuard {
    private val suppressUntilMs = java.util.concurrent.atomic.AtomicLong(0L)

    fun suppressAfterVolumeKey() {
        suppressUntilMs.set(android.os.SystemClock.elapsedRealtime() + 900L)
    }

    fun shouldSuppress(nowMs: Long = android.os.SystemClock.elapsedRealtime()): Boolean =
        nowMs < suppressUntilMs.get()
}
