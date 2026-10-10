package com.samanramezani.aichattest.voice

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.MediaRecorder
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
import com.k2fsa.sherpa.onnx.OnlineStream
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.concurrent.atomic.AtomicBoolean

class LocalPiperTts(private val entry: VoiceModelEntry) {
    private val engine: OfflineTts

    init {
        val vits = OfflineTtsVitsModelConfig().apply {
            model = entry.modelFile.absolutePath
            tokens = requireNotNull(entry.tokensFile).absolutePath
            dataDir = requireNotNull(entry.dataDirectory).absolutePath
        }
        val modelConfig = OfflineTtsModelConfig().apply {
            this.vits = vits
            numThreads = 2
            debug = false
            provider = "cpu"
        }
        engine = OfflineTts(config = OfflineTtsConfig().apply {
            model = modelConfig
            maxNumSentences = 1
            silenceScale = 0.15f
        })
    }

    suspend fun speak(text: String, cancelled: AtomicBoolean, onStarted: () -> Unit = {}) = withContext(Dispatchers.Default) {
        require(text.isNotBlank()) { "VOICE-TTS-002: متن گفتار خالی است." }
        val audio = engine.generate(text = text, sid = 0, speed = 1.0f)
        if (cancelled.get()) throw CancellationException("TTS interrupted")
        val samples = audio.samples
        val rate = audio.sampleRate
        val minBuffer = AudioTrack.getMinBufferSize(rate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT)
        require(minBuffer > 0) { "VOICE-TTS-003: پیکربندی خروجی صوتی پشتیبانی نمی‌شود." }
        val track = AudioTrack.Builder()
            .setAudioAttributes(AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_ASSISTANT).setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
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
            if (cancelled.get()) track.pause() else track.stop()
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

class LocalStreamingAsr(private val entry: VoiceModelEntry) {
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

    fun accept(samples: FloatArray, sampleRate: Int = 16000): AsrUpdate {
        stream.acceptWaveform(samples, sampleRate)
        while (recognizer.isReady(stream)) recognizer.decode(stream)
        val text = recognizer.getResult(stream).text
        val endpoint = recognizer.isEndpoint(stream)
        if (endpoint) recognizer.reset(stream)
        return AsrUpdate(text, endpoint)
    }

    fun close() {
        runCatching { stream.release() }
        recognizer.release()
    }
}

class LocalMicrophone {
    data class Frame(val samples: FloatArray, val rms: Float, val speech: Boolean)
    private var recorder: AudioRecord? = null
    private var echoCanceler: AcousticEchoCanceler? = null
    private var noiseSuppressor: NoiseSuppressor? = null
    private val running = AtomicBoolean(false)

    fun start(onFrame: (Frame) -> Unit) {
        check(running.compareAndSet(false, true)) { "VOICE-MIC-001: میکروفون از قبل فعال است." }
        val sampleRate = 16000
        val min = AudioRecord.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        if (min <= 0) {
            running.set(false)
            error("VOICE-MIC-002: اندازه بافر میکروفون نامعتبر است.")
        }
        val record = AudioRecord(
            MediaRecorder.AudioSource.VOICE_COMMUNICATION,
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            maxOf(min * 2, sampleRate / 2),
        )
        if (record.state != AudioRecord.STATE_INITIALIZED) {
            running.set(false); record.release()
            error("VOICE-MIC-003: راه‌اندازی میکروفون ناموفق بود.")
        }
        recorder = record
        if (AcousticEchoCanceler.isAvailable()) echoCanceler = AcousticEchoCanceler.create(record.audioSessionId)?.apply { enabled = true }
        if (NoiseSuppressor.isAvailable()) noiseSuppressor = NoiseSuppressor.create(record.audioSessionId)?.apply { enabled = true }
        record.startRecording()
        Thread({
            val buffer = ShortArray(640) // 40 ms at 16 kHz
            while (running.get()) {
                val n = record.read(buffer, 0, buffer.size, AudioRecord.READ_BLOCKING)
                if (n <= 0) continue
                var energy = 0.0
                val samples = FloatArray(n)
                for (i in 0 until n) {
                    val v = buffer[i].toFloat() / Short.MAX_VALUE
                    samples[i] = v
                    energy += v * v
                }
                val rms = kotlin.math.sqrt(energy / n).toFloat()
                onFrame(Frame(samples, rms, rms >= 0.018f))
            }
        }, "local-voice-microphone").apply { isDaemon = true; start() }
    }

    fun stop() {
        running.set(false)
        runCatching { recorder?.stop() }
        recorder?.release()
        recorder = null
        echoCanceler?.release(); echoCanceler = null
        noiseSuppressor?.release(); noiseSuppressor = null
    }
}
