package com.samanramezani.aichattest.voice

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.os.SystemClock
import com.k2fsa.sherpa.onnx.OfflineTts
import com.k2fsa.sherpa.onnx.OfflineTtsConfig
import com.k2fsa.sherpa.onnx.OfflineTtsModelConfig
import com.k2fsa.sherpa.onnx.OfflineTtsVitsModelConfig
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import org.json.JSONObject
import java.util.concurrent.atomic.AtomicBoolean

interface LocalTtsEngine : java.io.Closeable {
    suspend fun speak(
        text: String,
        cancelled: AtomicBoolean,
        onStarted: () -> Unit = {},
        onSynthesisComplete: (Long) -> Unit = {},
        speed: Float = 1.0f,
        playbackTailMs: Long = 0L,
    )
}

class LocalPiperTts(private val entry: VoiceModelEntry) : LocalTtsEngine {
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
            File(entry.modelFile.parentFile, "${entry.modelFile.name}.json"),
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

    override suspend fun speak(
        text: String,
        cancelled: AtomicBoolean,
        onStarted: () -> Unit,
        onSynthesisComplete: (Long) -> Unit,
        speed: Float,
        playbackTailMs: Long,
    ) = withContext(Dispatchers.Default) {
        require(text.isNotBlank()) { "VOICE-TTS-002: متن گفتار خالی است." }
        require(speed in 0.5f..1.5f) { "VOICE-TTS-008: سرعت گفتار خارج از محدودهٔ مجاز است." }
        require(playbackTailMs in 0L..2000L) { "VOICE-TTS-009: مکث انتهایی خارج از محدودهٔ مجاز است." }
        val synthesisStartedAtMs = SystemClock.elapsedRealtime()
        // Strip emoji/URLs, transliterate Latin words and fix "/" before espeak phonemization.
        // A segment that is only emoji/symbols has nothing to say: skip it instead of
        // throwing, because an exception here aborts the rest of the spoken reply.
        val normalizedText = PersianSpeechText.normalize(text)
        if (!PersianSpeechText.hasSpeakableContent(normalizedText)) return@withContext
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

    override fun close() = engine.release()
}
