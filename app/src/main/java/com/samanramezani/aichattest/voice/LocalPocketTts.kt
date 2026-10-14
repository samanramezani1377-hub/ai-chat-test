package com.samanramezani.aichattest.voice

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.os.SystemClock
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext

/** Playback adapter for the offline Pocket TTS Farsi ONNX synthesis engine. */
internal class LocalPocketTts(private val modelDir: File) : LocalTtsEngine {
    private val engine = PocketTtsFarsiEngine(modelDir)

    override suspend fun speak(
        text: String,
        cancelled: AtomicBoolean,
        onStarted: () -> Unit,
        onSynthesisComplete: (Long) -> Unit,
        speed: Float,
        playbackTailMs: Long,
    ) = withContext(Dispatchers.Default) {
        require(speed in 0.5f..1.5f && playbackTailMs in 0L..2000L) {
            "POCKET-TTS-PLAYBACK-001: تنظیمات سرعت یا مکث نامعتبر است."
        }
        val start = SystemClock.elapsedRealtime()
        val raw = engine.synthesize(text) { cancelled.get() }
        onSynthesisComplete(SystemClock.elapsedRealtime() - start)
        if (cancelled.get()) throw CancellationException("Pocket TTS interrupted")
        require(raw.isNotEmpty() && raw.any { kotlin.math.abs(it) > 1e-5f }) {
            "POCKET-TTS-PLAYBACK-002: خروجی صوتی خالی است."
        }
        val samples = if (speed == 1f) raw else {
            val size = maxOf(1, (raw.size / speed).toInt())
            FloatArray(size) { i ->
                val p = i * speed
                val a = minOf(raw.lastIndex, p.toInt())
                val b = minOf(raw.lastIndex, a + 1)
                raw[a] * (1f - (p - a)) + raw[b] * (p - a)
            }
        }
        val rate = 24_000
        val minBuffer = AudioTrack.getMinBufferSize(rate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT)
        require(minBuffer > 0) { "POCKET-TTS-PLAYBACK-003: AudioTrack پشتیبانی نمی‌شود." }
        val track = AudioTrack.Builder()
            .setAudioAttributes(AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA).setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
            .setAudioFormat(AudioFormat.Builder().setSampleRate(rate).setChannelMask(AudioFormat.CHANNEL_OUT_MONO).setEncoding(AudioFormat.ENCODING_PCM_16BIT).build())
            .setBufferSizeInBytes(maxOf(minBuffer, rate / 5 * 2))
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()
        try {
            track.play()
            onStarted()
            val pcm = ShortArray(4096)
            var offset = 0
            while (offset < samples.size && !cancelled.get()) {
                val count = minOf(pcm.size, samples.size - offset)
                for (i in 0 until count) pcm[i] = (samples[offset + i].coerceIn(-1f, 1f) * Short.MAX_VALUE).toInt().toShort()
                var written = 0
                while (written < count && !cancelled.get()) {
                    val n = track.write(pcm, written, count - written, AudioTrack.WRITE_BLOCKING)
                    if (n < 0) error("POCKET-TTS-PLAYBACK-004: AudioTrack=$n")
                    written += n
                }
                offset += count
            }
            if (cancelled.get()) track.pause() else {
                val deadline = SystemClock.elapsedRealtime() + samples.size * 1000L / rate + 1500L
                while (!cancelled.get() && track.playbackHeadPosition.toLong() < samples.size && SystemClock.elapsedRealtime() < deadline) delay(20)
                if (!cancelled.get() && track.playbackHeadPosition.toLong() < samples.size) error("POCKET-TTS-PLAYBACK-005: پخش پیش از پایان متوقف شد.")
                if (!cancelled.get() && playbackTailMs > 0) delay(playbackTailMs)
            }
        } finally {
            runCatching { track.flush() }
            track.release()
        }
        if (cancelled.get()) throw CancellationException("Pocket TTS interrupted")
    }

    override fun close() = engine.close()
}
