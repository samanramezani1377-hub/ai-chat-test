package com.samanramezani.aichattest.voice

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.SystemClock
import android.media.audiofx.AcousticEchoCanceler
import android.media.audiofx.NoiseSuppressor
import java.util.concurrent.atomic.AtomicBoolean

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
