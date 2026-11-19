package com.samanramezani.aichattest.screen

/**
 * Holds only the most recent sampled screen frame in process memory.
 * Frames are never written to disk and callers receive defensive copies.
 */
object ScreenFrameStore {
    data class Frame(val jpeg: ByteArray, val capturedAtMs: Long, val width: Int, val height: Int)

    @Volatile private var latest: Frame? = null
    @Volatile private var clock: () -> Long = { android.os.SystemClock.elapsedRealtime() }

    internal fun setClockForTests(now: () -> Long) { clock = now }
    internal fun resetClockForTests() { clock = { android.os.SystemClock.elapsedRealtime() } }

    fun publish(jpeg: ByteArray, width: Int, height: Int) {
        if (jpeg.isEmpty() || width <= 0 || height <= 0) return
        latest = Frame(jpeg.copyOf(), clock(), width, height)
    }

    fun latest(maxAgeMs: Long = 4_000L): Frame? {
        val frame = latest ?: return null
        if (clock() - frame.capturedAtMs > maxAgeMs) return null
        return frame.copy(jpeg = frame.jpeg.copyOf())
    }

    fun clear() { latest = null }
}
