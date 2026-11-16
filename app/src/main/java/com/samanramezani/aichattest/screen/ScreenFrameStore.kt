package com.samanramezani.aichattest.screen

/**
 * Holds only the most recent sampled screen frame in process memory.
 * Frames are never written to disk and callers receive defensive copies.
 */
object ScreenFrameStore {
    data class Frame(val jpeg: ByteArray, val capturedAtMs: Long, val width: Int, val height: Int)

    @Volatile private var latest: Frame? = null

    fun publish(jpeg: ByteArray, width: Int, height: Int) {
        if (jpeg.isEmpty() || width <= 0 || height <= 0) return
        latest = Frame(jpeg.copyOf(), android.os.SystemClock.elapsedRealtime(), width, height)
    }

    fun latest(maxAgeMs: Long = 4_000L): Frame? {
        val frame = latest ?: return null
        if (android.os.SystemClock.elapsedRealtime() - frame.capturedAtMs > maxAgeMs) return null
        return frame.copy(jpeg = frame.jpeg.copyOf())
    }

    fun clear() { latest = null }
}
