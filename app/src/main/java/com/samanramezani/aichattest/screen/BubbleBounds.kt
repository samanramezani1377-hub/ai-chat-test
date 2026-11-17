package com.samanramezani.aichattest.screen

/** Keeps the complete floating bubble inside the available display rectangle. */
internal object BubbleBounds {
    fun clampX(x: Int, screenWidth: Int, bubbleSize: Int): Int =
        x.coerceIn(0, (screenWidth - bubbleSize).coerceAtLeast(0))

    fun clampY(y: Int, screenHeight: Int, bubbleSize: Int): Int =
        y.coerceIn(0, (screenHeight - bubbleSize).coerceAtLeast(0))
}
