package com.samanramezani.aichattest.screen

import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test

class ScreenFrameStoreTest {
    private var now = 10_000L

    @Before fun setUp() {
        ScreenFrameStore.setClockForTests { now }
        ScreenFrameStore.clear()
    }

    @After fun tearDown() {
        ScreenFrameStore.clear()
        ScreenFrameStore.resetClockForTests()
    }

    @Test fun storesOnlyLatestFrameAndReturnsDefensiveCopies() {
        val original = byteArrayOf(1, 2, 3)
        ScreenFrameStore.publish(original, 100, 200)
        original[0] = 9
        val frame = ScreenFrameStore.latest()!!
        assertArrayEquals(byteArrayOf(1, 2, 3), frame.jpeg)
        frame.jpeg[1] = 9
        assertArrayEquals(byteArrayOf(1, 2, 3), ScreenFrameStore.latest()!!.jpeg)
    }

    @Test fun expiresFramesAndClearsThem() {
        ScreenFrameStore.publish(byteArrayOf(1), 10, 10)
        now += 4_001L
        assertNull(ScreenFrameStore.latest())
        ScreenFrameStore.clear()
        assertNull(ScreenFrameStore.latest())
    }
}
