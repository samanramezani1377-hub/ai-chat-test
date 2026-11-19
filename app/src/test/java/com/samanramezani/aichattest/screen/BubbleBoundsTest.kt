package com.samanramezani.aichattest.screen

import org.junit.Assert.assertEquals
import org.junit.Test

class BubbleBoundsTest {
    @Test fun clampsBothAxesToKeepTheWholeBubbleVisible() {
        assertEquals(0, BubbleBounds.clampX(-20, 1080, 120))
        assertEquals(960, BubbleBounds.clampX(1400, 1080, 120))
        assertEquals(0, BubbleBounds.clampY(-20, 2400, 120))
        assertEquals(2280, BubbleBounds.clampY(2600, 2400, 120))
    }

    @Test fun handlesDisplaysSmallerThanTheBubble() {
        assertEquals(0, BubbleBounds.clampX(50, 80, 120))
        assertEquals(0, BubbleBounds.clampY(50, 80, 120))
    }
}
