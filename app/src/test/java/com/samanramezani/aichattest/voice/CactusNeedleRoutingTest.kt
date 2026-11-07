package com.samanramezani.aichattest.voice

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class CactusNeedleRoutingTest {
    @Test fun chatFallbackIsAllowedOnlyWhenNeedleSelectedNoTool() {
        assertTrue(shouldForwardNeedleRequestToChat(toolSelected = false))
        assertFalse(shouldForwardNeedleRequestToChat(toolSelected = true))
    }
}
