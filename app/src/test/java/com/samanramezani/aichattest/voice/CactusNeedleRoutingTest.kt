package com.samanramezani.aichattest.voice

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class CactusNeedleRoutingTest {
    @Test fun onlyOfficialExplicitEmptyCallCanFallBackToChat() {
        assertTrue(shouldForwardNeedleRequestToChat(true, false))
    }

    @Test fun suppressedToolCallNeverFallsBackToChat() {
        assertFalse(shouldForwardNeedleRequestToChat(false, true))
    }

    @Test fun runtimeOrProtocolFailureNeverPretendsToBeAbstention() {
        assertFalse(shouldForwardNeedleRequestToChat(false, false))
    }

    @Test fun explicitAbstentionCannotOverrideAnySelectedTool() {
        assertFalse(shouldForwardNeedleRequestToChat(true, true))
    }
}
