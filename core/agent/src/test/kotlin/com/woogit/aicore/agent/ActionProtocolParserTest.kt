package com.woogit.aicore.agent

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class ActionProtocolParserTest {
    @Test
    fun parsesStrictV1RequestAndPreservesRequestId() {
        val intent = ActionProtocolParser().parse(
            "prefix {\"version\":1,\"actionId\":\"req-1\",\"action\":\"calculate\",\"arguments\":{\"expression\":\"2+2\"}} suffix"
        )
        requireNotNull(intent)
        assertEquals("calculate", intent.actionId)
        assertEquals("req-1", intent.requestId)
        assertEquals("{\"expression\":\"2+2\"}", intent.input)
    }

    @Test
    fun rejectsWrongVersion() {
        assertNull(ActionProtocolParser().parse("{\"version\":2,\"actionId\":\"req\",\"action\":\"calculate\",\"arguments\":{}}"))
    }

    @Test
    fun rejectsMissingArgumentsObject() {
        assertNull(ActionProtocolParser().parse("{\"version\":1,\"actionId\":\"req\",\"action\":\"calculate\"}"))
    }
}
