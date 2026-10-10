package com.woogit.aicore.agent

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class ActionProtocolParserTest {
    @Test
    fun parsesStrictV1RequestAndPreservesRequestId() {
        val intent = ActionProtocolParser().parse(
            """{"version":1,"actionId":"req-1","action":"calculate","arguments":{"expression":"2+2"}}"""
        )
        requireNotNull(intent)
        assertEquals("calculate", intent.actionId)
        assertEquals("req-1", intent.requestId)
        assertEquals("""{"expression":"2+2"}""", intent.input)
    }

    @Test
    fun rejectsFreeFormTextAroundRequest() {
        assertNull(ActionProtocolParser().parse("""prefix {"version":1,"actionId":"req","action":"calculate","arguments":{}} suffix"""))
    }

    @Test
    fun decodesJsonEscapesInActionId() {
        val intent = ActionProtocolParser().parse("""{"version":1,"actionId":"req-\u0031","action":"calculate","arguments":{}}""")
        requireNotNull(intent)
        assertEquals("req-1", intent.requestId)
    }

    @Test
    fun parsesActionRequestAfterClosedQwenReasoningBlock() {
        val intent = ActionProtocolParser().parse(
            """Let me calculate this carefully.
</think>
{"version":1,"actionId":"calc-2","action":"calculate","arguments":{"expression":"2+2"}}"""
        )
        requireNotNull(intent)
        assertEquals("calculate", intent.actionId)
        assertEquals("calc-2", intent.requestId)
    }

    @Test
    fun stillRejectsOrdinaryProseWrappedAroundActionRequest() {
        assertNull(
            ActionProtocolParser().parse(
                """Please run this: {"version":1,"actionId":"req","action":"calculate","arguments":{}}"""
            )
        )
    }

    @Test
    fun rejectsWrongVersion() {
        assertNull(ActionProtocolParser().parse("""{"version":2,"actionId":"req","action":"calculate","arguments":{}}"""))
    }

    @Test
    fun rejectsMissingArgumentsObject() {
        assertNull(ActionProtocolParser().parse("""{"version":1,"actionId":"req","action":"calculate"}"""))
    }
}
