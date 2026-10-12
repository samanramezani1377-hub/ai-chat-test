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
    fun parsesJsonFenceAfterQwenReasoningAndPreservesPersianSpaces() {
        val intent = ActionProtocolParser().parse(
            """Reasoning about the requested file
</think>
```json
{"version":1,"actionId":"file-1","action":"create_file","arguments":{"file_name":"متن خام.txt","content":"اینجا شروع یک کار است"}}
```"""
        )
        requireNotNull(intent)
        assertEquals("create_file", intent.actionId)
        assertEquals("""{"file_name":"متن خام.txt","content":"اینجا شروع یک کار است"}""", intent.input)
    }

    @Test
    fun rejectsBashInsteadOfAnActionRequest() {
        assertNull(
            ActionProtocolParser().parse(
                """I should create a file
</think>
touch "متن خام.txt" && echo 'اینجا شروع یک کار است' > متن خام.txt"""
            )
        )
    }

    @Test
    fun rejectsDuplicateRootKeysRatherThanChoosingOneSilently() {
        assertNull(
            ActionProtocolParser().parse(
                """{"version":1,"actionId":"req","action":"calculate","action":"delete_file","arguments":{}}"""
            )
        )
    }

    @Test
    fun rejectsMultipleJsonObjects() {
        assertNull(
            ActionProtocolParser().parse(
                """{"version":1,"actionId":"a","action":"calculate","arguments":{}} {"version":1,"actionId":"b","action":"calculate","arguments":{}}"""
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
