package com.woogit.aicore.actions

import java.nio.file.Files
import java.time.Instant
import java.time.ZoneOffset
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlinx.coroutines.test.runTest

class BuiltinActionsTest {
    @Test
    fun calculateSupportsArithmeticWithoutCodeExecution() = runTest {
        assertEquals("14", CalculateAction().execute("{\"expression\":\"2 + 3 * 4\"}"))
        assertFailsWith<IllegalArgumentException> {
            CalculateAction().execute("{\"expression\":\"2 / 0\"}")
        }
    }

    @Test
    fun timeUsesInjectedClock() = runTest {
        val action = GetTimeAction({ Instant.parse("2026-01-02T03:04:05Z") }, ZoneOffset.UTC)
        assertEquals("2026-01-02T03:04:05Z", action.execute("{}"))
    }

    @Test
    fun fileActionsStayInsideWorkspace() = runTest {
        val root = Files.createTempDirectory("ai-chat-actions")
        try {
            val resolver = WorkspacePathResolver(root)
            val create = CreateFileAction(resolver)
            val read = ReadFileAction(resolver)
            create.execute("{\"file_name\":\"nested/test.txt\",\"content\":\"hello\"}")
            assertEquals("hello", read.execute("{\"file_name\":\"nested/test.txt\"}"))
            assertFailsWith<IllegalArgumentException> {
                read.execute("{\"file_name\":\"../outside.txt\"}")
            }
        } finally {
            root.toFile().deleteRecursively()
        }
    }

    @Test
    fun deleteIsSensitive() {
        val root = Files.createTempDirectory("ai-chat-actions")
        try {
            val action = DeleteFileAction(WorkspacePathResolver(root))
            assertEquals(com.woogit.aicore.domain.RiskLevel.SENSITIVE, action.risk)
        } finally {
            root.toFile().deleteRecursively()
        }
    }
}
