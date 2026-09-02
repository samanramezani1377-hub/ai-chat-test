package com.woogit.aicore.actions

import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class BuiltinActionVerifierTest {
    @Test
    fun createFileVerificationChecksActualBytes() = kotlinx.coroutines.test.runTest {
        val root = Files.createTempDirectory("verifier")
        try {
            val resolver = WorkspacePathResolver(root)
            val action = CreateFileAction(resolver)
            val prepared = PreparedAction(actionId = "create_file", input = "{\"file_name\":\"hello.txt\",\"content\":\"سلام\"}", risk = com.woogit.aicore.domain.RiskLevel.NORMAL)
            val result = action.execute(prepared.input)
            val verification = BuiltinActionVerifier(root).verify(prepared, result)
            assertTrue(verification.success)
            Files.writeString(root.resolve("hello.txt"), "tampered")
            assertFalse(BuiltinActionVerifier(root).verify(prepared, result).success)
        } finally {
            Files.walk(root).sorted(Comparator.reverseOrder()).forEach { Files.deleteIfExists(it) }
        }
    }

    @Test
    fun deleteFileVerificationRequiresPathToBeGone() = kotlinx.coroutines.test.runTest {
        val root = Files.createTempDirectory("verifier-delete")
        try {
            Files.writeString(root.resolve("gone.txt"), "x")
            val prepared = PreparedAction(actionId = "delete_file", input = "{\"file_name\":\"gone.txt\"}", risk = com.woogit.aicore.domain.RiskLevel.SENSITIVE)
            val action = DeleteFileAction(WorkspacePathResolver(root))
            val result = action.execute(prepared.input)
            assertTrue(BuiltinActionVerifier(root).verify(prepared, result).success)
        } finally {
            Files.walk(root).sorted(Comparator.reverseOrder()).forEach { Files.deleteIfExists(it) }
        }
    }
}
