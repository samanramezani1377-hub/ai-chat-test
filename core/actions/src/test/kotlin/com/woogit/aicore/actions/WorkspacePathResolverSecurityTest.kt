package com.woogit.aicore.actions

import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertFailsWith

class WorkspacePathResolverSecurityTest {
    @Test
    fun rejectsSymlinkEscape() {
        val root = Files.createTempDirectory("workspace-root")
        val outside = Files.createTempDirectory("workspace-outside")
        try {
            val link = root.resolve("escape")
            try {
                Files.createSymbolicLink(link, outside)
            } catch (_: UnsupportedOperationException) {
                return
            } catch (_: java.nio.file.FileSystemException) {
                return
            }
            val resolver = WorkspacePathResolver(root)
            assertFailsWith<IllegalArgumentException> { resolver.resolve("escape/secret.txt") }
        } finally {
            Files.walk(root).sorted(Comparator.reverseOrder()).forEach { Files.deleteIfExists(it) }
            Files.walk(outside).sorted(Comparator.reverseOrder()).forEach { Files.deleteIfExists(it) }
        }
    }
}
