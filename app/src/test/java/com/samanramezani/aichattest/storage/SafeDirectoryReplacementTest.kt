package com.samanramezani.aichattest.storage

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class SafeDirectoryReplacementTest {
    @Test
    fun replacesExistingDirectoryAndRemovesBackup() {
        val root = createTempDir(prefix = "voice-model-directory-")
        try {
            val destination = File(root, "espeak-ng-data").apply {
                mkdirs()
                File(this, "old.dat").writeText("old")
            }
            val staging = File(root, ".extract").apply { mkdirs() }
            val source = File(staging, "espeak-ng-data").apply {
                mkdirs()
                File(this, "new.dat").writeText("new")
            }

            SafeDirectoryReplacement.replace(source, destination)

            assertTrue(File(destination, "new.dat").isFile)
            assertFalse(File(destination, "old.dat").exists())
            assertFalse(source.exists())
            assertEquals(0, root.listFiles().orEmpty().count { it.name.endsWith(".backup") })
        } finally {
            root.deleteRecursively()
        }
    }

    @Test
    fun missingSourceLeavesPreviousDirectoryUntouched() {
        val root = createTempDir(prefix = "voice-model-directory-failure-")
        try {
            val destination = File(root, "espeak-ng-data").apply {
                mkdirs()
                File(this, "known-good.dat").writeText("known-good")
            }

            try {
                SafeDirectoryReplacement.replace(File(root, "missing"), destination)
                throw AssertionError("Expected replacement to fail")
            } catch (_: IllegalArgumentException) {
                // Expected: preflight validation must not touch the previous directory.
            }

            assertEquals("known-good", File(destination, "known-good.dat").readText())
        } finally {
            root.deleteRecursively()
        }
    }
}
