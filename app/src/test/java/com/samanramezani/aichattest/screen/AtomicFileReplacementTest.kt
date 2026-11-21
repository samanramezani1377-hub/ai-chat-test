package com.samanramezani.aichattest.screen

import com.samanramezani.aichattest.storage.AtomicFileReplacement

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class AtomicFileReplacementTest {
    @Test fun replacesExistingFileWithNewContent() {
        val directory = createTempDir(prefix = "atomic-model-")
        try {
            val destination = File(directory, "model.gguf").apply { writeText("old-model") }
            val source = File(directory, "model.gguf.part").apply { writeText("new-model") }

            AtomicFileReplacement.replace(source, destination)

            assertEquals("new-model", destination.readText())
            assertFalse(source.exists())
            assertTrue(directory.listFiles().orEmpty().none { it.name.endsWith(".backup") })
        } finally {
            directory.deleteRecursively()
        }
    }

    @Test fun missingSourceLeavesPreviousFileUntouched() {
        val directory = createTempDir(prefix = "atomic-model-failure-")
        try {
            val destination = File(directory, "model.gguf").apply { writeText("known-good") }
            val missingSource = File(directory, "missing.part")

            try {
                AtomicFileReplacement.replace(missingSource, destination)
                throw AssertionError("Expected replacement to fail")
            } catch (_: IllegalArgumentException) {
                // Expected: the old destination must remain intact.
            }

            assertEquals("known-good", destination.readText())
        } finally {
            directory.deleteRecursively()
        }
    }
}
