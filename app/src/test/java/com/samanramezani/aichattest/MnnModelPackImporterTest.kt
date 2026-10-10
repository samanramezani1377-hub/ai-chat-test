package com.samanramezani.aichattest

import com.woogit.aicore.domain.ModelFormat
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.nio.file.Files
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class MnnModelPackImporterTest {
    @Test
    fun importsValidMnnPackAndUsesConfigAsDescriptorPath() {
        val root = Files.createTempDirectory("mnn-import-test")
        try {
            val model = MnnModelPackImporter().unpack(ByteArrayInputStream(zip(validEntries())), root)
            assertEquals(ModelFormat.MNN, model.format)
            assertEquals("config.json", model.path.fileName.toString())
            assertTrue(Files.isRegularFile(model.path))
            assertTrue(Files.isRegularFile(model.path.parent.resolve("llm.mnn")))
            assertTrue(Files.isRegularFile(model.path.parent.resolve("llm.mnn.weight")))
        } finally {
            MnnModelPackImporter().deleteRecursively(root)
        }
    }

    @Test
    fun rejectsZipSlipEntries() {
        val root = Files.createTempDirectory("mnn-import-zip-slip")
        try {
            val entries = validEntries() + ("../escape.txt" to "bad")
            assertFailsWith<IllegalArgumentException> {
                MnnModelPackImporter().unpack(ByteArrayInputStream(zip(entries)), root)
            }
            assertTrue(!Files.exists(root.parent.resolve("escape.txt")))
        } finally {
            MnnModelPackImporter().deleteRecursively(root)
        }
    }

    @Test
    fun rejectsIncompleteMnnPack() {
        val root = Files.createTempDirectory("mnn-import-incomplete")
        try {
            assertFailsWith<IllegalArgumentException> {
                MnnModelPackImporter().unpack(ByteArrayInputStream(zip(listOf("config.json" to "{}"))), root)
            }
        } finally {
            MnnModelPackImporter().deleteRecursively(root)
        }
    }

    private fun validEntries() = listOf(
        "TestModel/config.json" to """{"llm_model":"llm.mnn","llm_weight":"llm.mnn.weight","backend_type":"opencl"}""",
        "TestModel/llm.mnn" to "model",
        "TestModel/llm.mnn.weight" to "weights",
        "TestModel/llm_config.json" to "{}",
        "TestModel/tokenizer.mtok" to "tokenizer",
        "TestModel/embeddings_bf16.bin" to "embeddings",
    )

    private fun zip(entries: List<Pair<String, String>>): ByteArray {
        val output = ByteArrayOutputStream()
        ZipOutputStream(output).use { zip ->
            entries.forEach { (name, value) ->
                zip.putNextEntry(ZipEntry(name))
                zip.write(value.toByteArray())
                zip.closeEntry()
            }
        }
        return output.toByteArray()
    }
}
