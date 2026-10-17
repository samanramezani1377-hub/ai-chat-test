package com.woogit.aicore.runtime

import com.woogit.aicore.domain.ModelResult
import com.woogit.aicore.domain.ModelFormat
import kotlinx.coroutines.runBlocking
import java.nio.file.Files
import java.nio.file.Path
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertIs
import kotlin.test.assertTrue

class OnnxGenAiModelImporterTest {
    @Test
    fun importsValidPackageWithOptionalTopLevelDirectory() = runBlocking {
        val temp = Files.createTempDirectory("onnx-genai-import-test")
        try {
            val archive = temp.resolve("qwen3-test.zip")
            writeZip(archive, mapOf(
                "model/genai_config.json" to """{"model":{"type":"qwen3","context_length":4096,"decoder":{"filename":"model.onnx"}}}""",
                "model/model.onnx" to "test graph placeholder",
                "model/tokenizer.json" to """{"version":"1.0"}""",
            ))
            val result = OnnxGenAiModelImporter(temp.resolve("models")).import(archive)
            val model = assertIs<ModelResult.Success<com.woogit.aicore.domain.ModelDescriptor>>(result).value
            assertEquals(ModelFormat.ONNX_GENAI, model.format)
            assertEquals("qwen3", model.metadata.architecture)
            assertTrue(Files.isDirectory(model.path))
            assertTrue(Files.isRegularFile(model.path.resolve("genai_config.json")))
            assertTrue(model.sizeBytes > 0)
        } finally {
            deleteTree(temp)
        }
    }

    @Test
    fun rejectsPathTraversalArchiveAndDoesNotWriteOutsideStorage() = runBlocking {
        val temp = Files.createTempDirectory("onnx-genai-traversal-test")
        try {
            val storage = temp.resolve("models")
            Files.createDirectories(storage)
            val archive = temp.resolve("unsafe.zip")
            writeZip(archive, mapOf(
                "../escaped.txt" to "must not escape",
                "genai_config.json" to """{"model":{"type":"qwen3"}}""",
            ))
            val result = OnnxGenAiModelImporter(storage).import(archive)
            assertIs<ModelResult.Failure>(result)
            assertFalse(Files.exists(temp.resolve("escaped.txt")))
        } finally {
            deleteTree(temp)
        }
    }

    @Test
    fun rejectsSingleOnnxOrGgufFileInsteadOfPackageZip() = runBlocking {
        val temp = Files.createTempDirectory("onnx-genai-format-test")
        try {
            val file = temp.resolve("model.gguf")
            Files.writeString(file, "not a package")
            val result = OnnxGenAiModelImporter(temp.resolve("models")).import(file)
            val failure = assertIs<ModelResult.Failure>(result)
            assertTrue(failure.error.message.contains("ZIP"))
        } finally {
            deleteTree(temp)
        }
    }

    private fun writeZip(path: Path, files: Map<String, String>) {
        ZipOutputStream(Files.newOutputStream(path)).use { zip ->
            files.forEach { (name, value) ->
                zip.putNextEntry(ZipEntry(name))
                zip.write(value.toByteArray(Charsets.UTF_8))
                zip.closeEntry()
            }
        }
    }

    private fun deleteTree(path: Path) {
        if (!Files.exists(path)) return
        Files.walk(path).use { paths ->
            paths.sorted(Comparator.reverseOrder()).forEach { Files.deleteIfExists(it) }
        }
    }
}
