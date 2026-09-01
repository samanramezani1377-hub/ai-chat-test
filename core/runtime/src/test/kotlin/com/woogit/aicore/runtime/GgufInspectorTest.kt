package com.woogit.aicore.runtime

import com.woogit.aicore.domain.ModelResult
import com.woogit.aicore.domain.Quantization
import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs

class GgufInspectorTest {
    @Test
    fun `inspects GGUF v3 with canonical KV type ids and Q6_K file type`() {
        val file = Files.createTempFile("qwen3-test-", ".gguf")
        try {
            Files.write(file, buildMinimalQwen3Q6K())

            val result = kotlinx.coroutines.runBlocking { GgufInspector().inspect(file) }
            val inspection = assertIs<ModelResult.Success<*>>(result).value as com.woogit.aicore.domain.ModelInspection

            assertEquals(3, inspection.version)
            assertEquals(1L, inspection.tensorCount)
            assertEquals("qwen3", inspection.metadata.architecture)
            assertEquals(4096L, inspection.metadata.contextLength)
            assertEquals(2048L, inspection.metadata.embeddingLength)
            assertEquals(24L, inspection.metadata.blockCount)
            assertEquals(Quantization.Q6_K, inspection.quantization)
        } finally {
            Files.deleteIfExists(file)
        }
    }

    private fun buildMinimalQwen3Q6K(): ByteArray {
        val out = mutableListOf<Byte>()
        out.addAll("GGUF".encodeToByteArray().toList())
        out.writeIntLE(3)
        out.writeLongLE(1)
        out.writeLongLE(5)

        out.writeString("general.architecture")
        out.writeIntLE(8)
        out.writeString("qwen3")

        out.writeString("qwen3.context_length")
        out.writeIntLE(4)
        out.writeIntLE(4096)

        out.writeString("qwen3.embedding_length")
        out.writeIntLE(4)
        out.writeIntLE(2048)

        out.writeString("qwen3.block_count")
        out.writeIntLE(4)
        out.writeIntLE(24)

        out.writeString("general.file_type")
        out.writeIntLE(4)
        out.writeIntLE(14)

        return out.toByteArray()
    }

    private fun MutableList<Byte>.writeIntLE(value: Int) {
        repeat(4) { add((value ushr (it * 8)).toByte()) }
    }

    private fun MutableList<Byte>.writeLongLE(value: Long) {
        repeat(8) { add((value ushr (it * 8)).toByte()) }
    }

    private fun MutableList<Byte>.writeString(value: String) {
        val bytes = value.encodeToByteArray()
        writeLongLE(bytes.size.toLong())
        addAll(bytes.toList())
    }
}
