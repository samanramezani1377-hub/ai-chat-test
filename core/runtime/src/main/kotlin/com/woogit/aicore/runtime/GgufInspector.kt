package com.woogit.aicore.runtime

import com.woogit.aicore.domain.ModelError
import com.woogit.aicore.domain.ModelFormat
import com.woogit.aicore.domain.ModelInspection
import com.woogit.aicore.domain.ModelInspector
import com.woogit.aicore.domain.ModelMetadata
import com.woogit.aicore.domain.ModelResult
import com.woogit.aicore.domain.Quantization
import java.io.BufferedInputStream
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardOpenOption

/** Reads GGUF header and KV metadata without depending on a UI or model runtime. */
class GgufInspector : ModelInspector {
    override suspend fun inspect(path: Path): ModelResult<ModelInspection> {
        return try {
            if (!Files.isRegularFile(path) || !Files.isReadable(path)) {
                ModelResult.Failure(ModelError.FileAccess("Model file cannot be read"))
            } else {
                val size = Files.size(path)
                if (size < 24) {
                    ModelResult.Failure(ModelError.InvalidModel("GGUF file is too small"))
                } else {
                    inspectFile(path, size)
                }
            }
        } catch (t: Throwable) {
            ModelResult.Failure(ModelError.InvalidModel("Unable to inspect GGUF model: ${t.message ?: t::class.simpleName}"))
        }
    }

    private fun inspectFile(path: Path, size: Long): ModelResult<ModelInspection> =
        BufferedInputStream(Files.newInputStream(path, StandardOpenOption.READ)).use { input ->
            val magic = ByteArray(4)
            input.readFully(magic)
            if (magic.decodeToString() != "GGUF") {
                return ModelResult.Failure(ModelError.UnsupportedFormat("Selected file is not a GGUF model"))
            }
            val version = input.readLeInt()
            if (version !in 1..3) {
                return ModelResult.Failure(ModelError.InvalidModel("Unsupported GGUF version: $version"))
            }
            val tensorCount = input.readLeLong()
            val metadataCount = input.readLeLong()
            if (tensorCount < 0 || metadataCount < 0 || metadataCount > 100_000) {
                return ModelResult.Failure(ModelError.InvalidMetadata("Invalid GGUF counts"))
            }

            val values = linkedMapOf<String, Any?>()
            repeat(metadataCount.toInt()) {
                val key = input.readString()
                val type = input.readLeInt()
                values[key] = input.readValue(type)
            }

            val architecture = values["general.architecture"] as? String
            val prefix = architecture ?: "llama"
            val metadata = ModelMetadata(
                architecture = architecture,
                name = values["general.name"] as? String,
                contextLength = values["$prefix.context_length"].asLongOrNull(),
                embeddingLength = values["$prefix.embedding_length"].asLongOrNull(),
                blockCount = values["$prefix.block_count"].asLongOrNull(),
                tokenizerModel = values["tokenizer.ggml.model"] as? String,
                raw = values
            )
            return ModelResult.Success(
                ModelInspection(
                    format = ModelFormat.GGUF,
                    version = version,
                    metadata = metadata,
                    quantization = quantizationFromFileType(values["general.file_type"]),
                    sizeBytes = size,
                    tensorCount = tensorCount
                )
            )
        }

    private fun quantizationFromFileType(value: Any?): Quantization = when ((value as? Number)?.toInt()) {
        0 -> Quantization.F32
        1 -> Quantization.F16
        2 -> Quantization.Q4_0
        3 -> Quantization.Q4_1
        7 -> Quantization.Q8_0
        8 -> Quantization.Q5_0
        9 -> Quantization.Q5_1
        10 -> Quantization.Q2_K
        11 -> Quantization.Q3_K_S
        12 -> Quantization.Q3_K_M
        13 -> Quantization.Q3_K_L
        14 -> Quantization.Q4_K_S
        15 -> Quantization.Q4_K_M
        17 -> Quantization.Q5_K_M
        18 -> Quantization.Q6_K
        19 -> Quantization.Q5_K_S
        else -> Quantization.UNKNOWN
    }

    private fun BufferedInputStream.readFully(target: ByteArray) {
        var offset = 0
        while (offset < target.size) {
            val count = read(target, offset, target.size - offset)
            if (count < 0) error("Unexpected end of file")
            offset += count
        }
    }

    private fun BufferedInputStream.readLeByte(): Int {
        val value = read()
        if (value < 0) error("Unexpected end of file")
        return value
    }

    private fun BufferedInputStream.readLeShort(): Int = readLeByte() or (readLeByte() shl 8)

    private fun BufferedInputStream.readLeInt(): Int {
        var result = 0
        repeat(4) { index -> result = result or (readLeByte() shl (index * 8)) }
        return result
    }

    private fun BufferedInputStream.readLeLong(): Long {
        var result = 0L
        repeat(8) { index -> result = result or (readLeByte().toLong() shl (index * 8)) }
        return result
    }

    private fun BufferedInputStream.readString(): String {
        val length = readLeLong()
        require(length in 0..(1024L * 1024 * 1024)) { "Invalid GGUF string length" }
        val bytes = ByteArray(length.toInt())
        readFully(bytes)
        return bytes.decodeToString()
    }

    private fun BufferedInputStream.readValue(type: Int): Any? = when (type) {
        0 -> readLeByte()
        1 -> readLeByte().toByte().toInt()
        2 -> readLeShort()
        3 -> readLeShort().toShort().toInt()
        4 -> readLeInt().toLong() and 0xffff_ffffL
        5 -> readLeInt()
        6 -> Float.fromBits(readLeInt())
        7 -> readLeByte() != 0
        8 -> readString()
        9 -> {
            val elementType = readLeInt()
            val count = readLeLong()
            require(count >= 0) { "Invalid GGUF array length" }
            repeat(count.toIntExact()) { skipValue(elementType) }
            GgufArrayInfo(elementType, count)
        }
        10 -> readLeLong()
        11 -> readLeLong()
        12 -> Double.fromBits(readLeLong())
        else -> error("Unsupported GGUF metadata type: $type")
    }

    private fun BufferedInputStream.skipValue(type: Int) {
        when (type) {
            0, 1, 7 -> skipFully(1)
            2, 3 -> skipFully(2)
            4, 5, 6 -> skipFully(4)
            8 -> {
                val length = readLeLong()
                require(length >= 0) { "Invalid GGUF string length" }
                skipFully(length)
            }
            9 -> {
                val elementType = readLeInt()
                val count = readLeLong()
                require(count >= 0) { "Invalid GGUF array length" }
                repeat(count.toIntExact()) { skipValue(elementType) }
            }
            10, 11, 12 -> skipFully(8)
            else -> error("Unsupported GGUF metadata type: $type")
        }
    }

    private fun BufferedInputStream.skipFully(count: Long) {
        require(count >= 0) { "Invalid skip length" }
        var remaining = count
        while (remaining > 0) {
            val skipped = skip(remaining)
            if (skipped > 0) {
                remaining -= skipped
            } else {
                if (read() < 0) error("Unexpected end of file")
                remaining--
            }
        }
    }

    private fun Long.toIntExact(): Int {
        require(this in 0..Int.MAX_VALUE.toLong()) { "GGUF array is too large" }
        return toInt()
    }

    private data class GgufArrayInfo(
        val elementType: Int,
        val count: Long,
    )

    private fun Any?.asLongOrNull(): Long? = when (this) {
        is Byte -> toLong()
        is Short -> toLong()
        is Int -> toLong()
        is Long -> this
        else -> null
    }
}
