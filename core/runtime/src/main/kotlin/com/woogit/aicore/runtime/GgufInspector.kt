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

/** Reads the GGUF header and KV metadata without depending on a UI or runtime implementation. */
class GgufInspector : ModelInspector {
    override suspend fun inspect(path: Path): ModelResult<ModelInspection> {
        try {
            if (!Files.isRegularFile(path) || !Files.isReadable(path)) {
                return ModelResult.Failure(ModelError.FileAccess("Model file cannot be read"))
            }
            val size = Files.size(path)
            if (size < 24) return ModelResult.Failure(ModelError.InvalidModel("GGUF file is too small"))

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
                if (tensorCount < 0 || metadataCount < 0) {
                    return ModelResult.Failure(ModelError.InvalidMetadata("Invalid GGUF counts"))
                }

                val values = linkedMapOf<String, Any?>()
                repeat(metadataCount.coerceAtMost(100_000).toInt()) {
                    val key = input.readString()
                    val type = input.readLeInt()
                    values[key] = input.readValue(type)
                }

                val architecture = values["general.architecture"] as? String
                val name = values["general.name"] as? String
                val context = values["${architecture ?: "llama"}.context_length"] asLongOrNull()
                val embedding = values["${architecture ?: "llama"}.embedding_length"] asLongOrNull()
                val blocks = values["${architecture ?: "llama"}.block_count"] asLongOrNull()
                val tokenizer = values["general.file_type"]?.toString()

                // Tensor types are intentionally not guessed from the filename. The KV file_type
                // is mapped when available; tensor-level Q6_K detection is handled by the runtime.
                val quantization = quantizationFromFileType(values["general.file_type"])
                val metadata = ModelMetadata(
                    architecture = architecture,
                    name = name,
                    contextLength = context,
                    embeddingLength = embedding,
                    blockCount = blocks,
                    tokenizerModel = tokenizer,
                    raw = values
                )
                return ModelResult.Success(
                    ModelInspection(
                        format = ModelFormat.GGUF,
                        version = version,
                        metadata = metadata,
                        quantization = quantization,
                        sizeBytes = size,
                        tensorCount = tensorCount
                    )
                )
            }
        } catch (t: Throwable) {
            return ModelResult.Failure(ModelError.InvalidModel("Unable to inspect GGUF model", t))
        }
    }

    private fun quantizationFromFileType(value: Any?): Quantization = when ((value as? Number)?.toInt()) {
        0 -> Quantization.F32
        1 -> Quantization.F16
        2 -> Quantization.Q4_0
        3 -> Quantization.Q4_1
        6 -> Quantization.Q5_0
        7 -> Quantization.Q5_1
        8 -> Quantization.Q8_0
        10 -> Quantization.Q2_K
        11 -> Quantization.Q3_K_S
        12 -> Quantization.Q3_K_M
        13 -> Quantization.Q3_K_L
        14 -> Quantization.Q4_K_S
        15 -> Quantization.Q4_K_M
        16 -> Quantization.Q5_K_S
        17 -> Quantization.Q5_K_M
        18 -> Quantization.Q6_K
        else -> Quantization.UNKNOWN
    }

    private fun BufferedInputStream.readFully(target: ByteArray) {
        var offset = 0
        while (offset < target.size) {
            val read = read(target, offset, target.size - offset)
            if (read < 0) error("Unexpected end of file")
            offset += read
        }
    }

    private fun BufferedInputStream.readLeInt(): Int = Integer.reverseBytes(readInt())
    private fun BufferedInputStream.readLeLong(): Long = java.lang.Long.reverseBytes(readLong())

    private fun BufferedInputStream.readInt(): Int {
        val b0 = read(); val b1 = read(); val b2 = read(); val b3 = read()
        if (b3 < 0) error("Unexpected end of file")
        return b0 or (b1 shl 8) or (b2 shl 16) or (b3 shl 24)
    }

    private fun BufferedInputStream.readLong(): Long {
        var result = 0L
        repeat(8) { index ->
            val b = read()
            if (b < 0) error("Unexpected end of file")
            result = result or (b.toLong() shl (index * 8))
        }
        return result
    }

    private fun BufferedInputStream.readString(): String {
        val length = readLeLong()
        require(length in 0..(16L * 1024 * 1024)) { "Invalid GGUF string length" }
        val bytes = ByteArray(length.toInt())
        readFully(bytes)
        return bytes.decodeToString()
    }

    private fun BufferedInputStream.readValue(type: Int): Any? = when (type) {
        0 -> read() != 0
        1 -> readLeByte().toUByte().toInt()
        2 -> readLeByte().toByte().toInt()
        3 -> readLeShort().toInt()
        4 -> readLeShort().toShort().toInt()
        5 -> readLeInt().toLong()
        6 -> readLeInt()
        7 -> readLeFloat()
        8 -> readString()
        9 -> readLeInt().let { count -> List(count.coerceAtMost(100_000)) { readValue(readLeInt()) } }
        10 -> readLeLong().toULong()
        11 -> readLeLong()
        12 -> readLeDouble()
        13 -> readLeLong()
        14 -> readLeLong()
        15 -> readLeLong()
        16 -> readLeLong()
        else -> error("Unsupported GGUF metadata type: $type")
    }

    private fun BufferedInputStream.readLeByte(): Int { val b = read(); if (b < 0) error("Unexpected end of file"); return b }
    private fun BufferedInputStream.readLeShort(): Short {
        val lo = readLeByte(); val hi = readLeByte(); return (lo or (hi shl 8)).toShort()
    }
    private fun BufferedInputStream.readLeFloat(): Float = Float.fromBits(readLeInt())
    private fun BufferedInputStream.readLeDouble(): Double = Double.fromBits(readLeLong())
    private fun Any?.asLongOrNull(): Long? = this as? Long ?: (this as? Number)?.toLong()
}
