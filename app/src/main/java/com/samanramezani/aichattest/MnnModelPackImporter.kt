package com.samanramezani.aichattest

import com.woogit.aicore.domain.ModelDescriptor
import com.woogit.aicore.domain.ModelFormat
import com.woogit.aicore.domain.ModelMetadata
import com.woogit.aicore.domain.ModelState
import com.woogit.aicore.domain.Quantization
import com.woogit.aicore.domain.RuntimeCompatibility
import com.woogit.aicore.domain.ValidationStatus
import java.io.InputStream
import java.nio.file.Files
import java.nio.file.Path
import java.util.Comparator
import java.util.UUID
import java.util.zip.ZipInputStream

/** Safely expands an exported MNN model ZIP and validates required runtime files. */
internal class MnnModelPackImporter {
    fun unpack(input: InputStream, root: Path): ModelDescriptor {
        Files.createDirectories(root)
        val staging = root.resolve("import-${UUID.randomUUID()}")
        Files.createDirectories(staging)
        var totalBytes = 0L
        var entries = 0
        try {
            ZipInputStream(input.buffered()).use { zip ->
                while (true) {
                    val entry = zip.nextEntry ?: break
                    entries++
                    require(entries <= MAX_ENTRIES) { "MNN archive contains too many files" }
                    val target = staging.resolve(entry.name).normalize()
                    require(target.startsWith(staging) && target != staging) { "Unsafe path in MNN archive: ${entry.name}" }
                    if (entry.isDirectory) Files.createDirectories(target)
                    else {
                        Files.createDirectories(target.parent)
                        Files.newOutputStream(target).use { output ->
                            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                            while (true) {
                                val read = zip.read(buffer)
                                if (read < 0) break
                                totalBytes += read
                                require(totalBytes <= MAX_UNPACKED_BYTES) { "MNN archive exceeds the 8 GiB import limit" }
                                output.write(buffer, 0, read)
                            }
                        }
                    }
                    zip.closeEntry()
                }
            }
            val config = Files.walk(staging).use { paths ->
                paths.filter { Files.isRegularFile(it) && it.fileName.toString().equals("config.json", true) }
                    .filter { Files.isRegularFile(it.parent.resolve("llm.mnn")) && Files.isRegularFile(it.parent.resolve("llm.mnn.weight")) }
                    .findFirst().orElseThrow { IllegalArgumentException("ZIP is not an exported MNN LLM pack: config.json, llm.mnn and llm.mnn.weight must be together") }
            }
            val modelDir = config.parent
            val bytes = Files.walk(modelDir).use { paths -> paths.filter(Files::isRegularFile).mapToLong { Files.size(it) }.sum() }
            val name = modelDir.fileName?.toString()?.takeIf { it.isNotBlank() } ?: "MNN model"
            return ModelDescriptor(
                id = "mnn-${UUID.randomUUID()}",
                displayName = name,
                path = config,
                format = ModelFormat.MNN,
                quantization = Quantization.UNKNOWN,
                sizeBytes = bytes,
                metadata = ModelMetadata(name = name, contextLength = 8192L),
                validation = ValidationStatus.VALID,
                runtimeCompatibility = RuntimeCompatibility(true, "MNN pack; OpenCL first with CPU fallback"),
                state = ModelState.IMPORTED,
            )
        } catch (t: Throwable) {
            runCatching { deleteRecursively(staging) }
            throw t
        }
    }

    fun deleteRecursively(path: Path) {
        if (!Files.exists(path)) return
        Files.walk(path).use { paths -> paths.sorted(Comparator.reverseOrder<Path>()).forEach { Files.deleteIfExists(it) } }
    }

    companion object {
        private const val MAX_ENTRIES = 10000
        private const val MAX_UNPACKED_BYTES = 8L * 1024L * 1024L * 1024L
    }
}
