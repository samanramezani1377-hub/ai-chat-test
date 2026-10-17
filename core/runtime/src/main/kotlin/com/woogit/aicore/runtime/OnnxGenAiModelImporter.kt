package com.woogit.aicore.runtime

import com.woogit.aicore.domain.ModelDescriptor
import com.woogit.aicore.domain.ModelError
import com.woogit.aicore.domain.ModelFormat
import com.woogit.aicore.domain.ModelImporter
import com.woogit.aicore.domain.ModelMetadata
import com.woogit.aicore.domain.ModelResult
import com.woogit.aicore.domain.ModelState
import com.woogit.aicore.domain.Quantization
import com.woogit.aicore.domain.RuntimeCompatibility
import com.woogit.aicore.domain.ValidationStatus
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption
import java.util.UUID
import java.util.zip.ZipInputStream

/**
 * Imports a complete ONNX Runtime GenAI model package from a ZIP archive.
 *
 * The archive must contain genai_config.json, at least one ONNX graph and tokenizer assets.
 * ZIP entries are normalized and checked before extraction to prevent path traversal.
 */
class OnnxGenAiModelImporter(
    private val storageDirectory: Path,
    private val maxExpandedBytes: Long = 32L * 1024L * 1024L * 1024L,
    private val maxEntries: Int = 100_000,
) : ModelImporter {
    override suspend fun import(source: Path): ModelResult<ModelDescriptor> {
        if (!Files.isRegularFile(source) || !Files.isReadable(source)) {
            return ModelResult.Failure(ModelError.FileAccess("فایل بستهٔ مدل قابل خواندن نیست."))
        }
        if (!source.fileName.toString().lowercase().endsWith(".zip")) {
            return ModelResult.Failure(ModelError.UnsupportedFormat("برای ONNX Runtime GenAI باید بستهٔ کامل مدل با پسوند ZIP وارد شود؛ فایل GGUF یا ONNX تکی کافی نیست."))
        }

        val id = UUID.randomUUID().toString()
        val unpackRoot = storageDirectory.resolve("import-$id")
        val target = storageDirectory.resolve(id)
        try {
            Files.createDirectories(storageDirectory)
            Files.createDirectory(unpackRoot)
            var expandedBytes = 0L
            var entryCount = 0
            ZipInputStream(Files.newInputStream(source)).use { zip ->
                while (true) {
                    val entry = zip.nextEntry ?: break
                    entryCount++
                    if (entryCount > maxEntries) throw IllegalArgumentException("بستهٔ مدل تعداد فایل غیرمجاز دارد.")
                    val rawName = entry.name.replace('\\\\', '/')
                    val relative = Path.of(rawName).normalize()
                    if (rawName.startsWith("/") || relative.isAbsolute ||
                        relative.toString() == "." || relative.startsWith("..") ||
                        rawName.split('/').any { it == ".." }
                    ) {
                        throw IllegalArgumentException("مسیر ناامن داخل ZIP: $rawName")
                    }
                    val output = unpackRoot.resolve(relative).normalize()
                    if (!output.startsWith(unpackRoot)) throw IllegalArgumentException("مسیر ناامن داخل ZIP: $rawName")
                    if (entry.isDirectory) {
                        Files.createDirectories(output)
                    } else {
                        Files.createDirectories(output.parent)
                        Files.newOutputStream(output).use { sink ->
                            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                            while (true) {
                                val count = zip.read(buffer)
                                if (count < 0) break
                                expandedBytes += count
                                if (expandedBytes > maxExpandedBytes) {
                                    throw IllegalArgumentException("حجم بازشدهٔ مدل از سقف 32 GiB بیشتر است.")
                                }
                                sink.write(buffer, 0, count)
                            }
                        }
                    }
                    zip.closeEntry()
                }
            }

            val config = Files.walk(unpackRoot).use { paths ->
                paths.filter { Files.isRegularFile(it) && it.fileName.toString() == "genai_config.json" }
                    .findFirst().orElse(null)
            } ?: throw IllegalArgumentException("genai_config.json پیدا نشد؛ این ZIP بستهٔ معتبر ONNX Runtime GenAI نیست.")

            val modelRoot = config.parent
            val configText = Files.readString(config)
            val graphExists = Files.walk(modelRoot).use { paths ->
                paths.anyMatch { Files.isRegularFile(it) && it.fileName.toString().endsWith(".onnx", ignoreCase = true) }
            }
            if (!graphExists) throw IllegalArgumentException("هیچ فایل گراف ONNX در کنار پیکربندی مدل پیدا نشد.")

            val tokenizerExists = Files.walk(modelRoot).use { paths ->
                paths.anyMatch {
                    if (!Files.isRegularFile(it)) false else {
                        val name = it.fileName.toString().lowercase()
                        name == "tokenizer.json" || name == "tokenizer.model" ||
                            name == "tokenizer_config.json" || name == "vocab.json"
                    }
                }
            }
            if (!tokenizerExists) throw IllegalArgumentException("فایل‌های tokenizer پیدا نشدند؛ بستهٔ مدل ناقص است.")

            val modelType = Regex("""["']type["']\\s*:\\s*["']([^"']+)["']""")
                .find(configText)?.groupValues?.get(1)
            val context = Regex("""["']context_length["']\\s*:\\s*(\\d+)""")
                .find(configText)?.groupValues?.get(1)?.toLongOrNull()
            val displayName = modelType?.replaceFirstChar { it.uppercase() }
                ?: "ONNX Runtime GenAI model"

            // Store the model package itself at a stable path; the archive's optional
            // single top-level directory is flattened without moving files individually.
            if (modelRoot == unpackRoot) {
                Files.move(unpackRoot, target, StandardCopyOption.ATOMIC_MOVE)
            } else {
                Files.move(modelRoot, target, StandardCopyOption.ATOMIC_MOVE)
                deleteRecursively(unpackRoot)
            }
            val storedSize = directorySize(target)
            return ModelResult.Success(
                ModelDescriptor(
                    id = id,
                    displayName = displayName,
                    path = target,
                    format = ModelFormat.ONNX_GENAI,
                    quantization = Quantization.UNKNOWN,
                    sizeBytes = storedSize,
                    metadata = ModelMetadata(
                        architecture = modelType,
                        name = displayName,
                        contextLength = context,
                        raw = mapOf("runtime" to "onnxruntime-genai", "config" to "genai_config.json"),
                    ),
                    validation = ValidationStatus.VALID,
                    runtimeCompatibility = RuntimeCompatibility(true, null),
                    state = ModelState.READY,
                )
            )
        } catch (t: Throwable) {
            deleteRecursively(unpackRoot)
            deleteRecursively(target)
            val message = t.message ?: "وارد کردن بستهٔ ONNX Runtime GenAI ناموفق بود."
            return ModelResult.Failure(
                if (t is java.nio.file.AccessDeniedException) ModelError.Storage(message, t)
                else ModelError.InvalidModel(message),
            )
        }
    }

    private fun directorySize(root: Path): Long = Files.walk(root).use { paths ->
        paths.filter { Files.isRegularFile(it) }.mapToLong { Files.size(it) }.sum()
    }

    private fun deleteRecursively(path: Path) {
        if (!Files.exists(path)) return
        Files.walk(path).use { paths ->
            paths.sorted(Comparator.reverseOrder()).forEach { runCatching { Files.deleteIfExists(it) } }
        }
    }
}
