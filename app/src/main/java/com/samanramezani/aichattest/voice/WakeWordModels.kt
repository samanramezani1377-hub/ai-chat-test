package com.samanramezani.aichattest.voice

import android.content.ContentResolver
import android.net.Uri
import android.provider.OpenableColumns
import java.io.File
import java.io.FileOutputStream
import java.util.zip.ZipInputStream
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream
import org.json.JSONObject

data class WakeWordModelEntry(
    val id: String,
    val title: String,
    val directory: File,
    val encoder: File,
    val decoder: File,
    val joiner: File,
    val tokens: File,
    val keywords: File,
    val modelType: String,
    val modelingUnit: String,
    val bpeVocab: File?,
)

/**
 * Imports a complete sherpa-onnx streaming KWS package configured for the Persian wake phrase
 * "دینا". The archive must contain encoder/decoder/joiner ONNX files, tokens.txt, and
 * keywords.txt with the original keyword label "دینا". Optional wakeword.json may set
 * modelType and modelingUnit; bpe.model is detected automatically. The package's tokenizer/model
 * must actually support Persian; filename validation alone cannot guarantee recognition quality.
 */
class WakeWordModelStore(
    private val resolver: ContentResolver,
    private val root: File,
) {
    init { root.mkdirs() }

    fun list(): List<WakeWordModelEntry> =
        root.listFiles()?.filter { it.isDirectory && !it.name.startsWith(".") }
            ?.mapNotNull(::validateDirectory)?.sortedBy { it.title } ?: emptyList()

    fun importArchive(uri: Uri): WakeWordModelEntry {
        root.mkdirs()
        val displayName = resolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
            ?.use { cursor -> if (cursor.moveToFirst()) cursor.getString(0).orEmpty() else "" }.orEmpty()
        val archive = displayName.lowercase()
        require(archive.endsWith(".zip") || archive.endsWith(".tar.bz2") || archive.endsWith(".tbz2")) {
            "VOICE-KWS-IMPORT-001: بستهٔ مدل باید ZIP یا TAR.BZ2 باشد."
        }
        val staging = File(root, ".import-${System.currentTimeMillis()}")
        check(staging.mkdirs()) { "VOICE-KWS-IMPORT-002: پوشهٔ موقت مدل بیدارباش ساخته نشد." }
        var count = 0
        var expanded = 0L
        fun extract(nameValue: String, isDirectory: Boolean, entryStream: java.io.InputStream) {
            count++
            require(count <= MAX_ENTRIES) { "VOICE-KWS-IMPORT-004: تعداد فایل‌های بسته بیش از حد مجاز است." }
            val name = nameValue.replace('\\', '/')
            require(!name.startsWith("/") && name.split('/').none { it == ".." }) {
                "VOICE-KWS-IMPORT-005: مسیر نامعتبر داخل بستهٔ مدل وجود دارد."
            }
            val target = File(staging, name).canonicalFile
            require(target.toPath().startsWith(staging.canonicalFile.toPath())) {
                "VOICE-KWS-IMPORT-005: مسیر نامعتبر داخل بستهٔ مدل وجود دارد."
            }
            if (isDirectory) {
                target.mkdirs()
            } else {
                target.parentFile?.mkdirs()
                FileOutputStream(target).use { output ->
                    val buffer = ByteArray(64 * 1024)
                    while (true) {
                        val n = entryStream.read(buffer)
                        if (n < 0) break
                        expanded += n
                        require(expanded <= MAX_EXPANDED_BYTES) {
                            "VOICE-KWS-IMPORT-006: حجم استخراج‌شدهٔ مدل از حد مجاز بیشتر است."
                        }
                        output.write(buffer, 0, n)
                    }
                }
            }
        }
        try {
            val input = resolver.openInputStream(uri)
                ?: error("VOICE-KWS-IMPORT-003: فایل انتخاب‌شده قابل خواندن نیست.")
            input.use { source ->
                if (archive.endsWith(".zip")) {
                    ZipInputStream(source).use { zip ->
                        while (true) {
                            val item = zip.nextEntry ?: break
                            extract(item.name, item.isDirectory, zip)
                            zip.closeEntry()
                        }
                    }
                } else {
                    TarArchiveInputStream(BZip2CompressorInputStream(source)).use { tar ->
                        while (true) {
                            val item = tar.nextTarEntry ?: break
                            extract(item.name, item.isDirectory, tar)
                        }
                    }
                }
            }

            // The official tiny zh-en KWS model has an English phoneme inventory but no Persian
            // keyword file. For that specific model family, seed a phonetic approximation of
            // Persian "دینا" (DEE-NAA); custom-trained Persian packages should ship their own
            // keywords.txt and are never overwritten when they already define Dina.
            staging.walkTopDown().filter { it.isDirectory }.forEach { directory ->
                val directFiles = directory.listFiles()?.filter { it.isFile }.orEmpty()
                val hasKwsParts = directFiles.any { it.name.contains("encoder", true) && it.extension.equals("onnx", true) } &&
                    directFiles.any { it.name.contains("decoder", true) && it.extension.equals("onnx", true) } &&
                    directFiles.any { it.name.contains("joiner", true) && it.extension.equals("onnx", true) } &&
                    directFiles.any { it.name.equals("tokens.txt", true) }
                val supportsEnglishPhones = directory.name.contains("zh-en-3m", true) ||
                    directFiles.any { it.name.equals("en.phone", true) }
                val keywordFile = File(directory, "keywords.txt")
                val alreadyHasDina = keywordFile.isFile && runCatching {
                    keywordFile.readText(Charsets.UTF_8).contains("دینا")
                }.getOrDefault(false)
                if (hasKwsParts && supportsEnglishPhones && !alreadyHasDina) {
                    keywordFile.writeText("D IY1 N AA1 @دینا\n", Charsets.UTF_8)
                }
            }

            val candidate = staging.walkTopDown().filter { it.isDirectory }
                .firstOrNull { validateDirectory(it) != null }
                ?: error("VOICE-KWS-IMPORT-007: بسته معتبر نیست. encoder/decoder/joiner، tokens.txt و keywords.txt برای «دینا» لازم‌اند.")
            val destination = File(root, "wake-dina-${System.currentTimeMillis()}")
            if (candidate == staging) {
                check(destination.mkdirs()) { "VOICE-KWS-IMPORT-008: پوشهٔ نهایی مدل بیدارباش ساخته نشد." }
                candidate.listFiles().orEmpty().forEach { child ->
                    check(child.renameTo(File(destination, child.name))) {
                        "VOICE-KWS-IMPORT-008: انتقال فایل ${child.name} به پوشهٔ نهایی ناموفق بود."
                    }
                }
            } else {
                check(candidate.renameTo(destination)) { "VOICE-KWS-IMPORT-008: ذخیرهٔ مدل بیدارباش ناموفق بود." }
            }
            staging.deleteRecursively()
            return validateDirectory(destination)
                ?: run {
                    destination.deleteRecursively()
                    error("VOICE-KWS-IMPORT-009: اعتبارسنجی نهایی مدل بیدارباش ناموفق بود.")
                }
        } catch (t: Throwable) {
            staging.deleteRecursively()
            throw t
        }
    }

    private fun validateDirectory(directory: File): WakeWordModelEntry? {
        if (!directory.isDirectory) return null
        val files = directory.listFiles()?.filter { it.isFile }.orEmpty()
        val encoder = files.firstOrNull { it.name.contains("encoder", true) && it.extension.equals("onnx", true) } ?: return null
        val decoder = files.firstOrNull { it.name.contains("decoder", true) && it.extension.equals("onnx", true) } ?: return null
        val joiner = files.firstOrNull { it.name.contains("joiner", true) && it.extension.equals("onnx", true) } ?: return null
        val tokens = files.firstOrNull { it.name.equals("tokens.txt", true) } ?: return null
        val keywords = files.firstOrNull { it.name.equals("keywords.txt", true) } ?: return null
        if (listOf(encoder, decoder, joiner).any { it.length() < 1024L } || tokens.length() == 0L) return null
        val keywordText = runCatching { keywords.readText(Charsets.UTF_8) }.getOrNull() ?: return null
        if (!keywordText.contains("دینا")) return null
        val configFile = files.firstOrNull { it.name.equals("wakeword.json", true) }
        val config = configFile?.let { runCatching { JSONObject(it.readText(Charsets.UTF_8)) }.getOrNull() }
        val bpeVocab = files.firstOrNull { it.name.equals("bpe.model", true) }
        val inferredModelType = when {
            encoder.name.contains("zipformer2", true) || encoder.name.contains("chunk-16-left", true) -> "zipformer2"
            encoder.name.contains("lstm", true) -> "lstm"
            else -> "zipformer"
        }
        val modelType = config?.optString("modelType")?.takeIf { it in setOf("zipformer", "zipformer2", "lstm", "conformer") }
            ?: inferredModelType
        val modelingUnit = config?.optString("modelingUnit")?.takeIf { it.isNotBlank() }
            ?: if (bpeVocab != null) "bpe" else ""
        return WakeWordModelEntry(
            id = directory.name,
            title = "بیدارباش «دینا» · sherpa-onnx KWS",
            directory = directory,
            encoder = encoder,
            decoder = decoder,
            joiner = joiner,
            tokens = tokens,
            keywords = keywords,
            modelType = modelType,
            modelingUnit = modelingUnit,
            bpeVocab = bpeVocab,
        )
    }

    companion object {
        private const val MAX_ENTRIES = 4_000
        private const val MAX_EXPANDED_BYTES = 768L * 1024 * 1024
    }
}
