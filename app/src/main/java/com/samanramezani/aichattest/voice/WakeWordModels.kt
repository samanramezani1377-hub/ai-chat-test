package com.samanramezani.aichattest.voice

import android.content.ContentResolver
import android.net.Uri
import android.provider.OpenableColumns
import java.io.File
import java.io.FileOutputStream
import java.util.zip.ZipInputStream

data class WakeWordModelEntry(
    val id: String,
    val title: String,
    val directory: File,
    val encoder: File,
    val decoder: File,
    val joiner: File,
    val tokens: File,
    val keywords: File,
)

/**
 * Imports a complete sherpa-onnx streaming KWS package configured for the Persian wake phrase
 * "دینا". The archive must contain encoder/decoder/joiner ONNX files, tokens.txt, and
 * keywords.txt with the original keyword label "دینا". The package's tokenizer/model must
 * actually support Persian; filename validation alone cannot guarantee recognition quality.
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
        require(displayName.endsWith(".zip", true)) {
            "VOICE-KWS-IMPORT-001: بستهٔ تشخیص کلمه باید ZIP باشد و پنج فایل مدل/واژگان را در خود داشته باشد."
        }
        val staging = File(root, ".import-${System.currentTimeMillis()}")
        check(staging.mkdirs()) { "VOICE-KWS-IMPORT-002: پوشهٔ موقت مدل بیدارباش ساخته نشد." }
        try {
            val input = resolver.openInputStream(uri)
                ?: error("VOICE-KWS-IMPORT-003: فایل انتخاب‌شده قابل خواندن نیست.")
            var count = 0
            var expanded = 0L
            input.use { source ->
                ZipInputStream(source).use { zip ->
                    while (true) {
                        val item = zip.nextEntry ?: break
                        count++
                        require(count <= MAX_ENTRIES) { "VOICE-KWS-IMPORT-004: تعداد فایل‌های بسته بیش از حد مجاز است." }
                        val name = item.name.replace('\\\\', '/')
                        require(!name.startsWith("/") && name.split('/').none { it == ".." }) {
                            "VOICE-KWS-IMPORT-005: مسیر نامعتبر داخل بستهٔ مدل وجود دارد."
                        }
                        val target = File(staging, name).canonicalFile
                        require(target.toPath().startsWith(staging.canonicalFile.toPath())) {
                            "VOICE-KWS-IMPORT-005: مسیر نامعتبر داخل بستهٔ مدل وجود دارد."
                        }
                        if (item.isDirectory) {
                            target.mkdirs()
                        } else {
                            target.parentFile?.mkdirs()
                            FileOutputStream(target).use { output ->
                                val buffer = ByteArray(64 * 1024)
                                while (true) {
                                    val n = zip.read(buffer)
                                    if (n < 0) break
                                    expanded += n
                                    require(expanded <= MAX_EXPANDED_BYTES) {
                                        "VOICE-KWS-IMPORT-006: حجم استخراج‌شدهٔ مدل از حد مجاز بیشتر است."
                                    }
                                    output.write(buffer, 0, n)
                                }
                            }
                        }
                        zip.closeEntry()
                    }
                }
            }
            val candidate = staging.walkTopDown().filter { it.isDirectory }
                .firstOrNull { validateDirectory(it) != null }
                ?: error("VOICE-KWS-IMPORT-007: بسته معتبر نیست. encoder/decoder/joiner، tokens.txt و keywords.txt شامل عبارت «دینا» لازم‌اند.")
            val id = "wake-dina-${System.currentTimeMillis()}"
            val destination = File(root, id)
            check(candidate.renameTo(destination)) { "VOICE-KWS-IMPORT-008: ذخیرهٔ مدل بیدارباش ناموفق بود." }
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
        val files = directory.walkTopDown().filter { it.isFile }.toList()
        val encoder = files.firstOrNull { it.name.contains("encoder", true) && it.extension.equals("onnx", true) } ?: return null
        val decoder = files.firstOrNull { it.name.contains("decoder", true) && it.extension.equals("onnx", true) } ?: return null
        val joiner = files.firstOrNull { it.name.contains("joiner", true) && it.extension.equals("onnx", true) } ?: return null
        val tokens = files.firstOrNull { it.name.equals("tokens.txt", true) } ?: return null
        val keywords = files.firstOrNull { it.name.equals("keywords.txt", true) } ?: return null
        if (listOf(encoder, decoder, joiner).any { it.length() < 1024L } || tokens.length() == 0L) return null
        val keywordText = runCatching { keywords.readText(Charsets.UTF_8) }.getOrNull() ?: return null
        if (!keywordText.contains("دینا")) return null
        return WakeWordModelEntry(
            id = directory.name,
            title = "بیدارباش «دینا» · sherpa-onnx KWS",
            directory = directory,
            encoder = encoder,
            decoder = decoder,
            joiner = joiner,
            tokens = tokens,
            keywords = keywords,
        )
    }

    companion object {
        private const val MAX_ENTRIES = 4_000
        private const val MAX_EXPANDED_BYTES = 768L * 1024 * 1024
    }
}
