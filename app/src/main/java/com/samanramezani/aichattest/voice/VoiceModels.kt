package com.samanramezani.aichattest.voice

import android.content.ContentResolver
import android.net.Uri
import android.provider.OpenableColumns
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.util.zip.ZipInputStream
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream

enum class VoiceModelKind { STT, TTS }

data class VoiceModelEntry(
    val id: String,
    val title: String,
    val kind: VoiceModelKind,
    val directory: File,
    val modelFile: File,
    val tokensFile: File? = null,
    val dataDirectory: File? = null,
    val encoder: File? = null,
    val decoder: File? = null,
    val joiner: File? = null,
)

/**
 * Model files are user-owned and remain on-device. Arbitrary ONNX files are never guessed
 * to be compatible. Import validated .zip packages, not a bare model file.
 *
 * TTS: converted sherpa-onnx Piper/VITS bundle with *.onnx, tokens.txt, espeak-ng-data/.
 * STT: sherpa-onnx online transducer bundle with encoder/decoder/joiner ONNX and tokens.txt.
 *
 * The original rhasspy/piper-voices ONNX needs metadata/tokens conversion before sherpa use.
 */
class VoiceModelStore(
    private val resolver: ContentResolver,
    private val root: File,
) {
    init { root.mkdirs() }

    fun list(kind: VoiceModelKind): List<VoiceModelEntry> =
        root.listFiles()?.filter { it.isDirectory && !it.name.startsWith(".") }
            ?.mapNotNull { validateDirectory(it, kind) }?.sortedBy { it.title } ?: emptyList()

    fun importArchive(uri: Uri, kind: VoiceModelKind): VoiceModelEntry {
        root.mkdirs()
        val staging = File(root, ".import-${System.currentTimeMillis()}")
        if (!staging.mkdirs()) error("VOICE-IMPORT-001: ایجاد پوشه موقت مدل ناموفق بود.")
        try {
            val archiveName = resolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
                ?.use { cursor -> if (cursor.moveToFirst()) cursor.getString(0).orEmpty() else "" }.orEmpty()
            val source = resolver.openInputStream(uri)
                ?: error("VOICE-IMPORT-002: فایل انتخاب‌شده قابل خواندن نیست.")
            var count = 0
            var expandedBytes = 0L

            fun extract(nameValue: String, isDirectory: Boolean, entryStream: InputStream) {
                count++
                require(count <= MAX_ENTRIES) { "VOICE-IMPORT-003: تعداد فایل‌های بسته مدل بیش از حد مجاز است." }
                val name = nameValue.replace('\\\\', '/')
                require(!name.startsWith("/") && name.split('/').none { it == ".." }) {
                    "VOICE-IMPORT-004: مسیر نامعتبر داخل بسته مدل وجود دارد."
                }
                val target = File(staging, name).canonicalFile
                require(target.toPath().startsWith(staging.canonicalFile.toPath())) {
                    "VOICE-IMPORT-004: مسیر نامعتبر داخل بسته مدل وجود دارد."
                }
                if (isDirectory) {
                    target.mkdirs()
                } else {
                    target.parentFile?.mkdirs()
                    FileOutputStream(target).use { out ->
                        val buffer = ByteArray(64 * 1024)
                        while (true) {
                            val n = entryStream.read(buffer)
                            if (n < 0) break
                            expandedBytes += n
                            require(expandedBytes <= MAX_EXPANDED_BYTES) {
                                "VOICE-IMPORT-005: حجم استخراج‌شده بسته از حد مجاز بیشتر است."
                            }
                            out.write(buffer, 0, n)
                        }
                    }
                }
            }

            source.use { stream ->
                if (archiveName.endsWith(".tar.bz2", ignoreCase = true) || archiveName.endsWith(".tbz2", ignoreCase = true)) {
                    TarArchiveInputStream(BZip2CompressorInputStream(stream)).use { tar ->
                        while (true) {
                            val item = tar.nextTarEntry ?: break
                            extract(item.name, item.isDirectory, tar)
                        }
                    }
                } else {
                    ZipInputStream(stream).use { zip ->
                        while (true) {
                            val item = zip.nextEntry ?: break
                            extract(item.name, item.isDirectory, zip)
                            zip.closeEntry()
                        }
                    }
                }
            }

            val candidate = findCandidateRoot(staging, kind)
                ?: error(if (kind == VoiceModelKind.TTS)
                    "VOICE-TTS-001: بسته TTS معتبر نیست. بستهٔ تبدیل‌شدهٔ Piper/VITS همراه tokens.txt و پوشه espeak-ng-data لازم است."
                else
                    "VOICE-STT-001: بسته STT معتبر نیست. فایل‌های encoder/decoder/joiner و tokens.txt لازم است.")
            val id = "${kind.name.lowercase()}-${candidate.name.replace(Regex("[^A-Za-z0-9._-]"), "_")}-${System.currentTimeMillis()}"
            val destination = File(root, id)
            check(staging.renameTo(destination)) { "VOICE-IMPORT-006: ذخیره نهایی مدل ناموفق بود." }
            return validateDirectory(destination, kind)
                ?: run { destination.deleteRecursively(); error("VOICE-IMPORT-007: اعتبارسنجی مدل پس از وارد کردن ناموفق بود.") }
        } catch (t: Throwable) {
            staging.deleteRecursively()
            throw t
        }
    }

    private fun findCandidateRoot(staging: File, kind: VoiceModelKind): File? =
        staging.walkTopDown().filter { it.isDirectory }.firstOrNull { validateDirectory(it, kind) != null }

    private fun validateDirectory(directory: File, kind: VoiceModelKind): VoiceModelEntry? {
        val files = directory.walkTopDown().filter { it.isFile }.toList()
        val tokens = files.firstOrNull { it.name.equals("tokens.txt", true) } ?: return null
        return when (kind) {
            VoiceModelKind.TTS -> {
                val model = files.firstOrNull {
                    it.extension.equals("onnx", true) &&
                        !it.name.contains("encoder", true) &&
                        !it.name.contains("decoder", true) &&
                        !it.name.contains("joiner", true)
                } ?: return null
                val data = directory.walkTopDown().firstOrNull { it.isDirectory && it.name == "espeak-ng-data" }
                    ?: return null
                if (data.list()?.isEmpty() != false) return null
                VoiceModelEntry(directory.name, model.nameWithoutExtension, kind, directory, model, tokensFile = tokens, dataDirectory = data)
            }
            VoiceModelKind.STT -> {
                val encoder = files.firstOrNull { it.name.contains("encoder", true) && it.extension.equals("onnx", true) } ?: return null
                val decoder = files.firstOrNull { it.name.contains("decoder", true) && it.extension.equals("onnx", true) } ?: return null
                val joiner = files.firstOrNull { it.name.contains("joiner", true) && it.extension.equals("onnx", true) } ?: return null
                VoiceModelEntry(directory.name, directory.name, kind, directory, encoder, tokensFile = tokens, encoder = encoder, decoder = decoder, joiner = joiner)
            }
        }
    }

    companion object {
        private const val MAX_ENTRIES = 20_000
        private const val MAX_EXPANDED_BYTES = 2L * 1024 * 1024 * 1024
    }
}
