package com.samanramezani.aichattest.voice

import android.content.ContentResolver
import android.content.ContentValues
import android.net.Uri
import android.os.Environment
import android.provider.MediaStore
import android.provider.OpenableColumns
import org.json.JSONObject
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.util.zip.ZipEntry
import java.util.zip.ZipInputStream
import java.util.zip.ZipOutputStream

/** Stores the complete Pocket TTS Farsi v2 ONNX pack. This is not a Piper model. */
internal class PocketTtsFarsiBundleStore(private val resolver: ContentResolver, private val root: File) {
    companion object {
        private const val REPO = "Nimaone/pocket-tts-farsi-v2-onnx"
        private const val ZIP_NAME = "Pocket-TTS-Farsi-v2-ONNX.zip"
        private const val MAX_ENTRIES = 32
        private const val MAX_EXPANDED = 700L * 1024 * 1024
        private val required = linkedMapOf(
            "decode_state_init.npz" to 1_000_000L,
            "flow_lm_step.onnx" to 100_000L,
            "flow_lm_step.onnx.data" to 100_000_000L,
            "g2p_decoder.onnx" to 100_000L,
            "g2p_decoder.onnx.data" to 1_000_000L,
            "g2p_encoder.onnx" to 1_000_000L,
            "manifest.json" to 100L,
            // Required by the reference runtime: phoneme token IDs are looked up
            // through this SentencePiece model, and synthesis needs a reference
            // recording to initialize the speaker-conditioning KV cache.
            "male_hello.wav" to 100_000L,
            "normalize_fa.py" to 1_000L,
            "mimi_decoder_step_kv.onnx" to 100_000L,
            "mimi_decoder_step_kv.onnx.data" to 10_000_000L,
            "mimi_encoder.onnx" to 100_000L,
            "mimi_encoder.onnx.data" to 10_000_000L,
            "tokenizer_ph.model" to 100_000L,
            "weights.npz" to 1_000_000L,
        )
        private val sourceUrls = mapOf(
            "normalize_fa.py" to "https://huggingface.co/mehdi-hf/pocket-tts-farsi-v2/resolve/main/normalize_fa.py",
            "male_hello.wav" to "https://raw.githubusercontent.com/Nimaone/persian_tts/main/voices/male_hello.wav",
            "tokenizer_ph.model" to "https://huggingface.co/mehdi-hf/pocket-tts-farsi-v2/resolve/main/tokenizer_ph.model?download=true",
        )
    }

    fun isReady(): Boolean = validateDirectory(root).isEmpty()
    fun status(): String = when {
        isReady() -> "گراف‌های ONNX، توکنایزر، نرمال‌ساز فارسی و صدای مرجع نصب شده‌اند؛ موتور اجرا هنوز به مکالمه متصل نشده است."
        !root.exists() -> "بستهٔ Pocket TTS Farsi v2 هنوز وارد نشده است."
        else -> "بسته ناقص است: ${validateDirectory(root).joinToString("، ")}"
    }

    /** Downloads every required artifact into one ZIP in Downloads. Files stream to disk. */
    fun downloadArchive(onProgress: (String) -> Unit = {}): String {
        val temp = File(root.parentFile ?: root, ".pocket-tts-download.part")
        temp.parentFile?.mkdirs()
        temp.delete()
        try {
            ZipOutputStream(BufferedOutputStream(FileOutputStream(temp), 256 * 1024)).use { zip ->
                required.keys.forEachIndexed { index, path ->
                    onProgress("دریافت Pocket TTS — فایل ${index + 1} از ${required.size}: $path")
                    val sourceUrl = sourceUrls[path]
                        ?: "https://huggingface.co/$REPO/resolve/main/$path?download=true"
                    val conn = (URL(sourceUrl).openConnection() as HttpURLConnection).apply {
                        connectTimeout = 30_000
                        readTimeout = 180_000
                        instanceFollowRedirects = true
                        useCaches = false
                        setRequestProperty("User-Agent", "AI-Chat-Test-Android/1.0")
                    }
                    try {
                        val status = conn.responseCode
                        require(status == 200) { "HTTP $status هنگام دریافت $path" }
                        val expected = conn.contentLengthLong
                        var received = 0L
                        zip.putNextEntry(ZipEntry(path))
                        BufferedInputStream(conn.inputStream, 256 * 1024).use { input ->
                            val buffer = ByteArray(128 * 1024)
                            while (true) {
                                val n = input.read(buffer)
                                if (n < 0) break
                                zip.write(buffer, 0, n)
                                received += n
                                if (expected > 0 && received % (16L * 1024 * 1024) < n) {
                                    onProgress("$path: ${(received * 100 / expected).coerceIn(0, 100)}٪")
                                }
                            }
                        }
                        zip.closeEntry()
                        require(received >= required.getValue(path) && (expected <= 0 || received == expected)) {
                            "فایل $path ناقص است: $received/$expected بایت."
                        }
                    } finally { conn.disconnect() }
                }
            }
            val zipErrors = validateZip(temp)
            require(zipErrors.isEmpty()) { "اعتبارسنجی ZIP ناموفق بود: ${zipErrors.joinToString()}" }
            val values = ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, ZIP_NAME)
                put(MediaStore.MediaColumns.MIME_TYPE, "application/zip")
                put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS)
                put(MediaStore.MediaColumns.IS_PENDING, 1)
            }
            val uri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
                ?: error("ساخت فایل در پوشه Downloads ناموفق بود.")
            try {
                resolver.openOutputStream(uri, "w")?.use { output ->
                    temp.inputStream().buffered(256 * 1024).use { it.copyTo(output, 256 * 1024) }
                } ?: error("فایل مقصد قابل نوشتن نیست.")
                resolver.update(uri, ContentValues().apply {
                    put(MediaStore.MediaColumns.IS_PENDING, 0)
                }, null, null)
            } catch (t: Throwable) {
                resolver.delete(uri, null, null)
                throw t
            }
            return "دانلود کامل شد: Downloads/$ZIP_NAME. برای نصب، همین ZIP را با دکمهٔ وارد کردن انتخاب کن."
        } catch (t: Throwable) {
            throw IllegalStateException("POCKET-TTS-DOWNLOAD-001: ${t.message ?: t.javaClass.simpleName}", t)
        } finally { temp.delete() }
    }

    fun importArchive(uri: Uri, onProgress: (String) -> Unit = {}): String {
        val name = resolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
            ?.use { if (it.moveToFirst()) it.getString(0).orEmpty() else "" }.orEmpty()
        require(name.endsWith(".zip", true)) { "POCKET-TTS-IMPORT-001: بسته ZIP کامل را انتخاب کن؛ فایل ONNX تکی کافی نیست." }
        val stage = File(root.parentFile ?: root, ".pocket-tts-stage-${System.currentTimeMillis()}")
        stage.deleteRecursively()
        check(stage.mkdirs()) { "POCKET-TTS-IMPORT-002: ساخت پوشه موقت ناموفق بود." }
        try {
            val stream = resolver.openInputStream(uri) ?: error("POCKET-TTS-IMPORT-003: فایل قابل خواندن نیست.")
            stream.use { source ->
                ZipInputStream(BufferedInputStream(source, 128 * 1024)).use { zip ->
                    var entries = 0
                    var expanded = 0L
                    while (true) {
                        val entry = zip.nextEntry ?: break
                        entries++
                        require(entries <= MAX_ENTRIES) { "POCKET-TTS-IMPORT-004: تعداد فایل‌های ZIP بیش از حد مجاز است." }
                        val path = entry.name.replace('\\', '/').removePrefix("./")
                        require(!path.startsWith("/") && path.split('/').none { it == ".." || it.isBlank() }) {
                            "POCKET-TTS-IMPORT-005: مسیر ناامن داخل ZIP."
                        }
                        if (!entry.isDirectory) {
                            require(path in required) { "POCKET-TTS-IMPORT-006: فایل غیرمنتظره: $path" }
                            val target = File(stage, path).canonicalFile
                            require(target.toPath().startsWith(stage.canonicalFile.toPath())) {
                                "POCKET-TTS-IMPORT-005: مسیر ناامن داخل ZIP."
                            }
                            FileOutputStream(target).buffered(128 * 1024).use { output ->
                                val buffer = ByteArray(128 * 1024)
                                while (true) {
                                    val n = zip.read(buffer)
                                    if (n < 0) break
                                    expanded += n
                                    require(expanded <= MAX_EXPANDED) { "POCKET-TTS-IMPORT-007: حجم استخراج‌شده بیش از ۷۰۰ مگابایت است." }
                                    output.write(buffer, 0, n)
                                }
                            }
                            onProgress("اعتبارسنجی فایل $path")
                        }
                        zip.closeEntry()
                    }
                }
            }
            val missing = validateDirectory(stage)
            require(missing.isEmpty()) { "POCKET-TTS-IMPORT-008: فایل‌های ناقص: ${missing.joinToString("، ")}" }
            require(runCatching { JSONObject(File(stage, "manifest.json").readText()).length() > 0 }.getOrDefault(false)) {
                "POCKET-TTS-IMPORT-009: manifest.json نامعتبر است."
            }
            val backup = File(root.parentFile, ".pocket-tts-backup-${System.currentTimeMillis()}")
            val hadOld = root.exists()
            if (hadOld) check(root.renameTo(backup)) { "POCKET-TTS-IMPORT-010: نسخه قبلی کنار گذاشته نشد." }
            if (!stage.renameTo(root)) {
                if (hadOld) backup.renameTo(root)
                error("POCKET-TTS-IMPORT-011: نصب نهایی ناموفق بود.")
            }
            backup.deleteRecursively()
            return "بستهٔ Pocket TTS Farsi v2، توکنایزر، نرمال‌ساز فارسی و صدای مرجع وارد و اعتبارسنجی شدند؛ اتصال موتور اجرا به مکالمه هنوز کامل نشده است."
        } catch (t: Throwable) {
            stage.deleteRecursively()
            throw t
        }
    }

    private fun validateZip(file: File): List<String> {
        val seen = mutableSetOf<String>()
        return try {
            ZipInputStream(file.inputStream().buffered(128 * 1024)).use { zip ->
                while (true) {
                    val e = zip.nextEntry ?: break
                    if (!e.isDirectory) {
                        if (e.name !in required) return listOf("فایل غیرمنتظره: ${e.name}")
                        var size = 0L
                        val buffer = ByteArray(64 * 1024)
                        while (true) { val n = zip.read(buffer); if (n < 0) break; size += n }
                        if (size < required.getValue(e.name)) return listOf("فایل ناقص: ${e.name}")
                        seen += e.name
                    }
                    zip.closeEntry()
                }
            }
            (required.keys - seen).toList()
        } catch (t: Throwable) { listOf(t.message ?: "ZIP خراب") }
    }

    private fun validateDirectory(dir: File): List<String> {
        val missing = required.filter { (name, min) ->
            val file = File(dir, name)
            !file.isFile || file.length() < min
        }.keys.toMutableList()
        val manifest = File(dir, "manifest.json")
        if (manifest.isFile && runCatching { JSONObject(manifest.readText()) }.isFailure) {
            missing += "manifest.json (JSON نامعتبر)"
        }
        return missing.distinct()
    }
}
