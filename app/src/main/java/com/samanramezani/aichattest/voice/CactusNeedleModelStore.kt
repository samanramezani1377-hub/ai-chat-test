package com.samanramezani.aichattest.voice

import android.content.ContentValues
import android.content.Context
import android.net.Uri
import android.provider.MediaStore
import android.provider.OpenableColumns
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

/** Keeps the active .cact file private and publishes downloads in public Downloads. */
internal class CactusNeedleModelStore(
    private val context: Context,
    private val root: File,
) {
    companion object {
        const val MODEL_FILE = "needle3.cact"
        const val MODEL_BYTES = 35_335_380L
        const val MODEL_REVISION = "d6c7d6a2e9655505e0f8e23339356dd1705900eb"
        private const val MODEL_SHA256_PREFIX = "c9d915ec"
        private const val MODEL_SHA256_SUFFIX = "170c38"
        private const val MAX_IMPORT_BYTES = 128L * 1024L * 1024L
        private const val MIN_MODEL_BYTES = 1L * 1024L * 1024L
        private const val MODEL_URL =
            "https://huggingface.co/Cactus-Compute/needle3/resolve/$MODEL_REVISION/needle3.cact"
        private const val DOWNLOAD_RELATIVE_PATH = "Download/AI Chat Test/Cactus"
    }

    val modelFile: File get() = File(root, MODEL_FILE)

    init {
        if (!root.exists()) root.mkdirs()
        // The previous app-managed Laya weights are obsolete after this migration.
        root.parentFile?.resolve("laya-int4")?.deleteRecursively()
    }

    fun isReady(): Boolean = modelFile.isFile &&
        modelFile.length() in MIN_MODEL_BYTES..MAX_IMPORT_BYTES

    fun status(): String = if (isReady()) {
        "مدل Cactus Needle 3 آماده است (${modelFile.length() / (1024 * 1024)} مگابایت)."
    } else {
        "مدل Cactus Needle 3 هنوز نصب نشده است؛ آن را دانلود یا از حافظه گوشی وارد کن."
    }

    fun importModel(uri: Uri): String {
        root.mkdirs()
        val displayName = context.contentResolver.query(
            uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null
        )?.use { cursor ->
            if (cursor.moveToFirst()) cursor.getString(0).orEmpty() else ""
        }.orEmpty()
        require(displayName.endsWith(".cact", ignoreCase = true)) {
            "CACTUS-IMPORT-001: فایل انتخاب‌شده باید پسوند .cact داشته باشد."
        }
        val input = context.contentResolver.openInputStream(uri)
            ?: error("CACTUS-IMPORT-002: فایل انتخاب‌شده قابل خواندن نیست.")
        val temporary = File(root, ".$MODEL_FILE.importing")
        temporary.delete()
        try {
            input.use { source ->
                FileOutputStream(temporary).use { target ->
                    val buffer = ByteArray(64 * 1024)
                    var total = 0L
                    while (true) {
                        val count = source.read(buffer)
                        if (count < 0) break
                        total += count
                        require(total <= MAX_IMPORT_BYTES) {
                            "CACTUS-IMPORT-003: حجم فایل از سقف ۱۲۸ مگابایت بیشتر است."
                        }
                        target.write(buffer, 0, count)
                    }
                    target.fd.sync()
                    require(total >= MIN_MODEL_BYTES) {
                        "CACTUS-IMPORT-004: فایل .cact ناقص یا بیش از حد کوچک است."
                    }
                }
            }
            replaceModel(temporary)
            return "فایل .cact وارد شد. هنگام اولین استفاده، موتور Needle سازگاری آن را بررسی و بارگذاری می‌کند."
        } catch (failure: Throwable) {
            temporary.delete()
            throw failure
        }
    }

    fun downloadOfficialModel(onProgress: (String) -> Unit): String {
        root.mkdirs()
        val temporary = File(root, ".$MODEL_FILE.download")
        temporary.delete()
        val connection = (URL(MODEL_URL).openConnection() as HttpURLConnection).apply {
            connectTimeout = 20_000
            readTimeout = 60_000
            requestMethod = "GET"
            instanceFollowRedirects = true
            setRequestProperty("User-Agent", "AIChatTest-Android/Needle3")
        }
        try {
            connection.connect()
            require(connection.responseCode == HttpURLConnection.HTTP_OK) {
                "CACTUS-DOWNLOAD-001: دریافت فایل از Hugging Face ناموفق بود (HTTP ${connection.responseCode})."
            }
            val announcedLength = connection.contentLengthLong
            require(announcedLength <= 0L || announcedLength == MODEL_BYTES) {
                "CACTUS-DOWNLOAD-002: اندازه فایل سرور با نسخه ثابت‌شده Needle 3 سازگار نیست."
            }
            connection.inputStream.use { source ->
                FileOutputStream(temporary).use { target ->
                    val buffer = ByteArray(128 * 1024)
                    var total = 0L
                    var lastReported = 0
                    while (true) {
                        val count = source.read(buffer)
                        if (count < 0) break
                        total += count
                        require(total <= MODEL_BYTES) {
                            "CACTUS-DOWNLOAD-003: فایل دریافتی بزرگ‌تر از نسخه رسمی است."
                        }
                        target.write(buffer, 0, count)
                        val percent = (total * 100 / MODEL_BYTES).toInt().coerceIn(0, 100)
                        if (percent >= lastReported + 2) {
                            lastReported = percent
                            onProgress("در حال دانلود Cactus Needle 3: $percent٪")
                        }
                    }
                    target.fd.sync()
                    require(total == MODEL_BYTES) {
                        "CACTUS-DOWNLOAD-004: دانلود ناقص است؛ $total از $MODEL_BYTES بایت دریافت شد."
                    }
                }
            }
            val digest = MessageDigest.getInstance("SHA-256")
                .digest(temporary.readBytes())
                .joinToString("") { "%02x".format(it) }
            require(digest.startsWith(MODEL_SHA256_PREFIX) && digest.endsWith(MODEL_SHA256_SUFFIX)) {
                "CACTUS-DOWNLOAD-005: بررسی SHA-256 فایل رسمی ناموفق بود."
            }
            replaceModel(temporary)
            publishToDownloads(modelFile)
            onProgress("دانلود و بررسی فایل کامل شد؛ در پوشه Downloads/AI Chat Test/Cactus ذخیره شد.")
            return "Cactus Needle 3 نصب شد و فایل آن در Download/AI Chat Test/Cactus/needle3.cact قابل مشاهده است."
        } catch (failure: Throwable) {
            temporary.delete()
            throw failure
        } finally {
            connection.disconnect()
        }
    }

    private fun replaceModel(temporary: File) {
        check(temporary.isFile && temporary.length() >= MIN_MODEL_BYTES) {
            "CACTUS-MODEL-001: فایل مدل معتبر نیست."
        }
        val backup = File(root, ".$MODEL_FILE.previous")
        backup.delete()
        if (modelFile.exists()) {
            check(modelFile.renameTo(backup)) {
                "CACTUS-MODEL-002: پشتیبان‌گیری از مدل قبلی ناموفق بود."
            }
        }
        if (!temporary.renameTo(modelFile)) {
            if (backup.exists()) backup.renameTo(modelFile)
            error("CACTUS-MODEL-003: انتقال فایل مدل به پوشه داخلی برنامه ناموفق بود.")
        }
        backup.delete()
    }

    private fun publishToDownloads(source: File) {
        val resolver = context.contentResolver
        val projection = arrayOf(MediaStore.MediaColumns._ID)
        resolver.query(
            MediaStore.Downloads.EXTERNAL_CONTENT_URI,
            projection,
            "${MediaStore.MediaColumns.DISPLAY_NAME}=? AND ${MediaStore.MediaColumns.RELATIVE_PATH}=?",
            arrayOf(MODEL_FILE, "$DOWNLOAD_RELATIVE_PATH/"),
            null,
        )?.use { cursor ->
            val idColumn = cursor.getColumnIndexOrThrow(MediaStore.MediaColumns._ID)
            while (cursor.moveToNext()) {
                val existing = Uri.withAppendedPath(
                    MediaStore.Downloads.EXTERNAL_CONTENT_URI,
                    cursor.getLong(idColumn).toString(),
                )
                resolver.delete(existing, null, null)
            }
        }
        val values = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, MODEL_FILE)
            put(MediaStore.MediaColumns.MIME_TYPE, "application/octet-stream")
            put(MediaStore.MediaColumns.RELATIVE_PATH, "$DOWNLOAD_RELATIVE_PATH/")
            put(MediaStore.MediaColumns.IS_PENDING, 1)
        }
        val destination = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
            ?: error("CACTUS-DOWNLOAD-006: ساخت فایل در پوشه عمومی Downloads ناموفق بود.")
        try {
            resolver.openOutputStream(destination, "w")?.use { output ->
                source.inputStream().use { input -> input.copyTo(output) }
            } ?: error("CACTUS-DOWNLOAD-007: نوشتن فایل در Downloads ناموفق بود.")
            val published = ContentValues().apply {
                put(MediaStore.MediaColumns.IS_PENDING, 0)
            }
            resolver.update(destination, published, null, null)
        } catch (failure: Throwable) {
            resolver.delete(destination, null, null)
            throw failure
        }
    }
}
