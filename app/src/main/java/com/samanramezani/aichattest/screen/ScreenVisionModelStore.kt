package com.samanramezani.aichattest.screen

import android.content.ContentValues
import android.content.Context
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.provider.OpenableColumns
import com.samanramezani.aichattest.storage.AtomicFileReplacement
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL

/**
 * Imported GGUF files are copied once into private runtime storage. Recommended downloads
 * also retain a user-visible copy in Downloads; manual imports never create a second public copy.
 */
object ScreenVisionModelStore {
    data class Paths(val model: File, val projector: File)

    private const val PREFS = "screen_assistant"
    private const val MODEL_KEY = "vision_model_path"
    private const val PROJECTOR_KEY = "vision_projector_path"
    private const val DOWNLOAD_RELATIVE_PATH = "Download/AI Chat Test/Screen Vision/"
    private const val MODEL_NAME = "SmolVLM-256M-Instruct-Q8_0.gguf"
    private const val PROJECTOR_NAME = "mmproj-SmolVLM-256M-Instruct-Q8_0.gguf"
    private const val MODEL_URL = "https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/$MODEL_NAME?download=true"
    private const val PROJECTOR_URL = "https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/$PROJECTOR_NAME?download=true"

    fun paths(context: Context): Paths? {
        val p = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        val model = p.getString(MODEL_KEY, null)?.let(::File)
        val projector = p.getString(PROJECTOR_KEY, null)?.let(::File)
        return if (model?.isFile == true && projector?.isFile == true &&
            isGguf(model) && isGguf(projector)
        ) Paths(model, projector) else null
    }

    fun fileNames(context: Context): Pair<String?, String?> {
        val p = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        return p.getString(MODEL_KEY, null)?.let(::File)?.name to
            p.getString(PROJECTOR_KEY, null)?.let(::File)?.name
    }

    suspend fun import(context: Context, uri: Uri, projector: Boolean): File = withContext(Dispatchers.IO) {
        val name = displayName(context, uri)
            ?.substringAfterLast('/')
            ?.takeIf { it.endsWith(".gguf", true) }
            ?: if (projector) PROJECTOR_NAME else MODEL_NAME
        val directory = runtimeDirectory(context)
        val destination = File(directory, (if (projector) "projector-" else "model-") + safeFileName(name))
        val temp = File(directory, destination.name + ".part")
        try {
            context.contentResolver.openInputStream(uri)?.use { input ->
                FileOutputStream(temp).use { output -> input.copyTo(output, 128 * 1024) }
            } ?: error("فایل انتخاب‌شده قابل خواندن نیست.")
            require(isGguf(temp)) { "فایل انتخاب‌شده GGUF معتبر یا کامل نیست." }

            // The app needs a private filesystem path for native inference. Do not
            // also copy a manually imported source into Downloads: that creates a visible
            // duplicate beside the user's original file.
            replaceAtomically(temp, destination)

            val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            prefs.edit().putString(if (projector) PROJECTOR_KEY else MODEL_KEY, destination.absolutePath).apply()
            destination
        } catch (t: Throwable) {
            temp.delete()
            throw t
        }
    }

    suspend fun downloadRecommended(context: Context, onProgress: (String) -> Unit = {}) =
        withContext(Dispatchers.IO) {
            val directory = runtimeDirectory(context)
            val model = File(directory, MODEL_NAME)
            val projector = File(directory, PROJECTOR_NAME)

            prepareOne(context, MODEL_NAME, MODEL_URL, model) {
                onProgress("مدل بینایی: $it")
            }
            prepareOne(context, PROJECTOR_NAME, PROJECTOR_URL, projector) {
                onProgress("فایل بینایی‌ساز: $it")
            }

            context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
                .putString(MODEL_KEY, model.absolutePath)
                .putString(PROJECTOR_KEY, projector.absolutePath)
                .apply()
        }

    private fun runtimeDirectory(context: Context): File =
        File(context.filesDir, "screen-vision").apply { check(isDirectory || mkdirs()) }

    private fun prepareOne(
        context: Context,
        fileName: String,
        url: String,
        runtimeFile: File,
        onProgress: (String) -> Unit,
    ) {
        val cached = findDownload(context, fileName)
        if (cached != null) {
            if (copyUriToFile(context, cached, runtimeFile) && isGguf(runtimeFile)) {
                onProgress("از Downloads موجود بازیابی شد؛ دانلود مجدد انجام نشد.")
                return
            }
            // Do not let a corrupt cached entry shadow a valid replacement with the same name.
            deleteDownload(context, cached)
        }

        // Older app versions kept the runtime copy privately. Promote it to Downloads
        // rather than downloading it again.
        if (isGguf(runtimeFile)) {
            runCatching { publishFileToDownloads(context, runtimeFile, fileName) }
                .onSuccess { onProgress("فایل نصب‌شده به Downloads منتقل شد؛ دانلود مجدد لازم نبود.") }
                .onFailure { onProgress("هشدار: ذخیره نسخه در Downloads انجام نشد: ${it.message}") }
            return
        }

        val temp = File(runtimeFile.parentFile, "$fileName.download")
        temp.delete()
        try {
            download(url, temp) { onProgress(it) }
            require(isGguf(temp)) { "فایل دریافتی $fileName یک GGUF معتبر نیست." }
            publishFileToDownloads(context, temp, fileName)
            replaceAtomically(temp, runtimeFile)
            onProgress("ذخیره شد: Downloads/AI Chat Test/Screen Vision/$fileName")
        } catch (t: Throwable) {
            temp.delete()
            throw t
        }
    }

    private fun findDownload(context: Context, fileName: String): Uri? {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val collection = MediaStore.Downloads.EXTERNAL_CONTENT_URI
            val selection = "${MediaStore.MediaColumns.DISPLAY_NAME}=? AND ${MediaStore.MediaColumns.RELATIVE_PATH}=?"
            context.contentResolver.query(
                collection,
                arrayOf(MediaStore.Downloads._ID),
                selection,
                arrayOf(fileName, DOWNLOAD_RELATIVE_PATH),
                null,
            )?.use { cursor ->
                if (cursor.moveToFirst()) {
                    return Uri.withAppendedPath(collection, cursor.getLong(0).toString())
                }
            }
        } else {
            val fallback = legacyDownloadsDirectory(context)
            val file = File(fallback, fileName)
            if (file.isFile && file.length() > 0L) return Uri.fromFile(file)
        }
        return null
    }

    private fun publishFileToDownloads(context: Context, source: File, fileName: String) {
        require(isGguf(source)) { "فایل GGUF برای ذخیره در Downloads معتبر نیست." }
        val existingUri = findDownload(context, fileName)
        if (existingUri != null) {
            val checkFile = File(source.parentFile, "$fileName.cached-check")
            val existingIsValid = copyUriToFile(context, existingUri, checkFile) && isGguf(checkFile)
            checkFile.delete()
            if (existingIsValid) return
            deleteDownload(context, existingUri)
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val values = ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, fileName)
                put(MediaStore.MediaColumns.MIME_TYPE, "application/octet-stream")
                put(MediaStore.MediaColumns.RELATIVE_PATH, DOWNLOAD_RELATIVE_PATH)
                put(MediaStore.MediaColumns.IS_PENDING, 1)
            }
            val target = context.contentResolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
                ?: error("ساخت فایل در Downloads ناموفق بود.")
            try {
                context.contentResolver.openOutputStream(target, "w")?.use { output ->
                    FileInputStream(source).use { input -> input.copyTo(output, 128 * 1024) }
                } ?: error("نوشتن فایل در Downloads ناموفق بود.")
                val ready = ContentValues().apply { put(MediaStore.MediaColumns.IS_PENDING, 0) }
                context.contentResolver.update(target, ready, null, null)
            } catch (t: Throwable) {
                context.contentResolver.delete(target, null, null)
                throw t
            }
        } else {
            val target = File(legacyDownloadsDirectory(context), fileName)
            FileInputStream(source).use { input ->
                FileOutputStream(target).use { output -> input.copyTo(output, 128 * 1024) }
            }
        }
    }

    private fun legacyDownloadsDirectory(context: Context): File =
        File(context.getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS) ?: context.filesDir,
            "AI Chat Test/Screen Vision").apply { check(isDirectory || mkdirs()) }

    private fun copyUriToFile(context: Context, uri: Uri, destination: File): Boolean {
        val temp = File(destination.parentFile, destination.name + ".copying")
        return try {
            val input = if (uri.scheme == "file") FileInputStream(File(uri.path.orEmpty()))
                else context.contentResolver.openInputStream(uri)
                    ?: error("Cannot read cached model")
            input.use { source ->
                FileOutputStream(temp).use { output -> source.copyTo(output, 128 * 1024) }
            }
            if (!isGguf(temp)) error("Cached GGUF is invalid")
            replaceAtomically(temp, destination)
            true
        } catch (_: Throwable) {
            temp.delete()
            false
        }
    }

    private fun deleteDownload(context: Context, uri: Uri) {
        runCatching {
            if (uri.scheme == "file") File(uri.path.orEmpty()).delete()
            else context.contentResolver.delete(uri, null, null)
        }
    }

    private fun displayName(context: Context, uri: Uri): String? =
        context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
            ?.use { cursor -> if (cursor.moveToFirst()) cursor.getString(0) else null }

    private fun safeFileName(value: String): String =
        value.substringAfterLast('/').substringAfterLast('\\').replace(Regex("[^A-Za-z0-9._-]"), "_")
            .take(180).ifBlank { "vision-model.gguf" }

    private fun isGguf(file: File): Boolean {
        if (!file.isFile || file.length() <= 1024L * 1024L) return false
        return runCatching {
            FileInputStream(file).use { input ->
                val magic = ByteArray(4)
                input.read(magic) == 4 && magic.contentEquals(byteArrayOf(0x47, 0x47, 0x55, 0x46))
            }
        }.getOrDefault(false)
    }

    private fun replaceAtomically(source: File, destination: File) {
        AtomicFileReplacement.replace(source, destination)
    }

    private fun download(url: String, destination: File, progress: (String) -> Unit) {
        val connection = (URL(url).openConnection() as HttpURLConnection).apply {
            connectTimeout = 30_000
            readTimeout = 120_000
            instanceFollowRedirects = true
            useCaches = false
            setRequestProperty("User-Agent", "Dina-Android/1.0")
            setRequestProperty("Accept", "application/octet-stream")
        }
        try {
            connection.connect()
            check(connection.responseCode in 200..299) {
                "دانلود فایل ناموفق بود: HTTP ${connection.responseCode}"
            }
            val expected = connection.contentLengthLong
            var copied = 0L
            var lastReported = 0L
            connection.inputStream.use { input ->
                FileOutputStream(destination).use { output ->
                    val buffer = ByteArray(128 * 1024)
                    while (true) {
                        val count = input.read(buffer)
                        if (count < 0) break
                        output.write(buffer, 0, count)
                        copied += count
                        if (copied - lastReported >= 2L * 1024L * 1024L) {
                            lastReported = copied
                            progress(if (expected > 0L) "${(copied * 100L / expected).toInt().coerceIn(0, 100)}٪" else "در حال دریافت")
                        }
                    }
                }
            }
            check(copied > 0L && (expected <= 0L || copied == expected)) {
                "دانلود ناقص است؛ فایل موقت قابل استفاده نیست."
            }
            progress("۱۰۰٪")
        } catch (t: Throwable) {
            destination.delete()
            throw t
        } finally {
            connection.disconnect()
        }
    }
}
