package com.samanramezani.aichattest.screen

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

object ScreenVisionModelStore {
    data class Paths(val model: File, val projector: File)

    private const val PREFS = "screen_assistant"
    private const val MODEL_KEY = "vision_model_path"
    private const val PROJECTOR_KEY = "vision_projector_path"
    private const val MODEL_URL = "https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/SmolVLM-256M-Instruct-Q8_0.gguf"
    private const val PROJECTOR_URL = "https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf"
    private const val MODEL_SHA256 = "2a31195d3769c0b0fd0a4906201666108834848db768af11de1d2cef7cd35e65"
    private const val PROJECTOR_SHA256 = "7e943f7c53f0382a6fc41b6ee0c2def63ba4fded9ab8ed039cc9e2ab905e0edd"

    fun paths(context: Context): Paths? {
        val p = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        val model = p.getString(MODEL_KEY, null)?.let(::File)
        val projector = p.getString(PROJECTOR_KEY, null)?.let(::File)
        return if (model?.isFile == true && projector?.isFile == true) Paths(model, projector) else null
    }

    fun fileNames(context: Context): Pair<String?, String?> {
        val p = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        return p.getString(MODEL_KEY, null)?.let(::File)?.name to
            p.getString(PROJECTOR_KEY, null)?.let(::File)?.name
    }

    suspend fun import(context: Context, uri: Uri, projector: Boolean): File = withContext(Dispatchers.IO) {
        val directory = File(context.filesDir, "screen-vision").apply { mkdirs() }
        val displayName = context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
            ?.use { cursor -> if (cursor.moveToFirst()) cursor.getString(0) else null }
            ?.takeIf { it.endsWith(".gguf", true) } ?: if (projector) "mmproj.gguf" else "vision-model.gguf"
        val destination = File(directory, if (projector) "projector-" + displayName else "model-" + displayName)
        val temp = File(directory, destination.name + ".part")
        context.contentResolver.openInputStream(uri)?.use { input ->
            temp.outputStream().use { output -> input.copyTo(output) }
        } ?: error("فایل انتخاب‌شده قابل خواندن نیست.")
        require(temp.length() > 1024 * 1024L) { "فایل GGUF ناقص یا بیش از حد کوچک است." }
        check(temp.renameTo(destination)) { "ذخیره فایل مدل ناموفق بود." }
        val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        prefs.edit().putString(if (projector) PROJECTOR_KEY else MODEL_KEY, destination.absolutePath).apply()
        destination
    }

    suspend fun downloadRecommended(context: Context, onProgress: (String) -> Unit = {}) = withContext(Dispatchers.IO) {
        val directory = File(context.filesDir, "screen-vision").apply { mkdirs() }
        val model = File(directory, "SmolVLM-256M-Instruct-Q8_0.gguf")
        val projector = File(directory, "mmproj-SmolVLM-256M-Instruct-Q8_0.gguf")
        downloadVerified(MODEL_URL, model, MODEL_SHA256) { onProgress("مدل بینایی: " + it) }
        downloadVerified(PROJECTOR_URL, projector, PROJECTOR_SHA256) { onProgress("فایل بینایی‌ساز: " + it) }
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
            .putString(MODEL_KEY, model.absolutePath)
            .putString(PROJECTOR_KEY, projector.absolutePath)
            .apply()
    }

    private fun downloadVerified(url: String, destination: File, expectedSha: String, progress: (String) -> Unit) {
        if (destination.isFile && sha256(destination) == expectedSha) {
            progress("از قبل آماده است")
            return
        }
        val temp = File(destination.parentFile, destination.name + ".part")
        temp.delete()
        val connection = (URL(url).openConnection() as HttpURLConnection).apply {
            connectTimeout = 20_000
            readTimeout = 45_000
            instanceFollowRedirects = true
            setRequestProperty("User-Agent", "Dina-Android/1.0")
        }
        try {
            connection.connect()
            check(connection.responseCode in 200..299) { "دانلود مدل ناموفق بود: HTTP " + connection.responseCode }
            val total = connection.contentLengthLong
            var copied = 0L
            connection.inputStream.use { input ->
                temp.outputStream().use { output ->
                    val buffer = ByteArray(1024 * 1024)
                    while (true) {
                        val count = input.read(buffer)
                        if (count < 0) break
                        output.write(buffer, 0, count)
                        copied += count
                        if (total > 0) progress((copied * 100 / total).toInt().toString() + "٪")
                    }
                }
            }
            check(temp.length() > 1_000_000L) { "فایل دانلودشده ناقص است." }
            check(sha256(temp) == expectedSha) { "SHA-256 فایل دانلودشده معتبر نیست؛ فایل موقت پاک شد." }
            if (destination.exists()) destination.delete()
            check(temp.renameTo(destination)) { "انتقال فایل دانلودشده ناموفق بود." }
        } catch (t: Throwable) {
            temp.delete()
            throw t
        } finally {
            connection.disconnect()
        }
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buffer = ByteArray(1024 * 1024)
            while (true) {
                val n = input.read(buffer)
                if (n < 0) break
                digest.update(buffer, 0, n)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
}
