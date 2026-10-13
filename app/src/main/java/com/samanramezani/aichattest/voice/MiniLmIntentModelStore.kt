package com.samanramezani.aichattest.voice

import android.content.ContentResolver
import android.net.Uri
import android.provider.OpenableColumns
import java.io.File
import java.io.FileOutputStream
import java.util.zip.ZipInputStream
import org.json.JSONArray
import org.json.JSONObject

/**
 * Stores a user-supplied, fine-tuned MiniLM intent-classifier bundle locally.
 * Expected ZIP entries: model.onnx, tokenizer.json, labels.json.
 * A generic sentence-embedding MiniLM is not an intent classifier by itself.
 */
internal class MiniLmIntentModelStore(
    private val resolver: ContentResolver,
    private val root: File,
) {
    fun isReady(): Boolean = File(root, "model.onnx").let { it.isFile && it.length() > 1024 } &&
        File(root, "tokenizer.json").let { it.isFile && it.length() > 0 } &&
        File(root, "labels.json").let { it.isFile && it.length() > 0 }

    private fun parseLabels(raw: String): List<String> {
        val text = raw.trim()
        val labels = when {
            text.startsWith("[") -> {
                val array = JSONArray(text)
                (0 until array.length()).map { array.getString(it).uppercase(java.util.Locale.ROOT) }
            }
            else -> {
                val obj = JSONObject(text)
                val array = obj.optJSONArray("labels")
                if (array != null) {
                    (0 until array.length()).map { array.getString(it).uppercase(java.util.Locale.ROOT) }
                } else {
                    (0 until obj.length()).map { obj.optString(it.toString()).uppercase(java.util.Locale.ROOT) }
                }
            }
        }
        require(labels.distinct().size == labels.size) { "MINILM-IMPORT-011: labels.json برچسب تکراری دارد." }
        return labels
    }

    fun importBundle(uri: Uri): String {
        val name = resolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
            ?.use { if (it.moveToFirst()) it.getString(0).orEmpty() else "" }.orEmpty()
        require(name.endsWith(".zip", true)) {
            "MINILM-IMPORT-001: یک بسته ZIP شامل model.onnx، tokenizer.json و labels.json انتخاب کن."
        }
        val staging = File(root.parentFile, ".minilm-import-${System.currentTimeMillis()}")
        staging.deleteRecursively()
        check(staging.mkdirs()) { "MINILM-IMPORT-002: پوشه موقت مدل ساخته نشد." }
        try {
            val input = resolver.openInputStream(uri) ?: error("MINILM-IMPORT-003: فایل قابل خواندن نیست.")
            input.use { source ->
                ZipInputStream(source).use { zip ->
                    var entries = 0
                    var totalBytes = 0L
                    while (true) {
                        val entry = zip.nextEntry ?: break
                        entries++
                        require(entries <= 100) { "MINILM-IMPORT-004: بسته بیش از ۱۰۰ ورودی دارد." }
                        val path = entry.name.replace('\\', '/')
                        require(!path.startsWith("/") && path.split('/').none { it == ".." }) {
                            "MINILM-IMPORT-005: مسیر ناامن داخل بسته مدل شناسایی شد."
                        }
                        if (!entry.isDirectory) {
                            val basename = path.substringAfterLast('/')
                            if (basename in setOf("model.onnx", "tokenizer.json", "labels.json")) {
                                val target = File(staging, basename)
                                FileOutputStream(target).use { out ->
                                    val buffer = ByteArray(32 * 1024)
                                    while (true) {
                                        val n = zip.read(buffer)
                                        if (n < 0) break
                                        totalBytes += n
                                        require(totalBytes <= 600L * 1024 * 1024) {
                                            "MINILM-IMPORT-006: حجم بسته از ۶۰۰ مگابایت بیشتر است."
                                        }
                                        out.write(buffer, 0, n)
                                    }
                                }
                            }
                        }
                        zip.closeEntry()
                    }
                }
            }
            val model = File(staging, "model.onnx")
            val tokenizer = File(staging, "tokenizer.json")
            val labels = File(staging, "labels.json")
            require(model.isFile && model.length() > 1024 && tokenizer.isFile && tokenizer.length() > 0 && labels.isFile && labels.length() > 0) {
                "MINILM-IMPORT-007: بسته باید هر سه فایل model.onnx، tokenizer.json و labels.json را داشته باشد."
            }
            val labelText = labels.readText(Charsets.UTF_8)
            val parsedLabels = parseLabels(labelText)
            val allowedLabels = setOf("CHAT", "GET_TIME", "GET_DATE", "INCREASE_VOLUME", "DECREASE_VOLUME",
                "SET_VOLUME", "INCREASE_BRIGHTNESS", "DECREASE_BRIGHTNESS", "SET_BRIGHTNESS")
            require(parsedLabels.isNotEmpty() && parsedLabels.contains("CHAT") && parsedLabels.all { it in allowedLabels }) {
                "MINILM-IMPORT-008: labels.json باید CHAT و برچسب‌های معتبر فرمان‌های پشتیبانی‌شده را تعریف کند."
            }
            // Replace atomically where possible; retain the previous working bundle if promotion fails.
            val backup = File(root.parentFile, ".minilm-backup-${System.currentTimeMillis()}")
            val hadPrevious = root.exists()
            if (hadPrevious) check(root.renameTo(backup)) {
                "MINILM-IMPORT-009: مدل قبلی برای جایگزینی امن کنار گذاشته نشد."
            }
            if (!staging.renameTo(root)) {
                if (hadPrevious) backup.renameTo(root)
                error("MINILM-IMPORT-010: انتقال بسته مدل به پوشه نهایی ناموفق بود.")
            }
            if (hadPrevious) backup.deleteRecursively()
            return "بسته وارد شد. استنتاج MiniLM فقط وقتی فعال می‌شود که model.onnx یک طبقه‌بند سازگار با WordPiece و برچسب‌های labels.json باشد؛ در غیر این صورت فرمان‌های امن قاعده‌محور ادامه می‌یابند."
        } catch (t: Throwable) {
            staging.deleteRecursively()
            throw t
        }
    }
}
