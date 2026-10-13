package com.samanramezani.aichattest.voice

import android.content.ContentResolver
import android.net.Uri
import android.provider.OpenableColumns
import java.io.File
import java.io.FileOutputStream
import java.util.zip.ZipInputStream

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
            require(listOf("CHAT", "GET_TIME", "INCREASE_VOLUME", "DECREASE_VOLUME", "SET_VOLUME",
                "INCREASE_BRIGHTNESS", "DECREASE_BRIGHTNESS", "SET_BRIGHTNESS").all { labelText.contains(it) }) {
                "MINILM-IMPORT-008: labels.json باید برچسب‌های فرمان‌های پشتیبانی‌شده و CHAT را تعریف کند."
            }
            root.deleteRecursively()
            check(staging.renameTo(root)) { "MINILM-IMPORT-009: انتقال بسته مدل به پوشه نهایی ناموفق بود." }
            return "بسته MiniLM وارد شد. فرمان‌های فارسی محلی فعال‌اند؛ فایل‌های مدل فقط پس از تطبیق قرارداد ورودی/خروجی ONNX قابل استفاده برای طبقه‌بندی مدل‌محور خواهند بود."
        } catch (t: Throwable) {
            staging.deleteRecursively()
            throw t
        }
    }
}
