package com.samanramezani.aichattest.voice

import android.content.ContentResolver
import android.net.Uri
import android.provider.OpenableColumns
import java.io.File
import java.io.FileOutputStream
import java.util.zip.ZipInputStream
import org.json.JSONObject

/**
 * Private, transactional storage for the Laya multilingual INT4 decision model.
 * The ONNX graph and its external-data file must remain side-by-side under their original names.
 */
internal class LayaDecisionModelStore(
    private val resolver: ContentResolver,
    private val root: File,
) {
    companion object {
        const val MODEL_FILE = "laya-multilingual-int4-layer32-vocab128.onnx"
        private const val MAX_ENTRIES = 300
        private const val MAX_EXPANDED_BYTES = 650L * 1024L * 1024L
        private val REQUIRED = setOf(
            MODEL_FILE, "$MODEL_FILE.data", "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json", "rl_agent_config.json",
        )
    }

    fun isReady(): Boolean = REQUIRED.all { File(root, it).isFile && File(root, it).length() > 0 } &&
        File(root, MODEL_FILE).length() > 16_000 &&
        File(root, "$MODEL_FILE.data").length() > 100_000_000L

    fun status(): String = when {
        isReady() -> "Laya Multilingual INT4 آماده است؛ تصمیم‌گیری محلی فعال است."
        root.exists() -> "بسته Laya ناقص است؛ ZIP کامل مدل INT4 را دوباره وارد کن."
        else -> "مدل Laya هنوز وارد نشده است."
    }

    fun importBundle(uri: Uri): String {
        val name = resolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
            ?.use { if (it.moveToFirst()) it.getString(0).orEmpty() else "" }.orEmpty()
        require(name.endsWith(".zip", true)) {
            "LAYA-IMPORT-001: بسته ZIP مدل Laya را انتخاب کن؛ فایل ONNX به‌تنهایی کافی نیست."
        }
        val staging = File(root.parentFile, ".laya-stage-${System.currentTimeMillis()}")
        staging.deleteRecursively()
        check(staging.mkdirs()) { "LAYA-IMPORT-002: پوشه موقت ساخته نشد." }
        try {
            val input = resolver.openInputStream(uri) ?: error("LAYA-IMPORT-003: فایل ZIP قابل خواندن نیست.")
            input.use { source ->
                ZipInputStream(source).use { zip ->
                    var entries = 0
                    var expanded = 0L
                    while (true) {
                        val entry = zip.nextEntry ?: break
                        entries++
                        require(entries <= MAX_ENTRIES) { "LAYA-IMPORT-004: تعداد فایل‌های ZIP بیش از حد مجاز است." }
                        val relative = entry.name.replace('\\', '/').trimStart('.', '/').trimEnd('/')
                        require(relative.isNotBlank() && !relative.startsWith("/") &&
                            relative.split('/').none { it == ".." || it.isBlank() }) {
                            "LAYA-IMPORT-005: مسیر ناامن در ZIP مدل وجود دارد."
                        }
                        if (!entry.isDirectory) {
                            val target = File(staging, relative).canonicalFile
                            require(target.toPath().startsWith(staging.canonicalFile.toPath())) {
                                "LAYA-IMPORT-005: مسیر ناامن در ZIP مدل وجود دارد."
                            }
                            target.parentFile?.mkdirs()
                            FileOutputStream(target).use { out ->
                                val buffer = ByteArray(64 * 1024)
                                while (true) {
                                    val count = zip.read(buffer)
                                    if (count < 0) break
                                    expanded += count
                                    require(expanded <= MAX_EXPANDED_BYTES) {
                                        "LAYA-IMPORT-006: حجم استخراج‌شده از ۶۵۰ مگابایت بیشتر است."
                                    }
                                    out.write(buffer, 0, count)
                                }
                            }
                        }
                        zip.closeEntry()
                    }
                }
            }
            val sourceRoot = staging.walkTopDown().maxDepth(5).firstOrNull { candidate ->
                candidate.isDirectory && REQUIRED.all { File(candidate, it).isFile }
            } ?: error("LAYA-IMPORT-007: ZIP باید شامل مدل layer32-vocab128، فایل data، tokenizer و rl_agent_config.json باشد.")
            val tokenizer = JSONObject(File(sourceRoot, "tokenizer/tokenizer.json").readText(Charsets.UTF_8))
            require(tokenizer.optJSONObject("model") != null) {
                "LAYA-IMPORT-008: tokenizer/tokenizer.json ساختار نامعتبر دارد."
            }
            val graph = File(sourceRoot, MODEL_FILE)
            require(graph.length() > 16_000 && File(sourceRoot, "$MODEL_FILE.data").length() > 100_000_000L) {
                "LAYA-IMPORT-009: گراف یا وزن‌های INT4 ناقص‌اند."
            }
            val backup = File(root.parentFile, ".laya-backup-${System.currentTimeMillis()}")
            val hadPrevious = root.exists()
            if (hadPrevious) check(root.renameTo(backup)) {
                "LAYA-IMPORT-010: نسخه قبلی مدل برای جایگزینی امن کنار گذاشته نشد."
            }
            if (!sourceRoot.renameTo(root)) {
                if (hadPrevious) backup.renameTo(root)
                error("LAYA-IMPORT-011: انتقال مدل به پوشه نهایی ناموفق بود.")
            }
            if (hadPrevious) backup.deleteRecursively()
            staging.deleteRecursively()
            check(isReady()) { "LAYA-IMPORT-012: اعتبارسنجی نهایی مدل ناموفق بود." }
            return "مدل Laya Multilingual INT4 با tokenizer و وزن‌های خارجی وارد شد."
        } catch (t: Throwable) {
            staging.deleteRecursively()
            throw t
        }
    }
}
