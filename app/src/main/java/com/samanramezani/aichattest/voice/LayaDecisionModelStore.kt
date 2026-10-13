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
            "tokenizer/tokenizer_config.json", "encoder/config.json", "rl_agent_config.json",
        )
    }

    private fun missingRequirements(): List<String> {
        val missing = REQUIRED.filter { relative ->
            val file = File(root, relative)
            !file.isFile || file.length() <= 0L
        }.toMutableList()
        val graph = File(root, MODEL_FILE)
        if (graph.isFile && graph.length() <= 16_000L) missing += "$MODEL_FILE (گراف ناقص)"
        val weights = File(root, "$MODEL_FILE.data")
        if (weights.isFile && weights.length() <= 100_000_000L) missing += "$MODEL_FILE.data (وزن‌ها ناقص)"
        val encoderConfig = File(root, "encoder/config.json")
        if (encoderConfig.isFile &&
            runCatching { JSONObject(encoderConfig.readText(Charsets.UTF_8)) }.isFailure
        ) missing += "encoder/config.json (JSON نامعتبر)"
        return missing.distinct()
    }

    fun isReady(): Boolean = missingRequirements().isEmpty()

    fun status(): String = when {
        isReady() -> "Laya Multilingual INT4 آماده است؛ تصمیم‌گیری محلی فعال است."
        !root.exists() -> "مدل Laya هنوز نصب نشده است؛ از گزینهٔ دانلود و نصب خودکار استفاده کن."
        else -> "بستهٔ Laya ناقص است؛ موارد مفقود/ناقص: ${missingRequirements().joinToString(", ")}"
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
                        val rawPath = entry.name.replace('\\', '/')
                        require(!rawPath.startsWith("/") && rawPath.split('/').none { it == ".." }) {
                            "LAYA-IMPORT-005: مسیر ناامن در ZIP مدل وجود دارد."
                        }
                        val relative = rawPath.removePrefix("./").trimEnd('/')
                        require(relative.isNotBlank() && relative.split('/').none { it == ".." || it == "." || it.isBlank() }) {
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
            } ?: error("LAYA-IMPORT-007: ZIP باید شامل مدل و فایل data، tokenizer/tokenizer_config، encoder/config.json و rl_agent_config.json باشد.")
            return validateAndInstall(sourceRoot, staging)

        } catch (t: Throwable) {
            staging.deleteRecursively()
            throw t
        }
    }

    /**
     * Downloads the exact public multilingual INT4 bundle directly from its Hugging Face repo.
     * Large weights are streamed to private storage; they are never buffered in memory or APK assets.
     */
    fun downloadOfficialBundle(onProgress: (String) -> Unit = {}): String {
        root.parentFile?.mkdirs()
        val staging = File(root.parentFile, ".laya-stage-${System.currentTimeMillis()}")
        staging.deleteRecursively()
        check(staging.mkdirs()) { "LAYA-DOWNLOAD-001: پوشهٔ موقت دانلود ساخته نشد." }
        val files = listOf(
            "laya-multilingual-int4-layer32-vocab128.onnx" to "laya-multilingual-int4-layer32-vocab128.onnx",
            "laya-multilingual-int4-layer32-vocab128.onnx.data" to "laya-multilingual-int4-layer32-vocab128.onnx.data",
            "tokenizer/tokenizer.json" to "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json" to "tokenizer/tokenizer_config.json",
            "encoder/config.json" to "encoder/config.json",
            "rl_agent_config.json" to "rl_agent_config.json",
        )
        try {
            files.forEachIndexed { index, (remotePath, relativePath) ->
                val target = File(staging, relativePath)
                target.parentFile?.mkdirs()
                val label = "فایل ${index + 1} از ${files.size}: ${relativePath.substringAfterLast('/')}"
                downloadFile(remotePath, target) { percent ->
                    onProgress("دریافت بسته Laya — $label${percent?.let { " ($it٪)" } ?: ""}")
                }
                onProgress("دریافت بسته Laya — $label کامل شد.")
            }
            return validateAndInstall(staging, staging)
        } catch (t: Throwable) {
            staging.deleteRecursively()
            throw t
        }
    }

    private fun downloadFile(
        relativePath: String,
        target: File,
        onProgress: (Int?) -> Unit,
    ) {
        val url = java.net.URL(
            "https://huggingface.co/androidli/laya-multilingual-onnx-int4/resolve/main/$relativePath?download=true"
        )
        val connection = (url.openConnection() as java.net.HttpURLConnection).apply {
            connectTimeout = 30_000
            readTimeout = 120_000
            instanceFollowRedirects = true
            useCaches = false
            setRequestProperty("User-Agent", "AI-Chat-Test-Android/1.0")
            setRequestProperty("Accept", "application/octet-stream")
        }
        try {
            val code = connection.responseCode
            require(code == java.net.HttpURLConnection.HTTP_OK) {
                "LAYA-DOWNLOAD-002: دریافت $relativePath ناموفق بود؛ HTTP $code."
            }
            val expected = connection.contentLengthLong
            var received = 0L
            var lastReported = 0L
            connection.inputStream.use { input ->
                java.io.BufferedInputStream(input).use { buffered ->
                    java.io.BufferedOutputStream(FileOutputStream(target)).use { output ->
                        val buffer = ByteArray(128 * 1024)
                        while (true) {
                            val count = buffered.read(buffer)
                            if (count < 0) break
                            output.write(buffer, 0, count)
                            received += count
                            if (received - lastReported >= 2L * 1024L * 1024L) {
                                lastReported = received
                                onProgress(if (expected > 0L) ((received * 100L / expected).toInt().coerceIn(0, 100)) else null)
                            }
                        }
                    }
                }
            }
            require(received > 0L && (expected <= 0L || received == expected)) {
                "LAYA-DOWNLOAD-003: فایل $relativePath ناقص دریافت شد؛ bytes=$received expected=$expected."
            }
            onProgress(100)
        } catch (t: Throwable) {
            target.delete()
            throw t
        } finally {
            connection.disconnect()
        }
    }

    private fun validateAndInstall(sourceRoot: File, staging: File): String {
        val tokenizerJsonFile = File(sourceRoot, "tokenizer/tokenizer.json")
        val tokenizerConfigFile = File(sourceRoot, "tokenizer/tokenizer_config.json")
        val tokenizer = JSONObject(tokenizerJsonFile.readText(Charsets.UTF_8))
        val tokenizerConfig = JSONObject(tokenizerConfigFile.readText(Charsets.UTF_8))
        JSONObject(File(sourceRoot, "encoder/config.json").readText(Charsets.UTF_8))
        JSONObject(File(sourceRoot, "rl_agent_config.json").readText(Charsets.UTF_8))
        require(tokenizer.optJSONObject("model") != null) {
            "LAYA-IMPORT-008: tokenizer/tokenizer.json ساختار نامعتبر دارد."
        }
        // Validate with the same official native tokenizer used at inference before replacing
        // the previously working model bundle. JSON parsing alone does not prove parity.
        val nativeTokenizer = LayaNativeTokenizer(tokenizerJsonFile.readBytes())
        try {
            for (key in listOf("cls_token", "sep_token", "mask_token")) {
                val raw = tokenizerConfig.opt(key)
                val token = when (raw) {
                    is String -> raw
                    is JSONObject -> raw.optString("content").ifBlank { raw.optString("token") }
                    else -> ""
                }
                require(token.isNotBlank() && nativeTokenizer.tokenId(token) != null) {
                    "LAYA-IMPORT-008: توکن ویژه '$key' در tokenizer_config.json یا tokenizer.json معتبر نیست."
                }
            }
        } finally {
            nativeTokenizer.close()
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
        staging.deleteRecursively()
        if (!isReady()) {
            root.deleteRecursively()
            if (hadPrevious) backup.renameTo(root)
            error("LAYA-IMPORT-012: اعتبارسنجی نهایی مدل ناموفق بود؛ نسخه قبلی بازیابی شد.")
        }
        if (hadPrevious) backup.deleteRecursively()
        return "مدل Laya Multilingual INT4 همراه با tokenizer، encoder و وزن‌های خارجی نصب و اعتبارسنجی شد."
    }
}
