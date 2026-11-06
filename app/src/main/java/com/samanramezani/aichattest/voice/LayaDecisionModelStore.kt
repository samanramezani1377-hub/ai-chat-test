package com.samanramezani.aichattest.voice

import android.content.ContentResolver
import android.content.ContentValues
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.provider.OpenableColumns
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.util.zip.ZipEntry
import java.util.zip.ZipInputStream
import java.util.zip.ZipOutputStream
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
        private const val CACHE_FILE = "AI-Chat-Test-Laya-INT4-Bundle.zip"
        private const val DOWNLOAD_RELATIVE_PATH = "Download/AI Chat Test/Laya/"
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

        // If an older app version installed Laya without retaining the original bundle,
        // create the Downloads backup from the already-installed files instead of re-fetching.
        val existingCache = findCachedBundleUri()
        if (isReady()) {
            val retained = existingCache != null || runCatching { saveCachedBundle(root) }.isSuccess
            onProgress(if (retained) "مدل Laya از قبل نصب است؛ نسخهٔ دانلودی نگه‌داری شد و دانلود مجدد انجام نشد."
                else "مدل Laya از قبل نصب است؛ دانلود مجدد انجام نشد، اما ذخیره در Downloads ناموفق بود.")
            return if (retained) {
                "مدل Laya Multilingual INT4 از قبل نصب است و دانلود مجدد انجام نشد. نسخهٔ اصلی در Downloads/AI Chat Test/Laya/$CACHE_FILE نگه‌داری می‌شود."
            } else {
                "مدل Laya Multilingual INT4 از قبل نصب و آماده است؛ دانلود مجدد لازم نیست، اما ساخت نسخهٔ پشتیبان در Downloads ناموفق بود."
            }
        }

        // The bundle ZIP is kept in the public Downloads folder (Android 10+) so it
        // survives app updates and reinstalls. Restore it before contacting Hugging Face.
        val cachedUri = existingCache
        if (cachedUri != null) {
            val staging = newStagingDirectory()
            try {
                extractCachedBundle(cachedUri, staging)
                val result = validateAndInstall(staging, staging)
                onProgress("بستهٔ ذخیره‌شدهٔ Laya از Downloads بازیابی شد؛ دانلود اینترنتی لازم نشد.")
                return "$result\nبستهٔ اصلی در Downloads/AI Chat Test/Laya/$CACHE_FILE نگه داشته شد."
            } catch (t: Throwable) {
                staging.deleteRecursively()
                deleteCachedBundle(cachedUri)
                onProgress("نسخهٔ ذخیره‌شدهٔ Laya قابل استفاده نبود؛ دریافت دوباره آغاز می‌شود.")
            }
        }

        val staging = newStagingDirectory()
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

            // Cache a single portable bundle before moving the extracted files into the
            // app-private runtime directory. A cache-write failure must not block install.
            val retained = runCatching { saveCachedBundle(staging) }.fold(
                onSuccess = { true },
                onFailure = {
                    onProgress("هشدار: ذخیرهٔ نسخهٔ پشتیبان در Downloads ناموفق بود: ${it.message ?: "خطای ناشناخته"}")
                    false
                }
            )

            return validateAndInstall(staging, staging) + if (retained) {
                "\nنسخهٔ قابل‌بازیابی بسته در Downloads/AI Chat Test/Laya/$CACHE_FILE ذخیره شد."
            } else {
                "\nهشدار: نصب انجام شد، اما نگه‌داری بسته در Downloads موفق نبود."
            }
        } catch (t: Throwable) {
            staging.deleteRecursively()
            throw t
        }
    }

    private fun newStagingDirectory(): File {
        val staging = File(root.parentFile, ".laya-stage-${System.currentTimeMillis()}")
        staging.deleteRecursively()
        check(staging.mkdirs()) { "LAYA-DOWNLOAD-001: پوشهٔ موقت دانلود ساخته نشد." }
        return staging
    }

    private fun findCachedBundleUri(): Uri? {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val collection = MediaStore.Downloads.EXTERNAL_CONTENT_URI
            val projection = arrayOf(MediaStore.Downloads._ID)
            val selection = "${MediaStore.MediaColumns.DISPLAY_NAME}=? AND ${MediaStore.MediaColumns.RELATIVE_PATH}=?"
            resolver.query(
                collection, projection, selection,
                arrayOf(CACHE_FILE, DOWNLOAD_RELATIVE_PATH), null
            )?.use { cursor ->
                if (cursor.moveToFirst()) {
                    return Uri.withAppendedPath(collection, cursor.getLong(0).toString())
                }
            }
        }
        val fallback = File(root.parentFile, CACHE_FILE)
        return if (fallback.isFile && fallback.length() > 0L) Uri.fromFile(fallback) else null
    }

    private fun deleteCachedBundle(uri: Uri) {
        runCatching {
            if (uri.scheme == "file") {
                File(uri.path.orEmpty()).delete()
            } else {
                resolver.delete(uri, null, null)
            }
        }
    }

    private fun extractCachedBundle(uri: Uri, staging: File) {
        val input = if (uri.scheme == "file") FileInputStream(File(uri.path.orEmpty()))
            else resolver.openInputStream(uri)
                ?: error("LAYA-CACHE-001: بستهٔ ذخیره‌شده قابل خواندن نیست.")
        input.use { raw ->
            ZipInputStream(java.io.BufferedInputStream(raw)).use { zip ->
                var entries = 0
                var expanded = 0L
                while (true) {
                    val entry = zip.nextEntry ?: break
                    entries++
                    require(entries <= MAX_ENTRIES) {
                        "LAYA-CACHE-002: تعداد فایل‌های بستهٔ ذخیره‌شده بیش از حد مجاز است."
                    }
                    val relative = entry.name.replace('\\', '/').removePrefix("./").trimEnd('/')
                    require(relative.isNotBlank() && !relative.startsWith("/") &&
                        relative.split('/').none { it == ".." || it == "." || it.isBlank() }
                    ) { "LAYA-CACHE-003: مسیر ناامن در بستهٔ ذخیره‌شده وجود دارد." }
                    if (!entry.isDirectory) {
                        val target = File(staging, relative).canonicalFile
                        require(target.toPath().startsWith(staging.canonicalFile.toPath())) {
                            "LAYA-CACHE-003: مسیر ناامن در بستهٔ ذخیره‌شده وجود دارد."
                        }
                        target.parentFile?.mkdirs()
                        FileOutputStream(target).use { output ->
                            val buffer = ByteArray(64 * 1024)
                            while (true) {
                                val count = zip.read(buffer)
                                if (count < 0) break
                                expanded += count
                                require(expanded <= MAX_EXPANDED_BYTES) {
                                    "LAYA-CACHE-004: حجم بستهٔ ذخیره‌شده از ۶۵۰ مگابایت بیشتر است."
                                }
                                output.write(buffer, 0, count)
                            }
                        }
                    }
                    zip.closeEntry()
                }
            }
        }
    }

    private fun saveCachedBundle(sourceRoot: File) {
        val temporaryZip = File(root.parentFile, ".laya-cache-build.zip")
        temporaryZip.delete()
        try {
            ZipOutputStream(java.io.BufferedOutputStream(FileOutputStream(temporaryZip))).use { zip ->
                zip.setLevel(java.util.zip.Deflater.BEST_SPEED)
                sourceRoot.walkTopDown().filter { it.isFile }.forEach { file ->
                    val relative = file.relativeTo(sourceRoot).invariantSeparatorsPath
                    zip.putNextEntry(ZipEntry(relative))
                    FileInputStream(file).use { input ->
                        val buffer = ByteArray(128 * 1024)
                        while (true) {
                            val count = input.read(buffer)
                            if (count < 0) break
                            zip.write(buffer, 0, count)
                        }
                    }
                    zip.closeEntry()
                }
            }

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                val collection = MediaStore.Downloads.EXTERNAL_CONTENT_URI
                val selection = "${MediaStore.MediaColumns.DISPLAY_NAME}=? AND ${MediaStore.MediaColumns.RELATIVE_PATH}=?"
                resolver.query(
                    collection, arrayOf(MediaStore.Downloads._ID), selection,
                    arrayOf(CACHE_FILE, DOWNLOAD_RELATIVE_PATH), null
                )?.use { cursor ->
                    while (cursor.moveToNext()) {
                        resolver.delete(Uri.withAppendedPath(collection, cursor.getLong(0).toString()), null, null)
                    }
                }
                val values = ContentValues().apply {
                    put(MediaStore.MediaColumns.DISPLAY_NAME, CACHE_FILE)
                    put(MediaStore.MediaColumns.MIME_TYPE, "application/zip")
                    put(MediaStore.MediaColumns.RELATIVE_PATH, DOWNLOAD_RELATIVE_PATH)
                    put(MediaStore.MediaColumns.IS_PENDING, 1)
                }
                val uri = resolver.insert(collection, values)
                    ?: error("ساخت فایل در Downloads ناموفق بود.")
                try {
                    val output = resolver.openOutputStream(uri, "w")
                        ?: error("نوشتن فایل در Downloads ناموفق بود.")
                    output.use { stream ->
                        FileInputStream(temporaryZip).use { input -> input.copyTo(stream, 128 * 1024) }
                    }
                    val ready = ContentValues().apply { put(MediaStore.MediaColumns.IS_PENDING, 0) }
                    resolver.update(uri, ready, null, null)
                    return
                } catch (t: Throwable) {
                    resolver.delete(uri, null, null)
                    throw t
                }
            }

            // Pre-Android 10 fallback: keep a reusable app-private archive where scoped
            // Downloads writes are unavailable without extra permissions.
            val fallback = File(root.parentFile, CACHE_FILE)
            if (fallback.exists()) check(fallback.delete()) { "حذف بستهٔ ذخیره‌شدهٔ قبلی ناموفق بود." }
            check(temporaryZip.renameTo(fallback)) { "ذخیرهٔ بستهٔ Laya ناموفق بود." }
        } finally {
            temporaryZip.delete()
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
