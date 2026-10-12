package com.samanramezani.aichattest.voice

import android.content.ContentResolver
import android.net.Uri
import android.provider.OpenableColumns
import java.io.File
import java.io.FileOutputStream
import java.io.BufferedInputStream
import java.io.InputStream
import java.util.zip.ZipInputStream
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream
import org.json.JSONObject

/**
 * Accept only the configured wake phrase label, not arbitrary non-empty KWS results.
 * Sherpa may return the original Persian label or its Latin alias.
 */
internal fun isDinaWakeKeywordLabel(keyword: String): Boolean {
    val normalized = keyword.trim().lowercase().filter { it.isLetterOrDigit() }
    return normalized == "دینا" || normalized == "dina"
}

internal fun shouldKeepWakeServiceForConversation(
    armed: Boolean,
    listeningEnabled: Boolean,
    phraseDetected: Boolean,
    controllerActive: Boolean,
): Boolean = armed && !listeningEnabled && phraseDetected && controllerActive

data class WakeWordModelEntry(
    val id: String,
    val title: String,
    val directory: File,
    val encoder: File,
    val decoder: File,
    val joiner: File,
    val tokens: File,
    val keywords: File,
    val modelType: String,
    val modelingUnit: String,
    val bpeVocab: File?,
)

/**
 * Imports a complete sherpa-onnx streaming KWS package configured for the Persian wake phrase
 * "دینا". The archive must contain encoder/decoder/joiner ONNX files, tokens.txt, and
 * keywords.txt with the original keyword label "دینا". Optional wakeword.json may set
 * modelType and modelingUnit; bpe.model is detected automatically. The package's tokenizer/model
 * must actually support Persian; filename validation alone cannot guarantee recognition quality.
 */
class WakeWordModelStore(
    private val resolver: ContentResolver,
    private val root: File,
    /** Bundled exact token table for the known upstream zh-en KWS archive. */
    private val officialTokensProvider: (() -> InputStream)? = null,
) {
    init { root.mkdirs() }

    fun list(): List<WakeWordModelEntry> =
        root.listFiles()?.filter { it.isDirectory && !it.name.startsWith(".") }
            ?.mapNotNull(::validateDirectory)?.sortedBy { it.title } ?: emptyList()

    fun importArchive(uri: Uri, onProgress: (String) -> Unit = {}): WakeWordModelEntry {
        root.mkdirs()
        val importStartedAtNanos = System.nanoTime()
        onProgress("در حال بررسی فایل انتخاب‌شده برای بیدارباش…")
        val displayName = resolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
            ?.use { cursor -> if (cursor.moveToFirst()) cursor.getString(0).orEmpty() else "" }
            .orEmpty()
            .ifBlank { uri.lastPathSegment.orEmpty().substringAfterLast('/') }
        val archive = displayName.lowercase()
        val isOfficialZhEnArchive = archive.contains("zh-en-3m")
        val mime = resolver.getType(uri).orEmpty().lowercase()
        val isZip = archive.endsWith(".zip") || mime.contains("zip")
        val isTarBz2 = archive.endsWith(".tar.bz2") || archive.endsWith(".tbz2") ||
            mime.contains("bzip2") || mime.contains("x-bzip")
        require(isZip || isTarBz2) {
            "VOICE-KWS-IMPORT-001: بستهٔ مدل باید ZIP یا TAR.BZ2 باشد؛ فایل شناسایی‌شده: ${displayName.ifBlank { "نامشخص" }}."
        }
        val staging = File(root, ".import-${System.currentTimeMillis()}")
        check(staging.mkdirs()) { "VOICE-KWS-IMPORT-002: پوشهٔ موقت مدل بیدارباش ساخته نشد." }
        var count = 0
        var expanded = 0L
        var lastReportedBytes = 0L
        fun extract(nameValue: String, isDirectory: Boolean, entryStream: java.io.InputStream) {
            count++
            if (count == 1 || count % 32 == 0) {
                onProgress("در حال استخراج بستهٔ بیدارباش… فایل $count")
            }
            require(count <= MAX_ENTRIES) { "VOICE-KWS-IMPORT-004: تعداد فایل‌های بسته بیش از حد مجاز است." }
            val name = nameValue.replace('\\', '/')
            require(!name.startsWith("/") && name.split('/').none { it == ".." }) {
                "VOICE-KWS-IMPORT-005: مسیر نامعتبر داخل بستهٔ مدل وجود دارد."
            }
            val target = File(staging, name).canonicalFile
            require(target.toPath().startsWith(staging.canonicalFile.toPath())) {
                "VOICE-KWS-IMPORT-005: مسیر نامعتبر داخل بستهٔ مدل وجود دارد."
            }
            // The official 38 MB release includes test WAVs and duplicate chunk-8/FP32 graphs.
            // They are not needed for the app's chunk-16 INT8 KWS runtime. Consume skipped
            // entries to keep the archive stream valid, but do not write them to internal storage.
            val lowerName = name.lowercase()
            val inTestWavs = lowerName.split('/').any { it == "test_wavs" }
            val keepOfficialEntry = when {
                isDirectory -> !inTestWavs
                inTestWavs -> false
                lowerName.endsWith(".onnx") -> when {
                    lowerName.contains("encoder") -> lowerName.contains("chunk-16-left-64") && lowerName.contains("int8")
                    lowerName.contains("decoder") -> lowerName.contains("chunk-16-left-64") && !lowerName.contains("int8")
                    lowerName.contains("joiner") -> lowerName.contains("chunk-16-left-64") && lowerName.contains("int8")
                    else -> false
                }
                else -> lowerName.endsWith("/tokens.txt") || lowerName == "tokens.txt" ||
                    lowerName.endsWith("/en.phone") || lowerName == "en.phone" ||
                    lowerName.endsWith("/keywords.txt") || lowerName == "keywords.txt"
            }
            if (isOfficialZhEnArchive && !keepOfficialEntry) {
                if (!isDirectory) {
                    val buffer = ByteArray(256 * 1024)
                    while (true) {
                        val n = entryStream.read(buffer)
                        if (n < 0) break
                        expanded += n
                        if (expanded - lastReportedBytes >= 8L * 1024 * 1024) {
                            lastReportedBytes = expanded
                            onProgress("در حال بازکردن بستهٔ بیدارباش… ${expanded / (1024 * 1024)} مگابایت")
                        }
                        require(expanded <= MAX_EXPANDED_BYTES) {
                            "VOICE-KWS-IMPORT-006: حجم استخراج‌شدهٔ مدل از حد مجاز بیشتر است."
                        }
                    }
                }
                return
            }
            if (isDirectory) {
                target.mkdirs()
            } else {
                target.parentFile?.mkdirs()
                FileOutputStream(target).use { output ->
                    val buffer = ByteArray(256 * 1024)
                    while (true) {
                        val n = entryStream.read(buffer)
                        if (n < 0) break
                        expanded += n
                        if (expanded - lastReportedBytes >= 2L * 1024 * 1024) {
                            lastReportedBytes = expanded
                            onProgress("در حال استخراج مدل بیدارباش… ${expanded / (1024 * 1024)} مگابایت")
                        }
                        require(expanded <= MAX_EXPANDED_BYTES) {
                            "VOICE-KWS-IMPORT-006: حجم استخراج‌شدهٔ مدل از حد مجاز بیشتر است."
                        }
                        output.write(buffer, 0, n)
                    }
                }
            }
        }
        fun hasOfficialRuntimeSet(): Boolean {
            if (!isOfficialZhEnArchive || officialTokensProvider == null) return false
            return staging.walkTopDown().filter { it.isDirectory }.any { directory ->
                val files = directory.listFiles()?.filter { it.isFile }.orEmpty()
                fun hasOnnx(role: String, int8: Boolean): Boolean = files.any { file ->
                    val name = file.name.lowercase()
                    name.endsWith(".onnx") &&
                        name.contains(role) &&
                        name.contains("chunk-16-left-64") &&
                        name.contains("int8") == int8
                }
                hasOnnx("encoder", int8 = true) &&
                    hasOnnx("decoder", int8 = false) &&
                    hasOnnx("joiner", int8 = true) &&
                    files.any { it.name.equals("en.phone", true) }
            }
        }

        try {
            val input = resolver.openInputStream(uri)
                ?: error("VOICE-KWS-IMPORT-003: فایل انتخاب‌شده قابل خواندن نیست.")
            onProgress("در حال بازکردن و استخراج بستهٔ بیدارباش…")
            input.use { source ->
                if (isZip) {
                    ZipInputStream(BufferedInputStream(source, ARCHIVE_BUFFER_BYTES)).use { zip ->
                        while (true) {
                            val item = zip.nextEntry ?: break
                            extract(item.name, item.isDirectory, zip)
                            zip.closeEntry()
                            // The upstream archive bundles large test WAVs and duplicate graphs.
                            // Once the exact runtime graph and its vocabulary are present, do not
                            // decompress unrelated trailing entries just to reach archive EOF.
                            if (isOfficialZhEnArchive &&
                                (item.name.endsWith(".onnx", true) ||
                                    item.name.endsWith("tokens.txt", true) ||
                                    item.name.endsWith("en.phone", true)) &&
                                hasOfficialRuntimeSet()
                            ) break
                        }
                    }
                } else {
                    TarArchiveInputStream(BZip2CompressorInputStream(BufferedInputStream(source, ARCHIVE_BUFFER_BYTES), true)).use { tar ->
                        while (true) {
                            val item = tar.nextTarEntry ?: break
                            extract(item.name, item.isDirectory, tar)
                            // BZip2 is expensive to decode on mid-range Android devices. The
                            // official archive stores test audio and duplicate models after the
                            // runtime files; stop as soon as the complete compatible set is found.
                            if (isOfficialZhEnArchive &&
                                (item.name.endsWith(".onnx", true) ||
                                    item.name.endsWith("tokens.txt", true) ||
                                    item.name.endsWith("en.phone", true)) &&
                                hasOfficialRuntimeSet()
                            ) break
                        }
                    }
                }
            }

            // In the upstream tarball tokens.txt comes after the large test_wavs directory.
            // Ship its exact 263-token table with the app so import can stop after the three
            // compatible ONNX graphs and en.phone instead of decoding all test audio first.
            if (isOfficialZhEnArchive && officialTokensProvider != null) {
                val runtimeDirectory = staging.walkTopDown().filter { it.isDirectory }.firstOrNull { directory ->
                    val files = directory.listFiles()?.filter { it.isFile }.orEmpty()
                    files.any { it.name.contains("encoder", true) && it.name.contains("chunk-16-left-64", true) && it.name.contains("int8", true) && it.extension.equals("onnx", true) } &&
                        files.any { it.name.contains("decoder", true) && it.name.contains("chunk-16-left-64", true) && !it.name.contains("int8", true) && it.extension.equals("onnx", true) } &&
                        files.any { it.name.contains("joiner", true) && it.name.contains("chunk-16-left-64", true) && it.name.contains("int8", true) && it.extension.equals("onnx", true) } &&
                        files.any { it.name.equals("en.phone", true) }
                }
                if (runtimeDirectory != null && runtimeDirectory.listFiles().orEmpty().none { it.name.equals("tokens.txt", true) }) {
                    val tokensFile = File(runtimeDirectory, "tokens.txt")
                    officialTokensProvider.invoke().use { source ->
                        FileOutputStream(tokensFile).use { output -> source.copyTo(output) }
                    }
                    check(tokensFile.length() > 0L && tokensFile.readLines(Charsets.UTF_8).size == 263) {
                        "VOICE-KWS-IMPORT-010: جدول واژگان داخلی مدل zh-en معتبر نیست."
                    }
                }
            }

            onProgress("در حال بررسی فایل‌های مدل و آماده‌سازی عبارت «دینا»…")
            // The official zh-en KWS archive ships redundant chunk-8/chunk-16 graphs and
            // test audio that are not needed at runtime. Prune those only for this known archive;
            // leave arbitrary user-provided model packages unchanged.
            val isOfficialZhEn = isOfficialZhEnArchive
            if (isOfficialZhEn) {
                staging.walkTopDown().filter { it.isDirectory }.forEach { directory ->
                    directory.listFiles()?.filter { it.isDirectory && it.name.equals("test_wavs", true) }?.forEach {
                        it.deleteRecursively()
                    }
                    directory.listFiles()?.filter { it.isFile && it.extension.equals("onnx", true) }?.forEach { modelFile ->
                        val name = modelFile.name.lowercase()
                        val isChunk16 = name.contains("chunk-16-left-64")
                        val keep = when {
                            name.contains("encoder") -> isChunk16 && name.contains("int8")
                            name.contains("decoder") -> isChunk16 && !name.contains("int8")
                            name.contains("joiner") -> isChunk16 && name.contains("int8")
                            else -> true
                        }
                        if (!keep) modelFile.delete()
                    }
                }
            }

            // The official tiny zh-en KWS model has an English phoneme inventory but no Persian
            // keyword file. For that specific model family, seed a phonetic approximation of
            // Persian "دینا" (DEE-NAA); custom-trained Persian packages should ship their own
            // keywords.txt and are never overwritten when they already define Dina.
            staging.walkTopDown().filter { it.isDirectory }.forEach { directory ->
                val directFiles = directory.listFiles()?.filter { it.isFile }.orEmpty()
                val hasKwsParts = directFiles.any { it.name.contains("encoder", true) && it.extension.equals("onnx", true) } &&
                    directFiles.any { it.name.contains("decoder", true) && it.extension.equals("onnx", true) } &&
                    directFiles.any { it.name.contains("joiner", true) && it.extension.equals("onnx", true) } &&
                    directFiles.any { it.name.equals("tokens.txt", true) }
                val supportsEnglishPhones = directory.name.contains("zh-en-3m", true) ||
                    directFiles.any { it.name.equals("en.phone", true) }
                val keywordFile = File(directory, "keywords.txt")
                if (hasKwsParts && supportsEnglishPhones) {
                    // Use the exact token format generated by sherpa-onnx text2token
                    // (phone+ppinyin with en.phone), not hand-added per-line score syntax.
                    // Keep a small set of English-phone realizations of Persian /diːnɑː/.
                    keywordFile.writeText(
                        // English-phone approximations for common Persian /dina/ realizations.
                        // These aliases improve recall but cannot replace a Persian-trained KWS model.
                        "D IY1 N AA1 @دینا\nD IY1 N AH0 @دینا\nD IY0 N AA1 @دینا\nD IH1 N AA1 @دینا\nD IH1 N AH0 @دینا\n",
                        Charsets.UTF_8,
                    )
                }
            }

            onProgress("در حال اعتبارسنجی encoder، decoder، joiner و واژگان…")
            val candidate = staging.walkTopDown().filter { it.isDirectory }
                .firstOrNull { validateDirectory(it) != null }
                ?: error("VOICE-KWS-IMPORT-007: بسته معتبر نیست. encoder/decoder/joiner، tokens.txt و keywords.txt برای «دینا» لازم‌اند.")
            val destination = File(root, "wake-dina-${System.currentTimeMillis()}")
            if (candidate == staging) {
                check(destination.mkdirs()) { "VOICE-KWS-IMPORT-008: پوشهٔ نهایی مدل بیدارباش ساخته نشد." }
                candidate.listFiles().orEmpty().forEach { child ->
                    check(child.renameTo(File(destination, child.name))) {
                        "VOICE-KWS-IMPORT-008: انتقال فایل ${child.name} به پوشهٔ نهایی ناموفق بود."
                    }
                }
            } else {
                check(candidate.renameTo(destination)) { "VOICE-KWS-IMPORT-008: ذخیرهٔ مدل بیدارباش ناموفق بود." }
            }
            onProgress("در حال ذخیره نهایی مدل بیدارباش…")
            staging.deleteRecursively()
            val validated = validateDirectory(destination)
                ?: run {
                    destination.deleteRecursively()
                    error("VOICE-KWS-IMPORT-009: اعتبارسنجی نهایی مدل بیدارباش ناموفق بود.")
                }
            val elapsedSeconds = (System.nanoTime() - importStartedAtNanos) / 1_000_000_000.0
            onProgress("مدل بیدارباش آماده شد؛ زمان واردسازی: %.1f ثانیه".format(java.util.Locale.ROOT, elapsedSeconds))
            return validated
        } catch (t: Throwable) {
            staging.deleteRecursively()
            throw t
        }
    }

    private fun validateDirectory(directory: File): WakeWordModelEntry? {
        if (!directory.isDirectory) return null
        val files = directory.listFiles()?.filter { it.isFile }.orEmpty()
        val onnxFiles = files.filter { it.extension.equals("onnx", true) }
        val encoders = onnxFiles.filter { it.name.contains("encoder", true) }
        val decoders = onnxFiles.filter { it.name.contains("decoder", true) }
        val joiners = onnxFiles.filter { it.name.contains("joiner", true) }
        if (encoders.isEmpty() || decoders.isEmpty() || joiners.isEmpty()) return null

        // KWS releases can contain chunk-8 and chunk-16 graphs side by side, plus
        // FP32 and INT8 encoders/joiners. Select a compatible set, not each role independently.
        fun chunkKey(file: File): String? =
            Regex("chunk-(8|16)-left-\\d+", RegexOption.IGNORE_CASE)
                .find(file.name)?.value?.lowercase()
        val preferredEncoderChunk = encoders.firstOrNull { chunkKey(it) == "chunk-16-left-64" }?.let(::chunkKey)
            ?: encoders.firstOrNull { chunkKey(it) != null }?.let(::chunkKey)
        fun matching(roleFiles: List<File>): List<File> =
            if (preferredEncoderChunk == null) roleFiles
            else roleFiles.filter { chunkKey(it) == preferredEncoderChunk }.ifEmpty { roleFiles }
        val matchingEncoders = matching(encoders)
        val matchingDecoders = matching(decoders)
        val matchingJoiners = matching(joiners)
        val encoder = matchingEncoders.firstOrNull { it.name.contains("int8", true) }
            ?: matchingEncoders.firstOrNull() ?: return null
        // The upstream zh-en release has FP32 decoders; pair every component with the same chunk size.
        val decoder = matchingDecoders.firstOrNull { !it.name.contains("int8", true) }
            ?: matchingDecoders.firstOrNull() ?: return null
        val joiner = if (encoder.name.contains("int8", true)) {
            matchingJoiners.firstOrNull { it.name.contains("int8", true) } ?: matchingJoiners.firstOrNull()
        } else {
            matchingJoiners.firstOrNull { !it.name.contains("int8", true) } ?: matchingJoiners.firstOrNull()
        } ?: return null
        val tokens = files.firstOrNull { it.name.equals("tokens.txt", true) } ?: return null
        val keywords = files.firstOrNull { it.name.equals("keywords.txt", true) } ?: return null
        if (listOf(encoder, decoder, joiner).any { it.length() < 1024L } || tokens.length() == 0L) return null
        val keywordText = runCatching { keywords.readText(Charsets.UTF_8) }.getOrNull() ?: return null
        if (!keywordText.contains("دینا")) return null
        val configFile = files.firstOrNull { it.name.equals("wakeword.json", true) }
        val config = configFile?.let { runCatching { JSONObject(it.readText(Charsets.UTF_8)) }.getOrNull() }
        val bpeVocab = files.firstOrNull { it.name.equals("bpe.model", true) }
        val inferredModelType = when {
            encoder.name.contains("zipformer2", true) -> "zipformer2"
            encoder.name.contains("lstm", true) -> "lstm"
            encoder.name.contains("conformer", true) -> "conformer"
            // Standard Zipformer KWS uses an empty model_type in sherpa-onnx.
            else -> ""
        }
        val modelType = config?.optString("modelType")?.takeIf {
            it in setOf("", "zipformer", "zipformer2", "lstm", "conformer")
        } ?: inferredModelType
        val modelingUnit = config?.optString("modelingUnit")?.takeIf { it.isNotBlank() }
            ?: when {
                bpeVocab != null -> "bpe"
                files.any { it.name.equals("en.phone", true) } -> "cjkchar"
                else -> ""
            }
        return WakeWordModelEntry(
            id = directory.name,
            title = "بیدارباش «دینا» · sherpa-onnx KWS",
            directory = directory,
            encoder = encoder,
            decoder = decoder,
            joiner = joiner,
            tokens = tokens,
            keywords = keywords,
            modelType = modelType,
            modelingUnit = modelingUnit,
            bpeVocab = bpeVocab,
        )
    }

    companion object {
        private const val MAX_ENTRIES = 4_000
        // SAF-backed document streams may cross a Binder boundary for every small read.
        // Buffer before ZIP/BZip2 decoding to avoid thousands of tiny provider reads.
        private const val ARCHIVE_BUFFER_BYTES = 256 * 1024
        private const val MAX_EXPANDED_BYTES = 768L * 1024 * 1024
    }
}
