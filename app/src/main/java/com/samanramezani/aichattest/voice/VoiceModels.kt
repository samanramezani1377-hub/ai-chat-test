package com.samanramezani.aichattest.voice

import android.content.ContentResolver
import android.net.Uri
import android.provider.OpenableColumns
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.FileInputStream
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.util.zip.ZipInputStream
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream

enum class VoiceModelKind { STT, TTS }
enum class PiperComponent { MODEL, CONFIG, ESPEAK_DATA }
enum class SmallSttComponent { MODEL, TOKENS }
enum class SmallSttModel { RIZEH_PIZEH, KOOCHIK }

data class VoiceModelEntry(
    val id: String,
    val title: String,
    val kind: VoiceModelKind,
    val directory: File,
    val modelFile: File,
    val tokensFile: File? = null,
    val dataDirectory: File? = null,
    val encoder: File? = null,
    val decoder: File? = null,
    val joiner: File? = null,
    val modelType: String = "zipformer",
    val asrMode: String = "online-transducer",
    val convFrontend: File? = null,
    val tokenizerDirectory: File? = null,
)

/**
 * Model files are user-owned and remain on-device. Arbitrary ONNX files are never guessed
 * to be compatible. Import validated .zip packages, not a bare model file.
 *
 * TTS: converted sherpa-onnx Piper/VITS bundle with *.onnx, tokens.txt, espeak-ng-data/.
 * STT: sherpa-onnx online transducer bundle with encoder/decoder/joiner ONNX and tokens.txt.
 *
 * The original rhasspy/piper-voices ONNX needs metadata/tokens conversion before sherpa use.
 */
class VoiceModelStore(
    private val resolver: ContentResolver,
    private val root: File,
) {
    init { root.mkdirs() }

    fun list(kind: VoiceModelKind): List<VoiceModelEntry> =
        root.listFiles()?.filter { it.isDirectory && !it.name.startsWith(".") }
            ?.mapNotNull { validateDirectory(it, kind) }?.sortedBy { it.title } ?: emptyList()

    /** Human-readable progress for a multi-file Piper import; never reports a partial model as ready. */
    fun missingPiperComponents(): List<String> {
        if (list(VoiceModelKind.TTS).any { it.tokensFile != null && it.dataDirectory != null }) return emptyList()
        val staging = File(root, ".piper-import")
        val model = staging.listFiles()?.any { it.isFile && it.extension.equals("onnx", true) && it.length() > 1024 } == true
        val config = File(staging, "model.onnx.json").isFile ||
            staging.listFiles()?.any { it.isFile && it.name.endsWith(".onnx.json", true) } == true
        val espeak = hasRequiredEspeakData(File(staging, "espeak-ng-data"))
        return buildList {
            if (!model) add("فایل ONNX")
            if (!config) add("فایل JSON کنار مدل (معمولاً model.onnx.json)")
            if (!espeak) add("بسته espeak-ng-data")
        }
    }

    /** Human-readable progress for the two-file Shenava import. */
    fun missingSmallSttComponents(modelType: SmallSttModel = SmallSttModel.RIZEH_PIZEH): List<String> {
        val idPrefix = if (modelType == SmallSttModel.KOOCHIK) "stt-shenava-koochik-" else "stt-shenava-rizeh-pizeh-"
        if (list(VoiceModelKind.STT).any { it.asrMode == "nemo-ctc" && it.id.startsWith(idPrefix) }) return emptyList()
        val staging = File(root, if (modelType == SmallSttModel.KOOCHIK) ".shenava-koochik-import" else ".shenava-rizeh-import")
        val model = File(staging, "model.onnx").let { it.isFile && it.length() >= 1024 }
        val tokens = File(staging, "tokens.txt").let { it.isFile && it.length() > 0 }
        return buildList {
            if (!model) add("model.onnx")
            if (!tokens) add("tokens.txt")
        }
    }

    /**
     * Import the tiny Persian Shenava Rizeh-Pizeh NeMo-CTC model in two steps.
     * Model and tokens can be selected in either order; incomplete imports stay hidden.
     */
    fun importSmallPersianSttComponent(
        uri: Uri,
        component: SmallSttComponent,
        modelType: SmallSttModel = SmallSttModel.RIZEH_PIZEH,
    ): VoiceModelEntry? {
        root.mkdirs()
        val staging = File(root, if (modelType == SmallSttModel.KOOCHIK) ".shenava-koochik-import" else ".shenava-rizeh-import")
        if (!staging.exists() && !staging.mkdirs()) {
            error("VOICE-STT-IMPORT-001: ایجاد پوشه موقت مدل فارسی ناموفق بود.")
        }
        val displayName = resolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
            ?.use { cursor -> if (cursor.moveToFirst()) cursor.getString(0).orEmpty() else "" }.orEmpty()
        require(displayName.isNotBlank()) { "VOICE-STT-IMPORT-002: نام فایل انتخاب‌شده قابل خواندن نیست." }
        val source = resolver.openInputStream(uri)
            ?: error("VOICE-STT-IMPORT-003: فایل انتخاب‌شده قابل خواندن نیست.")
        source.use { input ->
            when (component) {
                SmallSttComponent.MODEL -> {
                    require(displayName.endsWith(".onnx", true)) {
                        "VOICE-STT-IMPORT-004: فایل مدل Shenava باید با پسوند .onnx باشد."
                    }
                    val target = File(staging, "model.onnx")
                    FileOutputStream(target).use { output ->
                        val buffer = ByteArray(64 * 1024)
                        var total = 0L
                        while (true) {
                            val n = input.read(buffer)
                            if (n < 0) break
                            total += n
                            require(total <= MAX_STT_MODEL_BYTES) {
                                "VOICE-STT-IMPORT-005: حجم مدل STT از حد مجاز بیشتر است."
                            }
                            output.write(buffer, 0, n)
                        }
                    }
                }
                SmallSttComponent.TOKENS -> {
                    require(displayName.equals("tokens.txt", true) || displayName.endsWith(".txt", true)) {
                        "VOICE-STT-IMPORT-006: فایل واژگان مدل باید tokens.txt باشد."
                    }
                    val bytes = input.readBytes()
                    require(bytes.size <= 4 * 1024 * 1024) {
                        "VOICE-STT-IMPORT-007: فایل tokens.txt بیش از حد بزرگ است."
                    }
                    val text = bytes.toString(Charsets.UTF_8)
                    val nonEmptyLines = text.lineSequence().count { it.isNotBlank() }
                    require(nonEmptyLines >= 100 && text.contains("0")) {
                        "VOICE-STT-IMPORT-008: فایل tokens.txt معتبر به نظر نمی‌رسد."
                    }
                    File(staging, "tokens.txt").writeBytes(bytes)
                }
            }
        }
        val model = File(staging, "model.onnx")
        val tokens = File(staging, "tokens.txt")
        if (!model.isFile || model.length() < 1024 || !tokens.isFile) return null
        val id = "stt-shenava-${if (modelType == SmallSttModel.KOOCHIK) "koochik" else "rizeh-pizeh"}-${System.currentTimeMillis()}"
        val destination = File(root, id)
        check(staging.renameTo(destination)) { "VOICE-STT-IMPORT-009: ذخیره نهایی مدل فارسی ناموفق بود." }
        return validateDirectory(destination, VoiceModelKind.STT)
            ?: run {
                destination.deleteRecursively()
                error("VOICE-STT-IMPORT-010: فایل‌های مدل Shenava با ساختار مورد انتظار سازگار نیستند.")
            }
    }

    fun importArchive(uri: Uri, kind: VoiceModelKind): VoiceModelEntry {
        root.mkdirs()
        val staging = File(root, ".import-${System.currentTimeMillis()}")
        if (!staging.mkdirs()) error("VOICE-IMPORT-001: ایجاد پوشه موقت مدل ناموفق بود.")
        try {
            val archiveName = resolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
                ?.use { cursor -> if (cursor.moveToFirst()) cursor.getString(0).orEmpty() else "" }.orEmpty()
            val source = resolver.openInputStream(uri)
                ?: error("VOICE-IMPORT-002: فایل انتخاب‌شده قابل خواندن نیست.")
            var count = 0
            var expandedBytes = 0L

            fun extract(nameValue: String, isDirectory: Boolean, entryStream: InputStream) {
                count++
                require(count <= MAX_ENTRIES) { "VOICE-IMPORT-003: تعداد فایل‌های بسته مدل بیش از حد مجاز است." }
                val name = nameValue.replace('\\', '/')
                require(!name.startsWith("/") && name.split('/').none { it == ".." }) {
                    "VOICE-IMPORT-004: مسیر نامعتبر داخل بسته مدل وجود دارد."
                }
                val target = File(staging, name).canonicalFile
                require(target.toPath().startsWith(staging.canonicalFile.toPath())) {
                    "VOICE-IMPORT-004: مسیر نامعتبر داخل بسته مدل وجود دارد."
                }
                if (isDirectory) {
                    target.mkdirs()
                } else {
                    target.parentFile?.mkdirs()
                    FileOutputStream(target).use { out ->
                        val buffer = ByteArray(64 * 1024)
                        while (true) {
                            val n = entryStream.read(buffer)
                            if (n < 0) break
                            expandedBytes += n
                            require(expandedBytes <= MAX_EXPANDED_BYTES) {
                                "VOICE-IMPORT-005: حجم استخراج‌شده بسته از حد مجاز بیشتر است."
                            }
                            out.write(buffer, 0, n)
                        }
                    }
                }
            }

            source.use { stream ->
                if (archiveName.endsWith(".tar.bz2", ignoreCase = true) || archiveName.endsWith(".tbz2", ignoreCase = true)) {
                    TarArchiveInputStream(BZip2CompressorInputStream(stream)).use { tar ->
                        while (true) {
                            val item = tar.nextTarEntry ?: break
                            extract(item.name, item.isDirectory, tar)
                        }
                    }
                } else {
                    ZipInputStream(stream).use { zip ->
                        while (true) {
                            val item = zip.nextEntry ?: break
                            extract(item.name, item.isDirectory, zip)
                            zip.closeEntry()
                        }
                    }
                }
            }

            val candidate = findCandidateRoot(staging, kind)
                ?: error(if (kind == VoiceModelKind.TTS)
                    "VOICE-TTS-001: بسته TTS معتبر نیست. بستهٔ تبدیل‌شدهٔ Piper/VITS همراه tokens.txt و پوشه espeak-ng-data لازم است."
                else
                    "VOICE-STT-001: بسته STT معتبر نیست. فایل‌های encoder/decoder/joiner و tokens.txt لازم است.")
            val id = "${kind.name.lowercase()}-${candidate.name.replace(Regex("[^A-Za-z0-9._-]"), "_")}-${System.currentTimeMillis()}"
            val destination = File(root, id)
            check(staging.renameTo(destination)) { "VOICE-IMPORT-006: ذخیره نهایی مدل ناموفق بود." }
            return validateDirectory(destination, kind)
                ?: run { destination.deleteRecursively(); error("VOICE-IMPORT-007: اعتبارسنجی مدل پس از وارد کردن ناموفق بود.") }
        } catch (t: Throwable) {
            staging.deleteRecursively()
            throw t
        }
    }

    /**
     * Import one component of the original Piper voice downloaded from rhasspy/piper-voices.
     * The user can select the .onnx, its .onnx.json companion, and espeak-ng-data.tar.bz2
     * separately. Once all components exist, create sherpa-onnx tokens and ONNX metadata locally.
     */
    fun importPiperComponent(uri: Uri, component: PiperComponent): VoiceModelEntry? {
        root.mkdirs()
        val staging = File(root, ".piper-import")
        staging.mkdirs()
        val displayName = resolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
            ?.use { cursor -> if (cursor.moveToFirst()) cursor.getString(0).orEmpty() else "" }.orEmpty()
        require(displayName.isNotBlank()) { "VOICE-PIPER-001: نام فایل انتخاب‌شده قابل خواندن نیست." }
        val source = resolver.openInputStream(uri)
            ?: error("VOICE-PIPER-002: فایل انتخاب‌شده قابل خواندن نیست.")
        source.use { input ->
            when (component) {
                PiperComponent.MODEL -> {
                    require(displayName.endsWith(".onnx", true)) { "VOICE-PIPER-003: برای مدل Piper یک فایل .onnx انتخاب کنید." }
                    // Normalize filenames: SAF providers and downloads often rename files,
                    // while the previous code required the JSON basename to match byte-for-byte.
                    // A canonical pair avoids a valid model silently remaining "not imported".
                    File(staging, "model.onnx").outputStream().use(input::copyTo)
                }
                PiperComponent.CONFIG -> {
                    require(displayName.endsWith(".onnx.json", true) || displayName.endsWith(".json", true)) {
                        "VOICE-PIPER-004: فایل پیکربندی Piper با پسوند JSON انتخاب کنید."
                    }
                    File(staging, "model.onnx.json").outputStream().use(input::copyTo)
                }
                PiperComponent.ESPEAK_DATA -> {
                    val archive = displayName.lowercase()
                    require(archive.endsWith(".tar.bz2") || archive.endsWith(".tbz2") || archive.endsWith(".zip")) {
                        "VOICE-PIPER-005: پوشهٔ آواشناسی باید به‌صورت .tar.bz2 یا .zip باشد."
                    }
                    extractEspeakArchive(input, archive, staging)
                    normalizeEspeakDirectory(staging)
                }
            }
        }
        val model = File(staging, "model.onnx")
        if (!model.isFile || model.length() <= 1024) return null
        val config = File(staging, "model.onnx.json")
        if (!config.isFile || config.length() == 0L) return null
        val espeak = File(staging, "espeak-ng-data")
        requireEspeakData(espeak, "VOICE-PIPER-017")

        val json = try {
            JSONObject(config.readText(Charsets.UTF_8))
        } catch (t: Throwable) {
            error("VOICE-PIPER-006: فایل JSON مدل Piper خراب یا نامعتبر است: ${t.message ?: t.javaClass.simpleName}")
        }
        require(json.has("phoneme_id_map") && json.has("audio") && json.has("language") && json.has("espeak")) {
            "VOICE-PIPER-006: پیکربندی Piper فاقد phoneme_id_map یا اطلاعات زبان/صدا است."
        }
        val phonemeMap = json.getJSONObject("phoneme_id_map")
        val tokens = mutableListOf<Pair<Int, String>>()
        val keys = phonemeMap.keys()
        while (keys.hasNext()) {
            val symbol = keys.next()
            val ids = phonemeMap.getJSONArray(symbol)
            // Match sherpa-onnx's official Piper conversion: one token entry per
            // phoneme, using the first ID in Piper's phoneme_id_map array.
            if (ids.length() > 0) tokens += ids.getInt(0) to symbol
        }
        require(tokens.isNotEmpty() && tokens.any { it.first == 0 }) {
            "VOICE-PIPER-006: نگاشت واج‌های مدل خالی است یا شناسهٔ صفر ندارد."
        }
        File(staging, "tokens.txt").writeText(tokens.sortedBy { it.first }.joinToString("\n") { "${it.second} ${it.first}" } + "\n", Charsets.UTF_8)

        val language = json.optJSONObject("language")?.optString("name_english", "Persian") ?: "Persian"
        val voice = json.optJSONObject("espeak")?.optString("voice", "fa") ?: "fa"
        val speakers = json.optInt("num_speakers", 1)
        val sampleRate = json.optJSONObject("audio")?.optInt("sample_rate", 22050) ?: 22050
        appendOnnxMetadata(model, mapOf(
            "model_type" to "vits",
            "comment" to "piper",
            "language" to language,
            "voice" to voice,
            "has_espeak" to "1",
            "n_speakers" to speakers.toString(),
            "sample_rate" to sampleRate.toString(),
        ))

        val id = "tts-${model.nameWithoutExtension}-${System.currentTimeMillis()}"
        val destination = File(root, id)
        check(staging.renameTo(destination)) { "VOICE-PIPER-007: ذخیره نهایی مدل Piper ناموفق بود." }
        return validateDirectory(destination, VoiceModelKind.TTS)
            ?: run { destination.deleteRecursively(); error("VOICE-PIPER-008: اعتبارسنجی مدل Piper پس از تبدیل ناموفق بود.") }
    }

    private fun extractEspeakArchive(input: InputStream, archiveName: String, destination: File) {
        var count = 0
        var total = 0L
        fun writeEntry(nameValue: String, isDirectory: Boolean, stream: InputStream) {
            count++
            require(count <= MAX_ENTRIES) { "VOICE-PIPER-009: تعداد فایل‌های بسته آواشناسی بیش از حد مجاز است." }
            val name = nameValue.replace('\\', '/')
            require(!name.startsWith("/") && name.split('/').none { it == ".." }) { "VOICE-PIPER-010: مسیر نامعتبر در بسته آواشناسی وجود دارد." }
            val target = File(destination, name).canonicalFile
            require(target.toPath().startsWith(destination.canonicalFile.toPath())) { "VOICE-PIPER-010: مسیر نامعتبر در بسته آواشناسی وجود دارد." }
            if (isDirectory) target.mkdirs() else {
                target.parentFile?.mkdirs()
                FileOutputStream(target).use { out ->
                    val buffer = ByteArray(64 * 1024)
                    while (true) {
                        val n = stream.read(buffer)
                        if (n < 0) break
                        total += n
                        require(total <= MAX_EXPANDED_BYTES) { "VOICE-PIPER-011: حجم بسته آواشناسی بیش از حد مجاز است." }
                        out.write(buffer, 0, n)
                    }
                }
            }
        }
        if (archiveName.endsWith(".zip")) {
            ZipInputStream(input).use { zip ->
                while (true) {
                    val item = zip.nextEntry ?: break
                    writeEntry(item.name, item.isDirectory, zip)
                    zip.closeEntry()
                }
            }
        } else {
            TarArchiveInputStream(BZip2CompressorInputStream(input)).use { tar ->
                while (true) {
                    val item = tar.nextTarEntry ?: break
                    writeEntry(item.name, item.isDirectory, tar)
                }
            }
        }
    }

    /**
     * Archives from different distributors either include an espeak-ng-data/ prefix or
     * put the data directly under a versioned root directory. Normalize both layouts.
     */
    private fun normalizeEspeakDirectory(staging: File) {
        val expected = File(staging, "espeak-ng-data")
        if (expected.isDirectory && expected.list()?.isNotEmpty() == true) return
        val nested = staging.walkTopDown().firstOrNull {
            it.isDirectory && it != staging && it.name.equals("espeak-ng-data", true) &&
                it.list()?.isNotEmpty() == true
        }
        if (nested != null && nested != expected) {
            // Move the nested directory out first: it may live inside the current expected
            // directory, so deleting expected before moving it would delete the source itself.
            val normalized = File(staging, ".espeak-ng-data-normalized")
            normalized.deleteRecursively()
            check(nested.renameTo(normalized)) {
                "VOICE-PIPER-012: نتوانستم پوشه espeak-ng-data را از ساختار بسته استخراج‌شده آماده کنم."
            }
            expected.deleteRecursively()
            check(normalized.renameTo(expected)) {
                "VOICE-PIPER-016: نتوانستم پوشهٔ داده‌های آواشناسی را در محل نهایی قرار بدهم."
            }
            return
        }
        // Some bundles contain the contents at their archive root (phontab/phondata/...).
        val rootHasData = listOf("phontab", "phondata", "phonindex").any { File(staging, it).exists() }
        if (rootHasData) {
            val temp = File(staging, ".espeak-root")
            if (!temp.mkdirs()) error("VOICE-PIPER-013: ایجاد پوشهٔ داده‌های آواشناسی ناموفق بود.")
            staging.listFiles()?.filter {
                it != temp && it != expected &&
                    it.name != "model.onnx" && it.name != "model.onnx.json" && it.name != "tokens.txt"
            }?.forEach { item ->
                check(item.renameTo(File(temp, item.name))) {
                    "VOICE-PIPER-014: انتقال فایل‌های دادهٔ آواشناسی ناموفق بود."
                }
            }
            check(temp.renameTo(expected)) { "VOICE-PIPER-015: آماده‌سازی پوشهٔ آواشناسی ناموفق بود." }
        }
    }

    private fun appendOnnxMetadata(model: File, values: Map<String, String>) {
        // ONNX ModelProto metadata_props is field 14 (wire type 2). Appending protobuf
        // fields is valid because field order is not significant and preserves the graph bytes.
        FileOutputStream(model, true).use { out ->
            values.forEach { (key, value) ->
                val entry = ByteArrayOutputStream()
                writeProtoString(entry, 1, key)
                writeProtoString(entry, 2, value)
                writeVarint(out, ((14 shl 3) or 2).toLong())
                writeVarint(out, entry.size().toLong())
                entry.writeTo(out)
            }
        }
    }

    private fun writeProtoString(out: ByteArrayOutputStream, field: Int, value: String) {
        val bytes = value.toByteArray(Charsets.UTF_8)
        writeVarint(out, ((field shl 3) or 2).toLong())
        writeVarint(out, bytes.size.toLong())
        out.write(bytes)
    }

    private fun writeVarint(out: java.io.OutputStream, raw: Long) {
        var value = raw
        while ((value and 0x7f.inv().toLong()) != 0L) {
            out.write(((value and 0x7f) or 0x80).toInt())
            value = value ushr 7
        }
        out.write(value.toInt())
    }

    private fun hasRequiredEspeakData(directory: File): Boolean =
        directory.isDirectory && listOf("phontab", "phonindex", "phondata", "intonations")
            .all { File(directory, it).isFile && File(directory, it).length() > 0L }

    private fun requireEspeakData(directory: File, code: String) {
        val missing = listOf("phontab", "phonindex", "phondata", "intonations")
            .filter { !File(directory, it).let { file -> file.isFile && file.length() > 0L } }
        require(missing.isEmpty()) {
            "$code: داده‌های espeak-ng ناقص است؛ فایل‌های مفقود/خالی: ${missing.joinToString()}. بستهٔ رسمی espeak-ng-data.tar.bz2 را دوباره وارد کنید."
        }
    }

    private fun findCandidateRoot(staging: File, kind: VoiceModelKind): File? =
        staging.walkTopDown().filter { it.isDirectory }.firstOrNull { validateDirectory(it, kind) != null }

    private fun validateDirectory(directory: File, kind: VoiceModelKind): VoiceModelEntry? {
        val files = directory.walkTopDown().filter { it.isFile }.toList()
        return when (kind) {
            VoiceModelKind.TTS -> {
                val tokens = files.firstOrNull { it.name.equals("tokens.txt", true) } ?: return null
                val model = files.firstOrNull {
                    it.extension.equals("onnx", true) &&
                        !it.name.contains("encoder", true) &&
                        !it.name.contains("decoder", true) &&
                        !it.name.contains("joiner", true)
                } ?: return null
                val data = directory.walkTopDown().firstOrNull { it.isDirectory && it.name == "espeak-ng-data" }
                    ?: return null
                if (!hasRequiredEspeakData(data)) return null
                if (model.length() <= 1024 || tokens.length() == 0L) return null
                VoiceModelEntry(directory.name, model.nameWithoutExtension, kind, directory, model, tokensFile = tokens, dataDirectory = data)
            }
            VoiceModelKind.STT -> {
                val simpleCtcModel = files.firstOrNull { it.name.equals("model.onnx", true) }
                val simpleCtcTokens = files.firstOrNull { it.name.equals("tokens.txt", true) }
                val hasTransducerParts = files.any { it.name.contains("encoder", true) } &&
                    files.any { it.name.contains("decoder", true) }
                if (simpleCtcModel != null && simpleCtcTokens != null && !hasTransducerParts &&
                    simpleCtcModel.length() >= 1024 && simpleCtcTokens.length() > 0L) {
                    return VoiceModelEntry(
                        directory.name, if (directory.name.startsWith("stt-shenava-koochik-")) "Shenava Koochik · فارسی · 114M" else "Shenava Rizeh-Pizeh · فارسی · 6.9M",
                        kind, directory, simpleCtcModel, tokensFile = simpleCtcTokens,
                        modelType = "nemo-ctc", asrMode = "nemo-ctc",
                    )
                }
                val convFrontend = files.firstOrNull { it.name.equals("conv_frontend.onnx", true) }
                val tokenizer = directory.walkTopDown().firstOrNull { it.isDirectory && it.name.equals("tokenizer", true) }
                val encoder = files.firstOrNull { it.name.contains("encoder", true) && it.extension.equals("onnx", true) } ?: return null
                val decoder = files.firstOrNull { it.name.contains("decoder", true) && it.extension.equals("onnx", true) } ?: return null
                if (convFrontend != null && tokenizer != null) {
                    return VoiceModelEntry(
                        directory.name, "Qwen3-ASR · فارسی", kind, directory, convFrontend,
                        encoder = encoder, decoder = decoder, modelType = "qwen3-asr",
                        asrMode = "qwen3-asr", convFrontend = convFrontend, tokenizerDirectory = tokenizer,
                    )
                }
                val joiner = files.firstOrNull { it.name.contains("joiner", true) && it.extension.equals("onnx", true) } ?: return null
                val tokens = files.firstOrNull { it.name.equals("tokens.txt", true) } ?: return null
                val modelPath = encoder.absolutePath.lowercase()
                val modelType = when {
                    modelPath.contains("lstm") -> "lstm"
                    modelPath.contains("zipformer2") || modelPath.contains("chunk-16-left") -> "zipformer2"
                    else -> "zipformer"
                }
                VoiceModelEntry(
                    directory.name, directory.name, kind, directory, encoder, tokensFile = tokens,
                    encoder = encoder, decoder = decoder, joiner = joiner, modelType = modelType,
                    asrMode = "online-transducer",
                )
            }
        }
    }

    companion object {
        private const val MAX_STT_MODEL_BYTES = 512L * 1024 * 1024
        private const val MAX_ENTRIES = 20_000
        private const val MAX_EXPANDED_BYTES = 2L * 1024 * 1024 * 1024
    }
}
