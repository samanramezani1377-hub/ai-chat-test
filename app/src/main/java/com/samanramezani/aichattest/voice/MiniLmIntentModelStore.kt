package com.samanramezani.aichattest.voice

import android.content.ContentResolver
import android.content.res.AssetManager
import android.net.Uri
import android.provider.OpenableColumns
import java.io.File
import java.io.FileOutputStream
import java.util.zip.ZipInputStream
import org.json.JSONArray
import org.json.JSONObject

/**
 * Stores a user-supplied, fine-tuned MiniLM intent-classifier bundle locally.
 * Accepted ZIPs:
 * - fine-tuned classifier: model.onnx, tokenizer.json, labels.json
 * - sentence-embedding decision tree: model.onnx, tokenizer.json, decision_trees.json
 * all-MiniLM-L6-v2 is English-trained; Persian input is routed to safe Persian rules.
 */
internal class MiniLmIntentModelStore(
    private val resolver: ContentResolver,
    private val root: File,
) {
    /**
     * Seeds the editable app-private decision tree from the version-controlled asset.
     * Never overwrites an existing file: imported/customized trees survive app restarts.
     */
    fun ensureDefaultDecisionTree(assets: AssetManager) {
        val target = File(root, "decision_trees.json")
        if (!root.exists() && !root.mkdirs()) {
            throw IllegalStateException("MINILM-TREE-INIT-001: پوشهٔ تنظیمات درخت تصمیم ساخته نشد.")
        }
        val bundled = assets.open("minilm/decision_trees.json").bufferedReader(Charsets.UTF_8).use { it.readText() }
        if (!target.isFile) {
            target.writeText(bundled, Charsets.UTF_8)
            return
        }

        // Upgrade older app-managed/imported trees without overwriting any existing custom choices.
        // New nodes are added only when absent, so a user-edited volume_level node is preserved.
        runCatching {
            val current = JSONObject(target.readText(Charsets.UTF_8))
            val defaults = JSONObject(bundled)
            val currentNodes = current.optJSONObject("nodes") ?: JSONObject().also { current.put("nodes", it) }
            val defaultNodes = defaults.optJSONObject("nodes") ?: return@runCatching
            val keys = defaultNodes.keys()
            var changed = false
            while (keys.hasNext()) {
                val key = keys.next()
                if (!currentNodes.has(key)) {
                    currentNodes.put(key, defaultNodes.getJSONObject(key))
                    changed = true
                }
            }
            if (changed) {
                current.put("version", maxOf(current.optInt("version", 1), defaults.optInt("version", 1)))
                val temp = File(root, "decision_trees.json.tmp")
                temp.writeText(current.toString(2), Charsets.UTF_8)
                if (!temp.renameTo(target)) {
                    temp.copyTo(target, overwrite = true)
                    temp.delete()
                }
            }
        }
    }

    fun isReady(): Boolean {
        val modelReady = File(root, "model.onnx").let { it.isFile && it.length() > 1024 }
        val tokenizerReady = File(root, "tokenizer.json").let { it.isFile && it.length() > 0 }
        val classifierBundle = File(root, "labels.json").let { it.isFile && it.length() > 0 }
        val decisionTreeBundle = File(root, "decision_trees.json").let { it.isFile && it.length() > 0 }
        return modelReady && tokenizerReady && (classifierBundle || decisionTreeBundle)
    }

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
                            if (basename in setOf("model.onnx", "tokenizer.json", "labels.json", "decision_trees.json")) {
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
            val decisionTrees = File(staging, "decision_trees.json")
            val hasDecisionTrees = decisionTrees.isFile
            require(model.isFile && model.length() > 1024 && tokenizer.isFile && tokenizer.length() > 0 &&
                (hasDecisionTrees || labels.isFile && labels.length() > 0)) {
                "MINILM-IMPORT-007: بسته باید model.onnx، tokenizer.json و یکی از labels.json یا decision_trees.json را داشته باشد."
            }
            if (decisionTrees.isFile) {
                require(decisionTrees.length() <= 256 * 1024) {
                    "MINILM-IMPORT-012: فایل decision_trees.json بیش از ۲۵۶ کیلوبایت است."
                }
                val config = JSONObject(decisionTrees.readText(Charsets.UTF_8))
                require(config.optString("model_type") == "sentence_embedding") {
                    "MINILM-IMPORT-013: model_type در decision_trees.json باید sentence_embedding باشد."
                }
                val nodes = config.optJSONObject("nodes")
                require(nodes != null && listOf("domain", "audio", "display").all { id ->
                    nodes.optJSONObject(id)?.optJSONArray("choices")?.length()?.let { it > 0 } == true
                }) {
                    "MINILM-IMPORT-014: درخت تصمیم باید گره‌های domain، audio و display را داشته باشد."
                }
                require(model.length() <= 100L * 1024 * 1024) {
                    "MINILM-IMPORT-015: فایل مدل تصمیم‌گیری باید حداکثر ۱۰۰ مگابایت باشد."
                }
            } else {
                require(labels.isFile && labels.length() > 0) {
                    "MINILM-IMPORT-007: بسته باید labels.json یا decision_trees.json داشته باشد."
                }
                val parsedLabels = parseLabels(labels.readText(Charsets.UTF_8))
                val allowedLabels = setOf("CHAT", "GET_TIME", "GET_DATE", "INCREASE_VOLUME", "DECREASE_VOLUME",
                    "SET_VOLUME", "MUTE", "UNMUTE", "INCREASE_BRIGHTNESS", "DECREASE_BRIGHTNESS", "SET_BRIGHTNESS")
                require(parsedLabels.isNotEmpty() && parsedLabels.contains("CHAT") && parsedLabels.all { it in allowedLabels }) {
                    "MINILM-IMPORT-008: labels.json باید CHAT و برچسب‌های معتبر فرمان‌های پشتیبانی‌شده را تعریف کند."
                }
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
            return if (hasDecisionTrees) {
                "مدل تصمیم‌گیری مرحله‌ای وارد شد. توجه: all-MiniLM-L6-v2 انگلیسی است؛ فرمان‌های فارسی با قواعد امن دینا پردازش می‌شوند."
            } else {
                "بسته طبقه‌بند MiniLM وارد شد؛ پیش‌بینی‌های نامطمئن به قواعد امن دینا برمی‌گردند."
            }
        } catch (t: Throwable) {
            staging.deleteRecursively()
            throw t
        }
    }
}
