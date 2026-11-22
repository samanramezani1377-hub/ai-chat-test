package com.samanramezani.gpuenginetest

import android.app.Activity
import android.app.ActivityManager
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.OpenableColumns
import android.view.Gravity
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import com.samanramezani.gpuengine.GpuEngineTestApi
import java.io.File
import java.io.FileInputStream
import java.util.Locale
import kotlin.concurrent.thread

class MainActivity : Activity() {
    private lateinit var logView: EditText
    private lateinit var statusView: TextView
    private lateinit var importButton: Button
    private val logBuffer = StringBuilder()
    private val requestImportModel = 4101
    private val contextLength = 2048

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        buildUi()
        appendLog("GPU_ENGINE_TEST_APP_STARTED")
        appendLog("device=${Build.MANUFACTURER} ${Build.MODEL}")
        appendLog("android_sdk=${Build.VERSION.SDK_INT}")
        appendLog("supported_abis=${Build.SUPPORTED_ABIS.joinToString(",")}")
        appendLog("app_private_storage=${filesDir.absolutePath}")
        appendLog("model_picker=Android Storage Access Framework")
        appendLog("context_length=$contextLength")
        appendLog("vulkan=disabled")
    }

    private fun buildUi() {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(16), dp(16), dp(12))
        }

        val title = TextView(this).apply {
            text = "Android GPU Engine Test"
            textSize = 21f
            gravity = Gravity.CENTER_VERTICAL
        }
        root.addView(title, LinearLayout.LayoutParams(-1, dp(48)))

        statusView = TextView(this).apply {
            text = "No model loaded"
            textSize = 14f
            setPadding(0, dp(8), 0, dp(8))
        }
        root.addView(statusView)

        importButton = Button(this).apply {
            text = "Import GGUF model and activate GPU engine"
            setOnClickListener { openModelPicker() }
        }
        root.addView(importButton, LinearLayout.LayoutParams(-1, -2))

        val actions = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        val copyButton = Button(this).apply {
            text = "Copy full log"
            setOnClickListener { copyLog() }
        }
        val unloadButton = Button(this).apply {
            text = "Unload model"
            setOnClickListener { unloadModel() }
        }
        actions.addView(copyButton, LinearLayout.LayoutParams(0, -2, 1f))
        actions.addView(unloadButton, LinearLayout.LayoutParams(0, -2, 1f))
        root.addView(actions)

        logView = EditText(this).apply {
            textSize = 11f
            minLines = 12
            typeface = android.graphics.Typeface.MONOSPACE
            setTextIsSelectable(true)
            isSingleLine = false
            gravity = Gravity.TOP or Gravity.START
            setHorizontallyScrolling(false)
            setPadding(dp(8), dp(8), dp(8), dp(8))
            hint = "Full diagnostic log will appear here"
        }
        val scroll = ScrollView(this).apply {
            isFillViewport = true
            addView(logView, LinearLayout.LayoutParams(-1, -2))
        }
        root.addView(scroll, LinearLayout.LayoutParams(-1, 0, 1f))
        setContentView(root)
    }

    private fun openModelPicker() {
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "*/*"
            putExtra(Intent.EXTRA_MIME_TYPES, arrayOf("application/octet-stream", "application/x-gguf"))
        }
        startActivityForResult(intent, requestImportModel)
    }

    @Deprecated("The simple test app uses the platform document picker to avoid extra UI dependencies.")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != requestImportModel || resultCode != RESULT_OK) return
        val uri = data?.data ?: return
        importButton.isEnabled = false
        statusView.text = "Importing model…"
        thread(name = "gpu-engine-model-import") {
            runCatching { importAndActivate(uri) }
                .onFailure { error ->
                    appendLog("APP_ERROR=${error::class.java.name}: ${error.message}")
                    runOnUiThread {
                        statusView.text = "Import/activation failed"
                        importButton.isEnabled = true
                    }
                }
        }
    }

    private fun importAndActivate(uri: Uri) {
        val displayName = queryDisplayName(uri) ?: "model.gguf"
        val safeName = displayName.replace(Regex("[^A-Za-z0-9._-]"), "_")
        val modelDirectory = File(filesDir, "models").apply { mkdirs() }
        val modelFile = File(modelDirectory, "${System.currentTimeMillis()}_$safeName")

        appendLog("MODEL_IMPORT_BEGIN")
        appendLog("source_uri=$uri")
        appendLog("source_name=$displayName")
        contentResolver.openInputStream(uri)?.use { input ->
            modelFile.outputStream().use { output -> input.copyTo(output) }
        } ?: error("Could not open selected model URI")

        val magic = ByteArray(4)
        FileInputStream(modelFile).use { input ->
            check(input.read(magic) == 4 && String(magic, Charsets.US_ASCII) == "GGUF") {
                "Selected file is not a GGUF model"
            }
        }

        appendLog("imported_model_path=${modelFile.absolutePath}")
        appendLog("model_size_bytes=${modelFile.length()}")
        appendLog("model_size_mib=${String.format(Locale.US, "%.2f", modelFile.length() / 1048576.0)}")

        val activityManager = getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memoryInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memoryInfo)
        val memoryBudget = (memoryInfo.availMem - memoryInfo.threshold).coerceAtLeast(0L) / 4L

        appendLog("device_total_memory_bytes=${memoryInfo.totalMem}")
        appendLog("device_available_memory_bytes=${memoryInfo.availMem}")
        appendLog("device_memory_threshold_bytes=${memoryInfo.threshold}")
        appendLog("device_low_memory=${memoryInfo.lowMemory}")
        appendLog("gpu_memory_budget_bytes=$memoryBudget")
        appendLog("requested_context_length=$contextLength")
        appendLog("activation_backend=dedicated_android_gpu_engine_only")

        runOnUiThread { statusView.text = "Activating GPU engine…" }
        val startedAt = System.nanoTime()
        val result = GpuEngineTestApi.loadModel(modelFile.absolutePath, contextLength, memoryBudget)
        val elapsedMs = (System.nanoTime() - startedAt) / 1_000_000L
        val nativeReport = GpuEngineTestApi.diagnostics()

        appendLog("GPU_ENGINE_LOAD_RESULT=$result")
        appendLog("GPU_ENGINE_LOAD_TIME_MS=$elapsedMs")
        appendLog("GPU_ENGINE_DIAGNOSTICS_BEGIN")
        appendLog(nativeReport)
        appendLog("GPU_ENGINE_DIAGNOSTICS_END")

        runOnUiThread {
            statusView.text = if (result == 0) "GPU model activation succeeded" else "GPU model activation failed (code $result)"
            importButton.isEnabled = true
        }
    }

    private fun queryDisplayName(uri: Uri): String? {
        contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { cursor ->
            if (cursor.moveToFirst()) {
                val index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                if (index >= 0) return cursor.getString(index)
            }
        }
        return null
    }

    private fun unloadModel() {
        thread(name = "gpu-engine-unload") {
            GpuEngineTestApi.unload()
            appendLog("GPU_ENGINE_UNLOAD_REQUESTED")
            appendLog(GpuEngineTestApi.diagnostics())
            runOnUiThread { statusView.text = "Model unloaded" }
        }
    }

    private fun appendLog(line: String) {
        synchronized(logBuffer) {
            logBuffer.append(line).append('\n')
        }
        runOnUiThread {
            logView.setText(logBuffer.toString())
            logView.setSelection(logView.text.length)
        }
    }

    private fun copyLog() {
        val clipboard = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        val content = synchronized(logBuffer) { logBuffer.toString() }
        clipboard.setPrimaryClip(ClipData.newPlainText("GPU Engine diagnostics", content))
        statusView.text = "Full diagnostic log copied"
    }

    private fun dp(value: Int): Int = (value * resources.displayMetrics.density).toInt()
}
