package com.samanramezani.aichattest

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.view.Gravity
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import com.woogit.aicore.domain.ModelResult
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch

class MainActivity : Activity() {
    private lateinit var container: AppContainer
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        container = (application as? AIChatApplication)?.container ?: AppContainer(this)
        showWorkspace()
    }

    override fun onDestroy() {
        scope.cancel()
        super.onDestroy()
    }

    private fun showWorkspace() {
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(40, 48, 40, 32) }
        root.addView(TextView(this).apply { text = "AI Chat Test"; textSize = 26f })
        val modelStatus = TextView(this).apply { textSize = 16f; setPadding(0, 16, 0, 24) }
        root.addView(modelStatus)
        root.addView(sectionButton("Chat") { showPlaceholder("Chat", "Chat آماده اتصال به مدل فعال است.") })
        root.addView(sectionButton("Workspace") { showPlaceholder("Workspace", "میز کار آماده اتصال به Task و Action Core است.") })
        root.addView(sectionButton("Errors") { showPlaceholder("Error Center", "مرکز خطا از Central Observability تغذیه خواهد شد.") })
        root.addView(sectionButton("Settings") { showSettings() })
        setContentView(root)
        refreshModelStatus(modelStatus)
    }

    private fun showSettings() {
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(40, 40, 40, 40) }
        root.addView(TextView(this).apply { text = "Settings → AI → Model"; textSize = 24f })
        root.addView(TextView(this).apply { text = "مدل داخل APK قرار ندارد. فایل GGUF را از گوشی Import کنید."; textSize = 16f; setPadding(0, 20, 0, 20) })
        root.addView(Button(this).apply { text = "Import Model (GGUF)"; setOnClickListener { openModelPicker() } })
        root.addView(Button(this).apply { text = "مدل‌های واردشده"; setOnClickListener { showModels() } })
        root.addView(Button(this).apply { text = "بازگشت"; setOnClickListener { showWorkspace() } })
        setContentView(root)
    }

    private fun openModelPicker() {
        startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "application/octet-stream"
        }, REQUEST_MODEL)
    }

    @Deprecated("Use Activity Result API when this screen is migrated to Compose")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != REQUEST_MODEL || resultCode != RESULT_OK) return
        val uri = data?.data ?: return
        val manager = container.modelManager ?: return showMessage("خطا", "Model Manager در دسترس نیست.")
        showMessage("Import", "در حال بررسی و وارد کردن فایل GGUF…")
        scope.launch {
            when (val result = manager.import(uri)) {
                is ModelResult.Success -> showMessage("Import موفق", "مدل «${result.value.displayName}» با موفقیت وارد شد.")
                is ModelResult.Failure -> showMessage("Import ناموفق", result.error.message)
            }
        }
    }

    private fun showModels() {
        val manager = container.modelManager ?: return showMessage("خطا", "Model Manager در دسترس نیست.")
        showMessage("مدل‌ها", "در حال خواندن فهرست مدل‌ها…")
        scope.launch {
            when (val result = manager.models()) {
                is ModelResult.Success -> {
                    val text = if (result.value.isEmpty()) "هنوز مدلی وارد نشده است." else
                        result.value.joinToString("\n\n") { "${it.displayName}\n${it.quantization}\n${it.sizeBytes} bytes" }
                    showMessage("مدل‌های واردشده", text)
                }
                is ModelResult.Failure -> showMessage("خطا", result.error.message)
            }
        }
    }

    private fun refreshModelStatus(view: TextView) {
        val manager = container.modelManager ?: run { view.text = "Model Manager در دسترس نیست"; return }
        scope.launch {
            when (val result = manager.models()) {
                is ModelResult.Success -> view.text = "مدل‌های واردشده: ${result.value.size}"
                is ModelResult.Failure -> view.text = "خطای مدل: ${result.error.message}"
            }
        }
    }

    private fun sectionButton(label: String, action: () -> Unit) = Button(this).apply { text = label; setOnClickListener { action() } }

    private fun showMessage(title: String, message: String) {
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; gravity = Gravity.CENTER; setPadding(40, 40, 40, 40) }
        root.addView(TextView(this).apply { text = title; textSize = 24f })
        root.addView(TextView(this).apply { text = message; textSize = 16f; setPadding(0, 20, 0, 20) })
        root.addView(Button(this).apply { text = "بازگشت به Settings"; setOnClickListener { showSettings() } })
        setContentView(root)
    }

    private fun showPlaceholder(title: String, message: String) = showMessage(title, message)

    companion object { private const val REQUEST_MODEL = 1001 }
}
