package com.samanramezani.aichattest

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import com.woogit.aicore.domain.ChatMessage
import com.woogit.aicore.domain.InferenceSettings
import com.woogit.aicore.domain.ModelDescriptor
import com.woogit.aicore.domain.ModelResult
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.Base64

class MainActivity : Activity() {
    private lateinit var container: AppContainer
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private val history = mutableListOf<ChatLine>()
    private var generationRunning = false
    private var lastStatus = "آماده"
    private lateinit var statusView: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        container = (application as? AIChatApplication)?.container ?: AppContainer(this)
        history += loadHistory()
        showHome()
        restoreActiveModel()
    }

    override fun onDestroy() {
        scope.cancel()
        super.onDestroy()
    }

    private fun restoreActiveModel() {
        val manager = container.modelManager ?: return
        scope.launch(Dispatchers.Default) {
            val result = manager.restoreActive()
            withContext(Dispatchers.Main) {
                lastStatus = when (result) {
                    is ModelResult.Success -> result.value?.let { "مدل فعال آماده است: ${it.displayName}" } ?: "مدلی فعال نیست"
                    is ModelResult.Failure -> "بازیابی مدل ناموفق: ${result.error.message}"
                }
                if (::statusView.isInitialized) statusView.text = lastStatus
            }
        }
    }

    private fun showHome() {
        val root = baseRoot("AI Chat Test")
        statusView = TextView(this).apply {
            text = lastStatus
            textSize = 15f
            setPadding(0, 12, 0, 20)
        }
        root.addView(statusView)
        root.addView(actionButton("گفت‌وگو") { showChat() })
        root.addView(actionButton("مدل‌های محلی") { showModels() })
        root.addView(actionButton("تنظیمات مدل") { showSettings() })
        root.addView(actionButton("Diagnostics") { showDiagnostics() })
        setContentView(root)
    }

    private fun showSettings() {
        val root = baseRoot("تنظیمات → هوش مصنوعی → مدل")
        root.addView(info("مدل داخل APK بسته نمی‌شود. فایل GGUF فقط از طریق Import Model وارد حافظه خصوصی برنامه می‌شود."))
        root.addView(actionButton("Import Model (GGUF)") { openModelPicker() })
        root.addView(actionButton("مدل‌های واردشده") { showModels() })
        root.addView(actionButton("گفت‌وگو") { showChat() })
        root.addView(actionButton("بازگشت") { showHome() })
        setContentView(root)
    }

    private fun openModelPicker() {
        startActivityForResult(
            Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                addCategory(Intent.CATEGORY_OPENABLE)
                type = "application/octet-stream"
            },
            REQUEST_MODEL
        )
    }

    @Deprecated("Use Activity Result API after migrating this prototype UI")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != REQUEST_MODEL || resultCode != RESULT_OK) return
        val uri = data?.data ?: return
        val manager = container.modelManager ?: return showMessage("خطا", "Model Manager در دسترس نیست.")
        lastStatus = "در حال وارد کردن و اعتبارسنجی مدل…"
        showMessage("Import Model", lastStatus)
        scope.launch(Dispatchers.Default) {
            val result = manager.import(uri)
            when (result) {
                is ModelResult.Success -> {
                    val activated = manager.activate(result.value.id)
                    withContext(Dispatchers.Main) {
                        lastStatus = when (activated) {
                            is ModelResult.Success -> "مدل وارد و فعال شد: ${activated.value.displayName}"
                            is ModelResult.Failure -> "Import موفق بود؛ فعال‌سازی ناموفق: ${activated.error.message}"
                        }
                        showModels()
                    }
                }
                is ModelResult.Failure -> withContext(Dispatchers.Main) {
                    lastStatus = "Import ناموفق: ${result.error.message}"
                    showMessage("Import ناموفق", result.error.message)
                }
            }
        }
    }

    private fun showModels() {
        val manager = container.modelManager ?: return showMessage("خطا", "Model Manager در دسترس نیست.")
        val root = baseRoot("مدل‌های محلی")
        val list = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        root.addView(list)
        root.addView(actionButton("Import Model (GGUF)") { openModelPicker() })
        root.addView(actionButton("گفت‌وگو") { showChat() })
        root.addView(actionButton("بازگشت") { showHome() })
        setContentView(root)
        scope.launch {
            when (val result = manager.models()) {
                is ModelResult.Success -> {
                    if (result.value.isEmpty()) {
                        list.addView(info("هنوز مدل GGUF وارد نشده است."))
                    } else {
                        result.value.forEach { model -> addModelCard(list, model, manager) }
                    }
                }
                is ModelResult.Failure -> list.addView(info("خطا در خواندن مدل‌ها: ${result.error.message}"))
            }
        }
    }

    private fun addModelCard(parent: LinearLayout, model: ModelDescriptor, manager: AndroidModelManager) {
        val card = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(0, 12, 0, 18)
        }
        card.addView(TextView(this).apply {
            text = "${model.displayName}\n${model.quantization} • ${formatBytes(model.sizeBytes)}\n${model.metadata.architecture ?: "معماری نامشخص"}"
            textSize = 16f
        })
        val buttons = LinearLayout(this)
        val activate = Button(this).apply {
            text = if (model.state.name == "ACTIVE") "فعال" else "فعال‌سازی"
            isEnabled = model.state.name != "ACTIVE"
            setOnClickListener {
                isEnabled = false
                scope.launch(Dispatchers.Default) {
                    val result = manager.activate(model.id)
                    withContext(Dispatchers.Main) {
                        lastStatus = when (result) {
                            is ModelResult.Success -> "مدل فعال شد: ${result.value.displayName}"
                            is ModelResult.Failure -> "فعال‌سازی ناموفق: ${result.error.message}"
                        }
                        showModels()
                    }
                }
            }
        }
        buttons.addView(activate, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        buttons.addView(Button(this).apply {
            text = "حذف"
            setOnClickListener {
                scope.launch(Dispatchers.Default) {
                    val result = manager.delete(model.id)
                    withContext(Dispatchers.Main) {
                        lastStatus = when (result) {
                            is ModelResult.Success -> "مدل حذف شد"
                            is ModelResult.Failure -> "حذف ناموفق: ${result.error.message}"
                        }
                        showModels()
                    }
                }
            }
        }, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        card.addView(buttons)
        parent.addView(card)
    }

    private fun showChat() {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(24, 32, 24, 24)
        }
        root.addView(TextView(this).apply { text = "گفت‌وگوی محلی"; textSize = 24f })
        val modelLabel = TextView(this).apply { textSize = 14f; setPadding(0, 8, 0, 12) }
        root.addView(modelLabel)

        val scroll = ScrollView(this)
        val messages = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(0, 8, 0, 12) }
        history.forEach { addChatBubble(messages, it) }
        scroll.addView(messages)
        root.addView(scroll, LinearLayout.LayoutParams(-1, 0, 1f))

        val input = EditText(this).apply {
            hint = "پیام خود را بنویسید…"
            minLines = 2
            maxLines = 5
        }
        root.addView(input)
        val actions = LinearLayout(this)
        lateinit var stopButton: Button
        val send = Button(this).apply {
            text = "ارسال"
            setOnClickListener {
                val text = input.text.toString().trim()
                if (text.isNotEmpty() && !generationRunning) sendMessage(text, input, messages, scroll, this, stopButton)
            }
        }
        actions.addView(send, LinearLayout.LayoutParams(0, -2, 1f))
        stopButton = Button(this).apply {
            text = "توقف"
            isEnabled = false
            setOnClickListener {
                scope.launch { container.modelManager?.stopGeneration() }
                lastStatus = "درخواست توقف ارسال شد؛ runtime فعلی interrupt بومی ندارد."
                modelLabel.text = lastStatus
            }
        }
        actions.addView(stopButton, LinearLayout.LayoutParams(0, -2, 1f))
        root.addView(actions)
        root.addView(actionButton("مدل‌ها") { showModels() })
        root.addView(actionButton("بازگشت") { showHome() })
        setContentView(root)
        refreshChatModelLabel(modelLabel)
    }

    private fun refreshChatModelLabel(view: TextView) {
        val manager = container.modelManager ?: return
        scope.launch {
            when (val result = manager.activeModel()) {
                is ModelResult.Success -> view.text = result.value?.let { "مدل فعال: ${it.displayName}" } ?: "مدل فعال نیست؛ ابتدا یک GGUF را فعال کنید."
                is ModelResult.Failure -> view.text = "خطا: ${result.error.message}"
            }
        }
    }

    private fun sendMessage(
        text: String,
        input: EditText,
        messages: LinearLayout,
        scroll: ScrollView,
        sendButton: Button,
        stopButton: Button,
    ) {
        val manager = container.modelManager ?: return
        val user = ChatLine(ChatMessage.Role.USER, text, System.currentTimeMillis())
        history += user
        saveHistory()
        addChatBubble(messages, user)
        input.text.clear()
        generationRunning = true
        sendButton.isEnabled = false
        stopButton.isEnabled = true
        scope.launch(Dispatchers.Default) {
            val requestMessages = history.takeLast(MAX_CONTEXT_MESSAGES).map { ChatMessage(it.role, it.content) }
            val result = try {
                manager.generate(requestMessages, InferenceSettings(maxNewTokens = 512))
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (t: Throwable) {
                ModelResult.Failure(com.woogit.aicore.domain.ModelError.Inference("Local generation failed", t))
            }
            withContext(Dispatchers.Main) {
                generationRunning = false
                sendButton.isEnabled = true
                stopButton.isEnabled = false
                when (result) {
                    is ModelResult.Success -> {
                        val assistant = ChatLine(ChatMessage.Role.ASSISTANT, result.value.text, System.currentTimeMillis())
                        history += assistant
                        saveHistory()
                        addChatBubble(messages, assistant)
                        lastStatus = "پاسخ تولید شد • ${result.value.generationTimeMs ?: 0} ms"
                    }
                    is ModelResult.Failure -> {
                        lastStatus = result.error.message
                        addChatBubble(messages, ChatLine(ChatMessage.Role.SYSTEM, "خطا: ${result.error.message}", System.currentTimeMillis()))
                    }
                }
                scroll.post { scroll.fullScroll(View.FOCUS_DOWN) }
            }
        }
    }

    private fun showDiagnostics() {
        val root = baseRoot("Diagnostics")
        root.addView(info("وضعیت: $lastStatus"))
        val runtime = container.modelRuntime.runtimeInfo()
        root.addView(info("Runtime: ${runtime.name} ${runtime.version}\nBackend: ${runtime.backend ?: "نامشخص"}\nStreaming: پاسخ کامل؛ AAR فعلی token streaming واقعی ندارد."))
        val manager = container.modelManager
        if (manager != null) {
            scope.launch {
                when (val models = manager.models()) {
                    is ModelResult.Success -> {
                        root.addView(info("تعداد مدل‌ها: ${models.value.size}"))
                        models.value.forEach { root.addView(info("${it.displayName}: ${it.state}")) }
                    }
                    is ModelResult.Failure -> root.addView(info("خطای registry: ${models.error.message}"))
                }
            }
        }
        root.addView(actionButton("مدل‌ها") { showModels() })
        root.addView(actionButton("بازگشت") { showHome() })
        setContentView(root)
    }

    private fun addChatBubble(parent: LinearLayout, line: ChatLine) {
        val label = when (line.role) {
            ChatMessage.Role.USER -> "شما"
            ChatMessage.Role.ASSISTANT -> "مدل"
            ChatMessage.Role.SYSTEM -> "سیستم"
            ChatMessage.Role.TOOL -> "ابزار"
        }
        parent.addView(TextView(this).apply {
            text = "$label\n${line.content}"
            textSize = 16f
            setPadding(12, 12, 12, 12)
        })
    }

    private fun baseRoot(title: String): LinearLayout = LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(32, 40, 32, 28)
        addView(TextView(this@MainActivity).apply { text = title; textSize = 26f })
    }

    private fun info(text: String) = TextView(this).apply {
        this.text = text
        textSize = 15f
        setPadding(0, 12, 0, 16)
    }

    private fun actionButton(label: String, action: () -> Unit) = Button(this).apply {
        text = label
        setOnClickListener { action() }
    }

    private fun showMessage(title: String, message: String) {
        val root = baseRoot(title)
        root.gravity = Gravity.CENTER
        root.addView(info(message))
        root.addView(actionButton("بازگشت به تنظیمات") { showSettings() })
        setContentView(root)
    }

    private fun formatBytes(bytes: Long): String {
        if (bytes < 1024) return "$bytes B"
        val units = arrayOf("KB", "MB", "GB", "TB")
        var value = bytes.toDouble()
        var index = -1
        while (value >= 1024 && index < units.lastIndex) { value /= 1024.0; index++ }
        return "%.2f %s".format(value, units[index])
    }

    private data class ChatLine(val role: ChatMessage.Role, val content: String, val timestamp: Long)

    private fun historyFile() = File(filesDir, "chat-history.txt")

    private fun loadHistory(): List<ChatLine> = runCatching {
        val file = historyFile()
        if (!file.isFile) return@runCatching emptyList()
        file.readLines().mapNotNull { line ->
            val parts = line.split('|', limit = 3)
            if (parts.size != 3) return@mapNotNull null
            val role = runCatching { ChatMessage.Role.valueOf(parts[0]) }.getOrNull() ?: return@mapNotNull null
            val timestamp = parts[1].toLongOrNull() ?: return@mapNotNull null
            val content = String(Base64.getDecoder().decode(parts[2]), Charsets.UTF_8)
            ChatLine(role, content, timestamp)
        }.takeLast(MAX_SAVED_MESSAGES)
    }.getOrDefault(emptyList())

    private fun saveHistory() {
        runCatching {
            historyFile().writeText(history.takeLast(MAX_SAVED_MESSAGES).joinToString("\n") {
                "${it.role}|${it.timestamp}|${Base64.getEncoder().encodeToString(it.content.toByteArray(Charsets.UTF_8))}"
            })
        }
    }

    companion object {
        private const val REQUEST_MODEL = 1001
        private const val MAX_SAVED_MESSAGES = 100
        private const val MAX_CONTEXT_MESSAGES = 10
    }
}
