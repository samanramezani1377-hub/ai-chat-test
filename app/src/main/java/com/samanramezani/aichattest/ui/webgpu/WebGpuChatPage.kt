package com.samanramezani.aichattest.ui.webgpu

import android.annotation.SuppressLint
import android.content.Context
import android.util.Log
import android.webkit.ConsoleMessage
import android.webkit.JavascriptInterface
import android.webkit.WebChromeClient
import android.webkit.WebResourceRequest
import android.webkit.WebView
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import androidx.webkit.WebViewAssetLoader
import androidx.webkit.WebViewClientCompat
import org.json.JSONObject
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/** WebGPU Qwen weights persist beside Laya's private model directory, not in WebView cache. */
@SuppressLint("SetJavaScriptEnabled")
@Composable
fun WebGpuChatPage() {
    val context = LocalContext.current
    val appContext = context.applicationContext
    val modelDirectory = remember(appContext) { File(appContext.filesDir, "voice-models/laya-int4/webgpu") }
    val assetLoader = remember(appContext, modelDirectory) {
        WebViewAssetLoader.Builder()
            .addPathHandler("/assets/", WebViewAssetLoader.AssetsPathHandler(appContext))
            .addPathHandler("/models/", WebViewAssetLoader.InternalStoragePathHandler(appContext, modelDirectory))
            .build()
    }
    val webView = remember(context) {
        WebView(context).apply {
            setBackgroundColor(android.graphics.Color.rgb(248, 250, 249))
            settings.javaScriptEnabled = true
            settings.domStorageEnabled = true
            settings.allowFileAccess = false
            settings.allowContentAccess = false
            settings.javaScriptCanOpenWindowsAutomatically = false
            settings.mediaPlaybackRequiresUserGesture = true
            settings.mixedContentMode = android.webkit.WebSettings.MIXED_CONTENT_NEVER_ALLOW
            settings.setSupportMultipleWindows(false)
            webViewClient = object : WebViewClientCompat() {
                override fun shouldInterceptRequest(view: WebView, request: WebResourceRequest): android.webkit.WebResourceResponse? =
                    assetLoader.shouldInterceptRequest(request.url)

                override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest): Boolean =
                    request.url.scheme != "https" || request.url.host != "appassets.androidplatform.net"
            }
            webChromeClient = object : WebChromeClient() {
                override fun onConsoleMessage(consoleMessage: ConsoleMessage): Boolean {
                    val line = "webgpu:" + consoleMessage.lineNumber() + " " + consoleMessage.message()
                    if (consoleMessage.messageLevel() == ConsoleMessage.MessageLevel.ERROR) Log.e("DinaWebGPU", line)
                    else Log.d("DinaWebGPU", line)
                    return true
                }
            }
            addJavascriptInterface(WebGpuModelStoreBridge(appContext, modelDirectory, this), "DinaModelStore")
            loadUrl("https://appassets.androidplatform.net/assets/webgpu/index.html")
        }
    }
    AndroidView(factory = { webView }, modifier = Modifier.fillMaxSize())
    DisposableEffect(webView) {
        onDispose {
            webView.stopLoading()
            webView.loadUrl("about:blank")
            webView.destroy()
        }
    }
}

/** Downloads only the Q4 text-inference artifacts. Final files are atomically renamed. */
private class WebGpuModelStoreBridge(
    context: Context,
    private val modelDirectory: File,
    private val webView: WebView,
) {
    companion object {
        private const val MODEL_ID = "onnx-community/Qwen3.5-2B-ONNX-OPT"
        private const val MODEL_URL = "https://huggingface.co/$MODEL_ID/resolve/main/"
        private val REQUIRED_FILES = listOf(
            "chat_template.jinja", "config.json", "generation_config.json",
            "preprocessor_config.json", "processor_config.json", "tokenizer.json",
            "tokenizer_config.json", "onnx/embed_tokens_q4.onnx", "onnx/embed_tokens_q4.onnx_data",
            "onnx/decoder_model_merged_q4.onnx", "onnx/decoder_model_merged_q4.onnx_data",
        )
    }

    private val modelRoot = File(modelDirectory, MODEL_ID)
    private val executor = Executors.newSingleThreadExecutor()
    private val downloading = AtomicBoolean(false)

    @JavascriptInterface
    fun isModelReady(): Boolean = REQUIRED_FILES.all { relative ->
        val file = File(modelRoot, relative)
        file.isFile && file.length() > 0L
    }

    @JavascriptInterface
    fun startModelDownload() {
        if (isModelReady()) {
            notifyPage("done", "مدل از قبل در پوشهٔ محلی دینا موجود است؛ دانلود مجدد انجام نشد.")
            return
        }
        if (!downloading.compareAndSet(false, true)) {
            notifyPage("progress", "دانلود مدل از قبل در حال انجام است.")
            return
        }
        executor.execute {
            try {
                modelRoot.mkdirs()
                REQUIRED_FILES.forEachIndexed { index, relative ->
                    val target = File(modelRoot, relative)
                    if (target.isFile && target.length() > 0L) {
                        notifyPage("progress", "فایل ${index + 1} از ${REQUIRED_FILES.size} از قبل موجود است؛ رد شد: ${relative.substringAfterLast('/')}")
                        return@forEachIndexed
                    }
                    target.parentFile?.mkdirs()
                    val temporary = File(target.parentFile, target.name + ".part")
                    temporary.delete()
                    val encodedPath = relative.split('/').joinToString("/") {
                        java.net.URLEncoder.encode(it, "UTF-8").replace("+", "%20")
                    }
                    val connection = (URL(MODEL_URL + encodedPath + "?download=true").openConnection() as HttpURLConnection).apply {
                        connectTimeout = 30_000
                        readTimeout = 60_000
                        instanceFollowRedirects = true
                        requestMethod = "GET"
                    }
                    try {
                        val code = connection.responseCode
                        check(code in 200..299) { "HTTP $code هنگام دریافت $relative" }
                        val total = connection.contentLengthLong
                        connection.inputStream.use { input ->
                            temporary.outputStream().buffered().use { output ->
                                val buffer = ByteArray(256 * 1024)
                                var received = 0L
                                var lastReported = -5
                                while (true) {
                                    val count = input.read(buffer)
                                    if (count < 0) break
                                    output.write(buffer, 0, count)
                                    received += count
                                    if (total > 0L) {
                                        val percent = ((received * 100L) / total).toInt().coerceIn(0, 100)
                                        if (percent >= lastReported + 5 || percent == 100) {
                                            lastReported = percent
                                            notifyPage("progress", "دانلود فایل ${index + 1}/${REQUIRED_FILES.size}: ${relative.substringAfterLast('/')} — ${percent}٪")
                                        }
                                    } else if (received % (16L * 1024L * 1024L) < buffer.size) {
                                        notifyPage("progress", "دانلود ${relative.substringAfterLast('/')} — ${received / (1024 * 1024)} مگابایت")
                                    }
                                }
                            }
                        }
                        check(temporary.isFile && temporary.length() > 0L) { "فایل دانلودشده خالی است: $relative" }
                        check(temporary.renameTo(target)) { "ذخیرهٔ نهایی فایل ناموفق بود: $relative" }
                    } finally {
                        connection.disconnect()
                        temporary.delete()
                    }
                }
                check(isModelReady()) { "پس از دانلود، اعتبارسنجی فایل‌های مدل ناموفق بود." }
                notifyPage("done", "دانلود مدل کامل شد و فایل‌ها در پوشهٔ پایدار دینا ذخیره شدند.")
            } catch (error: Throwable) {
                Log.e("DinaWebGPU", "Model download failed", error)
                notifyPage("error", error.message ?: "دانلود مدل ناموفق بود.")
            } finally {
                downloading.set(false)
            }
        }
    }

    private fun notifyPage(state: String, message: String) {
        val stateJson = JSONObject.quote(state)
        val messageJson = JSONObject.quote(message)
        webView.post {
            runCatching { webView.evaluateJavascript("window.dinaModelProgress && window.dinaModelProgress($stateJson, $messageJson);", null) }
        }
    }
}
