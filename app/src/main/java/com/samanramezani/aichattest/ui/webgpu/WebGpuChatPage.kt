package com.samanramezani.aichattest.ui.webgpu

import android.annotation.SuppressLint
import android.content.ContentValues
import android.content.Context
import android.net.Uri
import android.os.Build
import android.provider.MediaStore
import android.util.Log
import android.webkit.ConsoleMessage
import android.webkit.JavascriptInterface
import android.webkit.WebChromeClient
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
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
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

private const val WEBGPU_MODEL_ID = "onnx-community/Qwen3.5-2B-ONNX-OPT"
private const val WEBGPU_DOWNLOAD_PATH = "Download/AI Chat Test/Qwen3.5-2B-ONNX-OPT/"

/** Stores model files in the same public Downloads/AI Chat Test root used by Laya's bundle. */
@SuppressLint("SetJavaScriptEnabled")
@Composable
fun WebGpuChatPage() {
    val context = LocalContext.current
    val appContext = context.applicationContext
    val assetLoader = remember(appContext) {
        WebViewAssetLoader.Builder()
            .addPathHandler("/assets/", WebViewAssetLoader.AssetsPathHandler(appContext))
            .addPathHandler("/models/", DownloadsModelPathHandler(appContext))
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
                override fun shouldInterceptRequest(view: WebView, request: WebResourceRequest): WebResourceResponse? =
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
            addJavascriptInterface(WebGpuModelStoreBridge(appContext, this), "DinaModelStore")
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

/** Serves model files directly from public Downloads; no second private copy is needed. */
private class DownloadsModelPathHandler(private val context: Context) : WebViewAssetLoader.PathHandler {
    override fun handle(path: String): WebResourceResponse? {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) return null
        val normalized = path.replace('\\', '/')
        if (normalized.startsWith("/") || normalized.split('/').any { it == ".." || it.isBlank() }) return null
        val modelPrefix = "$WEBGPU_MODEL_ID/"
        if (!normalized.startsWith(modelPrefix)) return null
        val relative = normalized.removePrefix(modelPrefix)
        val parent = relative.substringBeforeLast('/', "")
        val name = relative.substringAfterLast('/')
        val relativePath = WEBGPU_DOWNLOAD_PATH + if (parent.isEmpty()) "" else "$parent/"
        val collection = MediaStore.Downloads.EXTERNAL_CONTENT_URI
        val projection = arrayOf(MediaStore.Downloads._ID, MediaStore.MediaColumns.SIZE)
        val selection = "${MediaStore.MediaColumns.DISPLAY_NAME}=? AND ${MediaStore.MediaColumns.RELATIVE_PATH}=?"
        val uri = context.contentResolver.query(collection, projection, selection, arrayOf(name, relativePath), null)?.use { cursor ->
            if (cursor.moveToFirst() && cursor.getLong(1) > 0L) Uri.withAppendedPath(collection, cursor.getLong(0).toString()) else null
        } ?: return null
        val mime = when {
            name.endsWith(".json") || name.endsWith(".jinja") -> "application/octet-stream"
            else -> "application/octet-stream"
        }
        val stream = context.contentResolver.openInputStream(uri) ?: return null
        return WebResourceResponse(mime, null, stream)
    }
}

/** Downloads each required artifact into Downloads/AI Chat Test and reuses it on later launches. */
private class WebGpuModelStoreBridge(context: Context, private val webView: WebView) {
    companion object {
        private const val MODEL_URL = "https://huggingface.co/$WEBGPU_MODEL_ID/resolve/main/"
        private val REQUIRED_FILES = listOf(
            "chat_template.jinja", "config.json", "generation_config.json",
            "preprocessor_config.json", "processor_config.json", "tokenizer.json",
            "tokenizer_config.json", "onnx/embed_tokens_q4.onnx", "onnx/embed_tokens_q4.onnx_data",
            "onnx/decoder_model_merged_q4.onnx", "onnx/decoder_model_merged_q4.onnx_data",
        )
    }

    private val resolver = context.contentResolver
    private val executor = Executors.newSingleThreadExecutor()
    private val downloading = AtomicBoolean(false)

    @JavascriptInterface
    fun isModelReady(): Boolean = Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q &&
        REQUIRED_FILES.all { findFile(it)?.second?.let { size -> size > 0L } == true }

    @JavascriptInterface
    fun startModelDownload() {
        if (isModelReady()) {
            notifyPage("done", "مدل از قبل در پوشهٔ Downloads/AI Chat Test ذخیره شده است؛ دانلود مجدد انجام نشد.")
            return
        }
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
            notifyPage("error", "ذخیرهٔ مستقیم و پایدار در Downloads/AI Chat Test به Android 10 یا بالاتر نیاز دارد.")
            return
        }
        if (!downloading.compareAndSet(false, true)) {
            notifyPage("progress", "دانلود مدل از قبل در حال انجام است.")
            return
        }
        executor.execute {
            try {
                REQUIRED_FILES.forEachIndexed { index, relative ->
                    val existing = findFile(relative)
                    if (existing != null && existing.second > 0L) {
                        notifyPage("progress", "فایل ${index + 1} از ${REQUIRED_FILES.size} از قبل در Downloads موجود است: ${relative.substringAfterLast('/')}")
                        return@forEachIndexed
                    }
                    existing?.first?.let { runCatching { resolver.delete(it, null, null) } }
                    val parent = relative.substringBeforeLast('/', "")
                    val fileName = relative.substringAfterLast('/')
                    val folder = WEBGPU_DOWNLOAD_PATH + if (parent.isEmpty()) "" else "$parent/"
                    val values = ContentValues().apply {
                        put(MediaStore.MediaColumns.DISPLAY_NAME, fileName)
                        put(MediaStore.MediaColumns.MIME_TYPE, "application/octet-stream")
                        put(MediaStore.MediaColumns.RELATIVE_PATH, folder)
                        put(MediaStore.MediaColumns.IS_PENDING, 1)
                    }
                    val uri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
                        ?: error("ساخت فایل در Downloads/AI Chat Test ناموفق بود: $relative")
                    try {
                        val encodedPath = relative.split('/').joinToString("/") {
                            java.net.URLEncoder.encode(it, "UTF-8").replace("+", "%20")
                        }
                        val connection = (URL(MODEL_URL + encodedPath + "?download=true").openConnection() as HttpURLConnection).apply {
                            connectTimeout = 30_000
                            readTimeout = 120_000
                            instanceFollowRedirects = true
                            useCaches = false
                            setRequestProperty("User-Agent", "AI-Chat-Test-Android/1.0")
                        }
                        try {
                            val code = connection.responseCode
                            check(code in 200..299) { "HTTP $code هنگام دریافت $relative" }
                            val total = connection.contentLengthLong
                            var received = 0L
                            var lastReported = -5
                            val output = resolver.openOutputStream(uri, "w") ?: error("نوشتن فایل در Downloads ناموفق بود: $relative")
                            connection.inputStream.use { input ->
                                output.buffered().use { out ->
                                    val buffer = ByteArray(256 * 1024)
                                    while (true) {
                                        val count = input.read(buffer)
                                        if (count < 0) break
                                        out.write(buffer, 0, count)
                                        received += count
                                        if (total > 0L) {
                                            val percent = ((received * 100L) / total).toInt().coerceIn(0, 100)
                                            if (percent >= lastReported + 5 || percent == 100) {
                                                lastReported = percent
                                                notifyPage("progress", "دانلود ${index + 1}/${REQUIRED_FILES.size}: ${relative.substringAfterLast('/')} — ${percent}٪")
                                            }
                                        } else if (received % (16L * 1024L * 1024L) < buffer.size) {
                                            notifyPage("progress", "دانلود ${relative.substringAfterLast('/')} — ${received / (1024 * 1024)} مگابایت")
                                        }
                                    }
                                }
                            }
                            val published = ContentValues().apply { put(MediaStore.MediaColumns.IS_PENDING, 0) }
                            check(resolver.update(uri, published, null, null) == 1) { "ثبت فایل دانلودشده در Downloads ناموفق بود: $relative" }
                            val size = findFile(relative)?.second ?: 0L
                            check(size > 0L) { "فایل دانلودشده خالی است: $relative" }
                        } finally {
                            connection.disconnect()
                        }
                    } catch (error: Throwable) {
                        runCatching { resolver.delete(uri, null, null) }
                        throw error
                    }
                }
                check(isModelReady()) { "اعتبارسنجی فایل‌های مدل در Downloads/AI Chat Test ناموفق بود." }
                notifyPage("done", "دانلود کامل شد؛ فایل‌های مدل در Downloads/AI Chat Test/Qwen3.5-2B-ONNX-OPT ذخیره شدند.")
            } catch (error: Throwable) {
                Log.e("DinaWebGPU", "Model download failed", error)
                notifyPage("error", error.message ?: "دانلود مدل ناموفق بود.")
            } finally {
                downloading.set(false)
            }
        }
    }

    private fun findFile(relative: String): Pair<Uri, Long>? {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) return null
        val parent = relative.substringBeforeLast('/', "")
        val name = relative.substringAfterLast('/')
        val folder = WEBGPU_DOWNLOAD_PATH + if (parent.isEmpty()) "" else "$parent/"
        val projection = arrayOf(MediaStore.Downloads._ID, MediaStore.MediaColumns.SIZE)
        val selection = "${MediaStore.MediaColumns.DISPLAY_NAME}=? AND ${MediaStore.MediaColumns.RELATIVE_PATH}=?"
        return resolver.query(MediaStore.Downloads.EXTERNAL_CONTENT_URI, projection, selection, arrayOf(name, folder), null)?.use { cursor ->
            if (cursor.moveToFirst()) Uri.withAppendedPath(MediaStore.Downloads.EXTERNAL_CONTENT_URI, cursor.getLong(0).toString()) to cursor.getLong(1)
            else null
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
