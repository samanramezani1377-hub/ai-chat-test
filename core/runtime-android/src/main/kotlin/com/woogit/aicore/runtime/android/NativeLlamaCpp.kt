package com.woogit.aicore.runtime.android

import android.app.ActivityManager
import android.content.Context
import android.os.Debug
import android.util.Log
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import java.io.File

internal object NativeLlamaCpp {
    private const val TAG = "AIChatRuntime"
    private const val PREFLIGHT_FILE = "ai-chat-model-preflight.txt"
    private var nativeInitialized = false

    const val GPU_LAYERS_MAX = 99

    init {
        Log.i(TAG, "ACTIVATION_NATIVE_LIBRARY_LOAD_STARTED")
        System.loadLibrary("ai_chat_runtime")
        Log.i(TAG, "ACTIVATION_NATIVE_LIBRARY_LOAD_RETURNED")
    }

    /** Vulkan is the only native backend; every model load requests full GPU offload. */
    private fun ensureNativeInitialized() {
        synchronized(this) {
            Log.i(TAG, "ACTIVATION_NATIVE_INIT_REQUESTED backend=Vulkan previously_initialized=$nativeInitialized")
            nativeInit(true)
            nativeInstallFatalHandlers()
            nativeInitialized = true
            Log.i(TAG, "ACTIVATION_NATIVE_INIT_RETURNED backend=Vulkan")
        }
    }

    /** One serialized native activation transaction. Native side owns model/context lifetime. */
    @Synchronized
    fun load(path: String, contextLength: Int, gpuLayers: Int, draftPath: String? = null): Int {
        require(gpuLayers > 0) { "Vulkan GPU-only runtime requires at least one GPU layer" }
        require(path.isNotBlank()) { "Model path must not be blank" }
        val file = File(path)
        require(file.isFile && file.canRead()) { "Model file is not readable: $path" }

        persistModelLoadPreflight(path, contextLength, gpuLayers)
        ensureNativeInitialized()
        Log.i(TAG, "ACTIVATION_LOAD_BEGIN ctx_len=$contextLength gpu_layers=$gpuLayers file=${file.name}")
        return try {
            val result = nativeLoad(path, contextLength, gpuLayers, draftPath)
            Log.i(TAG, "ACTIVATION_LOAD_END result=$result gpu_layers=$gpuLayers")
            result
        } catch (t: Throwable) {
            Log.e(TAG, "ACTIVATION_LOAD_EXCEPTION type=${t::class.java.name} message=${t.message}", t)
            throw t
        }
    }

    @Synchronized
    fun unload() {
        Log.i(TAG, "ACTIVATION_UNLOAD_BEGIN")
        nativeUnload()
        Log.i(TAG, "ACTIVATION_UNLOAD_END")
    }

    private fun persistModelLoadPreflight(path: String, contextLength: Int, gpuLayers: Int) {
        runCatching {
            val file = File(path)
            val memoryInfo = ActivityManager.MemoryInfo()
            val activityManager = try {
                Class.forName("android.app.ActivityThread")
                    .getMethod("currentApplication")
                    .invoke(null) as? Context
            } catch (_: Throwable) { null }
            activityManager?.getSystemService(ActivityManager::class.java)?.getMemoryInfo(memoryInfo)
            val processMemory = Debug.MemoryInfo()
            Debug.getMemoryInfo(processMemory)
            val text = buildString {
                appendLine("MODEL_LOAD_PREFLIGHT")
                appendLine("path=${file.absolutePath}")
                appendLine("file_exists=${file.isFile}")
                appendLine("file_size_bytes=${if (file.isFile) file.length() else -1}")
                appendLine("file_size_mib=${if (file.isFile) file.length() / 1048576.0 else -1.0}")
                appendLine("requested_context=$contextLength")
                appendLine("gpu_layers=$gpuLayers")
                appendLine("backend_mode=Vulkan_GPU_ONLY")
                appendLine("device_mem_total_bytes=${memoryInfo.totalMem}")
                appendLine("device_mem_available_bytes=${memoryInfo.availMem}")
                appendLine("device_mem_available_mib=${memoryInfo.availMem / 1048576.0}")
                appendLine("device_low_memory=${memoryInfo.lowMemory}")
                appendLine("device_low_memory_threshold_bytes=${memoryInfo.threshold}")
                appendLine("process_pss_kib=${processMemory.totalPss}")
                appendLine("process_private_dirty_kib=${processMemory.totalPrivateDirty}")
            }
            File(System.getProperty("java.io.tmpdir") ?: ".", PREFLIGHT_FILE).apply {
                parentFile?.mkdirs()
                writeText(text)
            }
        }.onFailure {
            Log.w(TAG, "MODEL_LOAD_PREFLIGHT_FAILED type=${it::class.java.name} message=${it.message}")
        }
    }

    fun countTokens(prompt: String): Int = nativeCountTokens(prompt)

    fun generate(prompt: String, maxTokens: Int, temperature: Float, topK: Int, topP: Float, minP: Float): Flow<String> = callbackFlow {
        val listener = object : TokenListener {
            override fun onToken(token: String) { trySend(token) }
        }
        val worker = launch {
            val result = nativeGenerate(prompt, maxTokens, temperature, topK, topP, minP, listener)
            if (result == 0 || result == 9) close()
            else close(IllegalStateException("llama.cpp generation failed: code=$result"))
        }
        awaitClose {
            if (!worker.isCompleted) {
                // callbackFlow cancellation does not interrupt a blocking JNI call.
                // Stop the native decode and wait for it to return before releasing
                // the Flow, otherwise the next generation can race the previous one.
                nativeStop()
                runBlocking { worker.join() }
            }
        }
    }

    @Synchronized
    fun stop() = nativeStop()
    fun runtimeInfo(): String = nativeRuntimeInfo()
    fun contextLength(): Int = nativeContextLength()

    private interface TokenListener { fun onToken(token: String) }
    @JvmStatic private external fun nativeInit(enableGpu: Boolean)
    @JvmStatic private external fun nativeInstallFatalHandlers()
    @JvmStatic private external fun nativeLoad(path: String, contextLength: Int, gpuLayers: Int, draftPath: String?): Int
    @JvmStatic private external fun nativeCountTokens(prompt: String): Int
    @JvmStatic private external fun nativeGenerate(prompt: String, maxTokens: Int, temperature: Float, topK: Int, topP: Float, minP: Float, listener: TokenListener): Int
    @JvmStatic private external fun nativeStop()
    @JvmStatic private external fun nativeUnload()
    @JvmStatic private external fun nativeRuntimeInfo(): String
    @JvmStatic private external fun nativeContextLength(): Int
}
