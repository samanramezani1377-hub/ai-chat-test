package com.woogit.aicore.runtime.android

import android.app.ActivityManager
import android.content.Context
import android.os.Debug
import android.util.Log
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import java.io.File

internal object NativeLlamaCpp {
    private const val TAG = "AIChatRuntime"
    private const val PREFLIGHT_FILE = "ai-chat-model-preflight.txt"
    private var nativeInitialized = false
    @Volatile private var gpuEngineActive = false
    private var loadedContextLength = 0

    const val GPU_LAYERS_MAX = 99
    const val GPU_LAYERS_ALL = -1

    init {
        Log.i(TAG, "ACTIVATION_NATIVE_LIBRARY_LOAD_STARTED")
        System.loadLibrary("ai_chat_runtime")
        Log.i(TAG, "ACTIVATION_NATIVE_LIBRARY_LOAD_RETURNED")
    }

    /** Initialize llama.cpp for the CPU fallback only. GPU inference is owned by AIChatGpuEngine. */
    private fun ensureNativeInitialized() {
        synchronized(this) {
            if (!nativeInitialized) {
                Log.i(TAG, "ACTIVATION_NATIVE_INIT_REQUESTED backend=CPU previously_initialized=false")
                nativeInit(false)
                nativeInstallFatalHandlers()
                nativeInitialized = true
                Log.i(TAG, "ACTIVATION_NATIVE_INIT_RETURNED backend=CPU")
            }
        }
    }

    /** Shared-memory mobile GPUs use a conservative quarter of currently safe system memory. */
    private fun conservativeGpuMemoryBudgetBytes(): Long {
        val appContext = try {
            Class.forName("android.app.ActivityThread")
                .getMethod("currentApplication")
                .invoke(null) as? Context
        } catch (_: Throwable) { null } ?: return 0L
        val manager = appContext.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager ?: return 0L
        val memoryInfo = ActivityManager.MemoryInfo()
        manager.getMemoryInfo(memoryInfo)
        return (memoryInfo.availMem - memoryInfo.threshold).coerceAtLeast(0L) / 4L
    }

    /** One serialized native activation transaction. Native side owns model/context lifetime. */
    @Synchronized
    fun load(path: String, contextLength: Int, gpuLayers: Int): Int {
        require(gpuLayers == GPU_LAYERS_ALL || gpuLayers in 0..GPU_LAYERS_MAX) { "GPU layer preference must be ALL (-1) or between 0 and $GPU_LAYERS_MAX" }
        require(path.isNotBlank()) { "Model path must not be blank" }
        val file = File(path)
        require(file.isFile && file.canRead()) { "Model file is not readable: $path" }

        persistModelLoadPreflight(path, contextLength, gpuLayers)
        if (gpuEngineActive) {
            nativeGpuEngineUnload()
            gpuEngineActive = false
        }

        val fullGpuRequest = gpuLayers == GPU_LAYERS_ALL || gpuLayers == GPU_LAYERS_MAX
        if (fullGpuRequest) {
            val budgetBytes = conservativeGpuMemoryBudgetBytes()
            if (budgetBytes > 0L) {
                val gpuResult = try {
                    nativeGpuEngineLoadModel(path, contextLength, budgetBytes)
                } catch (t: Throwable) {
                    Log.e(TAG, "GPU_ENGINE_LOAD_EXCEPTION type=${t::class.java.name} message=${t.message}", t)
                    -1
                }
                if (gpuResult == 0) {
                    gpuEngineActive = true
                    loadedContextLength = contextLength
                    Log.i(TAG, "ACTIVATION_LOAD_END result=0 runtime=AIChatGpuEngine gpu_backend=selected memory_budget_bytes=$budgetBytes")
                    return 0
                }
                val gpuDiagnostics = nativeGpuEngineDiagnostics()
                Log.w(TAG, "GPU_ENGINE_LOAD_FALLBACK code=$gpuResult diagnostics=$gpuDiagnostics")
                nativeGpuEngineUnload()
            } else {
                Log.w(TAG, "GPU_ENGINE_LOAD_FALLBACK reason=NO_SAFE_SHARED_MEMORY_BUDGET")
            }
        } else if (gpuLayers != 0) {
            Log.w(TAG, "GPU_ENGINE_LOAD_FALLBACK reason=PARTIAL_GPU_LAYER_MODE_NOT_SUPPORTED requested_layers=$gpuLayers")
        }

        configurePersistentPrefixCache()
        ensureNativeInitialized()
        Log.i(TAG, "ACTIVATION_LOAD_BEGIN ctx_len=$contextLength gpu_layers=0 runtime=llama.cpp-CPU file=${file.name}")
        return try {
            val result = nativeLoad(path, contextLength, 0)
            loadedContextLength = if (result == 0) contextLength else 0
            Log.i(TAG, "ACTIVATION_LOAD_END result=$result runtime=llama.cpp-CPU")
            result
        } catch (t: Throwable) {
            Log.e(TAG, "ACTIVATION_LOAD_EXCEPTION type=${t::class.java.name} message=${t.message}", t)
            throw t
        }
    }

    @Synchronized
    fun unload() {
        Log.i(TAG, "ACTIVATION_UNLOAD_BEGIN")
        if (gpuEngineActive) {
            nativeGpuEngineUnload()
            gpuEngineActive = false
        } else if (nativeInitialized) {
            nativeUnload()
        }
        loadedContextLength = 0
        Log.i(TAG, "ACTIVATION_UNLOAD_END")
    }

    /**
     * Keep native sequence snapshots in the app-private, non-backed-up directory.
     * Native code validates model identity, llama.cpp revision, context, and exact
     * token prefix before restoring any state.
     */
    private fun configurePersistentPrefixCache() {
        val appContext = try {
            Class.forName("android.app.ActivityThread")
                .getMethod("currentApplication")
                .invoke(null) as? Context
        } catch (_: Throwable) { null }
        val directory = appContext?.noBackupFilesDir?.let { File(it, "kv-prefix-cache") }
        if (directory == null || (!directory.isDirectory && !directory.mkdirs())) {
            Log.w(TAG, "NATIVE_KV_CACHE_DIRECTORY unavailable; disk persistence disabled")
            return
        }
        val configured = runCatching { nativeSetCacheDirectory(directory.absolutePath) }.getOrDefault(false)
        Log.i(TAG, "NATIVE_KV_CACHE_DIRECTORY configured=$configured")
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
                appendLine("backend_mode=WEBGPU_OPENGL_ES_COMPATIBILITY_CPU_FALLBACK_ALLOWED")
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

    fun countTokens(prompt: String): Int = if (gpuEngineActive) nativeGpuEngineCountTokens(prompt) else nativeCountTokens(prompt)

    fun generate(prompt: String, maxTokens: Int, temperature: Float, topK: Int, topP: Float, minP: Float, cachePrefixTokens: Int = 0, continuationPrefixTokens: Int = 0): Flow<String> = callbackFlow {
        val listener = object : TokenListener {
            override fun onToken(token: String) {
                // JNI callbacks are synchronous and cannot suspend. The default callbackFlow
                // buffer is bounded; ignoring trySend's failure silently drops generated text
                // when native decode outruns the collector. Output is already bounded by
                // maxTokens, so an unlimited fused channel preserves every token in order.
                val result = trySend(token)
                if (result.isFailure && !result.isClosed) {
                    Log.e(TAG, "STREAM_TOKEN_ENQUEUE_FAILED: generated token was not queued")
                }
            }
        }
        val useGpuEngine = gpuEngineActive
        // The GPU session resets its own cancellation state at generation start.
        // llama.cpp retains its existing reset/stop protocol on the CPU path.
        if (useGpuEngine) nativeGpuEngineResetStop() else nativeResetStop()
        val worker = launch {
            val result = if (useGpuEngine) {
                nativeGpuEngineGenerate(prompt, maxTokens, temperature, topK, topP, minP, listener)
            } else {
                nativeGenerate(prompt, maxTokens, temperature, topK, topP, minP, cachePrefixTokens, continuationPrefixTokens, listener)
            }
            if (result == 0 || result == 9) close()
            else close(IllegalStateException("AI inference generation failed: code=$result"))
        }
        awaitClose {
            if (!worker.isCompleted) {
                // Stop the active runtime and wait for the blocking JNI call to return.
                if (useGpuEngine) nativeGpuEngineStop() else nativeStop()
                runBlocking { worker.join() }
            }
        }
    }.buffer(Channel.UNLIMITED)

    @Synchronized
    fun stop() { if (gpuEngineActive) nativeGpuEngineStop() else nativeStop() }

    /** Probes the independent Dawn/Universal GLES engine without loading a model or changing llama.cpp state. */
    @Synchronized
    fun gpuEngineDiagnostics(): String = if (gpuEngineActive) nativeGpuEngineDiagnostics() else nativeGpuEngineProbe()

    fun runtimeInfo(): String = if (gpuEngineActive) nativeGpuEngineDiagnostics() else nativeRuntimeInfo()
    fun contextLength(): Int = if (gpuEngineActive) nativeGpuEngineContextLength() else nativeContextLength()

    private interface TokenListener { fun onToken(token: String) }
    @JvmStatic private external fun nativeSetCacheDirectory(path: String): Boolean
    @JvmStatic private external fun nativeInit(enableGpu: Boolean)
    @JvmStatic private external fun nativeInstallFatalHandlers()
    @JvmStatic private external fun nativeResetStop()
    @JvmStatic private external fun nativeLoad(path: String, contextLength: Int, gpuLayers: Int): Int
    @JvmStatic private external fun nativeCountTokens(prompt: String): Int
    @JvmStatic private external fun nativeGenerate(prompt: String, maxTokens: Int, temperature: Float, topK: Int, topP: Float, minP: Float, cachePrefixTokens: Int, continuationPrefixTokens: Int, listener: TokenListener): Int
    @JvmStatic private external fun nativeStop()
    @JvmStatic private external fun nativeUnload()
    @JvmStatic private external fun nativeGpuEngineProbe(): String
    @JvmStatic private external fun nativeGpuEngineLoadModel(path: String, contextLength: Int, memoryBudgetBytes: Long): Int
    @JvmStatic private external fun nativeGpuEngineGenerate(prompt: String, maxTokens: Int, temperature: Float, topK: Int, topP: Float, minP: Float, listener: TokenListener): Int
    @JvmStatic private external fun nativeGpuEngineCountTokens(text: String): Int
    @JvmStatic private external fun nativeGpuEngineContextLength(): Int
    @JvmStatic private external fun nativeGpuEngineStop()
    @JvmStatic private external fun nativeGpuEngineResetStop()
    @JvmStatic private external fun nativeGpuEngineUnload()
    @JvmStatic private external fun nativeGpuEngineDiagnostics(): String
    @JvmStatic private external fun nativeRuntimeInfo(): String
    @JvmStatic private external fun nativeContextLength(): Int
}
