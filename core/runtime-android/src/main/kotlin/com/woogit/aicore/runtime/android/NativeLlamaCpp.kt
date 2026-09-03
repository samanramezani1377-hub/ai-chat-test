package com.woogit.aicore.runtime.android

import android.util.Log
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.launch

internal object NativeLlamaCpp {
    private const val TAG = "AIChatRuntime"

    /** Temporary activation diagnostics: switch between CPU-only, 70 layers, and max-offload. */
    const val GPU_LAYERS_CPU_ONLY = 0
    const val GPU_LAYERS_70 = 70
    const val GPU_LAYERS_MAX = 99

    init {
        Log.i(TAG, "ACTIVATION_NATIVE_LIBRARY_LOAD_STARTED")
        System.loadLibrary("ai_chat_runtime")
        Log.i(TAG, "ACTIVATION_NATIVE_LIBRARY_LOAD_RETURNED")
        nativeInit()
        Log.i(TAG, "ACTIVATION_NATIVE_INIT_RETURNED")
    }

    fun load(path: String, contextLength: Int, gpuLayers: Int): Int {
        require(gpuLayers >= 0) { "gpuLayers must be >= 0" }
        Log.i(TAG, "ACTIVATION_KOTLIN_NATIVE_LOAD_STARTED ctx_len=$contextLength gpu_layers=$gpuLayers file=${path.substringAfterLast('/')}")
        return try {
            val result = nativeLoad(path, contextLength, gpuLayers)
            Log.i(TAG, "ACTIVATION_KOTLIN_NATIVE_LOAD_RETURNED result=$result gpu_layers=$gpuLayers")
            result
        } catch (t: Throwable) {
            Log.e(TAG, "ACTIVATION_KOTLIN_NATIVE_LOAD_THROWN type=${t::class.java.name} message=${t.message}", t)
            throw t
        }
    }

    fun generate(
        prompt: String,
        maxTokens: Int,
        temperature: Float,
        topK: Int,
        topP: Float,
        minP: Float,
    ): Flow<String> = callbackFlow {
        val listener = object : TokenListener {
            override fun onToken(token: String) { trySend(token) }
        }
        val worker = launch {
            val result = nativeGenerate(prompt, maxTokens, temperature, topK, topP, minP, listener)
            if (result == 0 || result == 9) close()
            else close(IllegalStateException("llama.cpp generation failed: code=$result"))
        }
        awaitClose {
            if (!worker.isCompleted) nativeStop()
            worker.cancel()
        }
    }

    fun stop() = nativeStop()
    fun unload() = nativeUnload()
    fun runtimeInfo(): String = nativeRuntimeInfo()
    fun contextLength(): Int = nativeContextLength()

    private interface TokenListener { fun onToken(token: String) }

    @JvmStatic private external fun nativeInit()
    @JvmStatic private external fun nativeLoad(path: String, contextLength: Int, gpuLayers: Int): Int
    @JvmStatic private external fun nativeGenerate(
        prompt: String,
        maxTokens: Int,
        temperature: Float,
        topK: Int,
        topP: Float,
        minP: Float,
        listener: TokenListener,
    ): Int
    @JvmStatic private external fun nativeStop()
    @JvmStatic private external fun nativeUnload()
    @JvmStatic private external fun nativeRuntimeInfo(): String
    @JvmStatic private external fun nativeContextLength(): Int
}
