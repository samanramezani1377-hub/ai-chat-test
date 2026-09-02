package com.woogit.aicore.runtime.android

import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.launch

internal object NativeLlamaCpp {
    private const val GPU_LAYERS = 99

    init {
        System.loadLibrary("ai_chat_runtime")
        nativeInit()
    }

    fun load(path: String, contextLength: Int): Int = nativeLoad(path, contextLength, GPU_LAYERS)

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
