package com.woogit.aicore.runtime.android

/**
 * JNI compatibility name retained for the native entry points. This module contains
 * only the independent GPU engine; there is no llama.cpp runtime in this project.
 */
internal object NativeLlamaCpp {
    init {
        System.loadLibrary("ai_chat_runtime")
    }

    @JvmStatic external fun nativeGpuEngineProbe(): String
    @JvmStatic external fun nativeGpuEngineLoadModel(path: String, contextLength: Int, memoryBudgetBytes: Long): Int
    @JvmStatic external fun nativeGpuEngineUnload()
    @JvmStatic external fun nativeGpuEngineDiagnostics(): String
}

/** Small public API consumed by the standalone test application. */
object GpuEngineTestApi {
    @JvmStatic
    fun probe(): String = NativeLlamaCpp.nativeGpuEngineProbe()

    @JvmStatic
    fun loadModel(path: String, contextLength: Int, memoryBudgetBytes: Long): Int =
        NativeLlamaCpp.nativeGpuEngineLoadModel(path, contextLength, memoryBudgetBytes)

    @JvmStatic
    fun diagnostics(): String = NativeLlamaCpp.nativeGpuEngineDiagnostics()

    @JvmStatic
    fun unload() = NativeLlamaCpp.nativeGpuEngineUnload()
}
