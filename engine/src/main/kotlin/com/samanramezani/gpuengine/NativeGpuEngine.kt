package com.samanramezani.gpuengine

/**
 * JNI compatibility name retained for the native entry points. This module contains
 * only the independent GPU engine; there is no llama.cpp runtime in this project.
 */
internal object NativeGpuEngine {
    init {
        System.loadLibrary("gpu_engine_android")
    }

    @JvmStatic external fun nativeGpuEngineProbe(): String
    @JvmStatic external fun nativeGpuEngineLoadModel(path: String, contextLength: Int, memoryBudgetBytes: Long): Int
    @JvmStatic external fun nativeGpuEngineUnload()
    @JvmStatic external fun nativeGpuEngineDiagnostics(): String
}

/** Small public API consumed by the standalone test application. */
object GpuEngineTestApi {
    @JvmStatic
    fun probe(): String = NativeGpuEngine.nativeGpuEngineProbe()

    @JvmStatic
    fun loadModel(path: String, contextLength: Int, memoryBudgetBytes: Long): Int =
        NativeGpuEngine.nativeGpuEngineLoadModel(path, contextLength, memoryBudgetBytes)

    @JvmStatic
    fun diagnostics(): String = NativeGpuEngine.nativeGpuEngineDiagnostics()

    @JvmStatic
    fun unload() = NativeGpuEngine.nativeGpuEngineUnload()
}
