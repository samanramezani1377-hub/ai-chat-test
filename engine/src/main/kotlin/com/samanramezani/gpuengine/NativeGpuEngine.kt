package com.samanramezani.gpuengine

/** JNI boundary for the standalone Android GPU inference engine. */
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
