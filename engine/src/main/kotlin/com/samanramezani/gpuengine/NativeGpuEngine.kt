package com.samanramezani.gpuengine

/** Receives UTF-8-safe text chunks from the native inference thread. */
fun interface GpuEngineTokenListener {
    fun onToken(text: String)
}

/** JNI boundary for the standalone Android GPU inference engine. */
internal object NativeGpuEngine {
    init {
        System.loadLibrary("gpu_engine_android")
    }

    @JvmStatic external fun nativeGpuEngineProbe(): String
    @JvmStatic external fun nativeGpuEngineLoadModel(path: String, contextLength: Int, memoryBudgetBytes: Long): Int
    @JvmStatic external fun nativeGpuEngineUpdateMemoryBudget(allocationCapBytes: Long): Int
    @JvmStatic external fun nativeGpuEngineGenerate(
        prompt: String,
        maxTokens: Int,
        temperature: Float,
        topK: Int,
        topP: Float,
        minP: Float,
        listener: GpuEngineTokenListener?
    ): Int
    @JvmStatic external fun nativeGpuEngineStop()
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
    fun updateMemoryBudget(allocationCapBytes: Long): Int =
        NativeGpuEngine.nativeGpuEngineUpdateMemoryBudget(allocationCapBytes)

    @JvmStatic
    fun generate(
        prompt: String,
        maxTokens: Int = 64,
        temperature: Float = 0.7f,
        topK: Int = 40,
        topP: Float = 0.95f,
        minP: Float = 0.05f,
        onToken: (String) -> Unit
    ): Int = NativeGpuEngine.nativeGpuEngineGenerate(
        prompt, maxTokens, temperature, topK, topP, minP,
        GpuEngineTokenListener(onToken)
    )

    @JvmStatic
    fun stop() = NativeGpuEngine.nativeGpuEngineStop()

    @JvmStatic
    fun diagnostics(): String = NativeGpuEngine.nativeGpuEngineDiagnostics()

    @JvmStatic
    fun unload() = NativeGpuEngine.nativeGpuEngineUnload()
}
