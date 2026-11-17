package com.samanramezani.aichattest.screen

import java.util.concurrent.Executors

/** Independent local vision runtime; it never changes the active text-model/provider selection. */
object VisionNativeRuntime {
    private val lock = Any()
    private val unloadExecutor = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "screen-vision-unload").apply { isDaemon = true }
    }
    @Volatile private var loadedModel: String? = null
    @Volatile private var loadedProjector: String? = null

    init { System.loadLibrary("ai_chat_runtime") }

    fun describe(modelPath: String, projectorPath: String, jpeg: ByteArray, question: String): String =
        synchronized(lock) {
            if (loadedModel != modelPath || loadedProjector != projectorPath) {
                nativeUnload()
                val result = nativeLoad(modelPath, projectorPath, 4096)
                check(result == 0) { "VISION_MODEL_LOAD_FAILED:" + result }
                loadedModel = modelPath
                loadedProjector = projectorPath
            }
            nativeDescribe(jpeg, question, 192)
        }

    /** Queue native teardown off the UI/service callback thread; describe() and unload() serialize on lock. */
    fun unloadAsync() { unloadExecutor.execute { runCatching { unload() } } }

    fun unload() = synchronized(lock) {
        if (loadedModel != null || loadedProjector != null) nativeUnload()
        loadedModel = null
        loadedProjector = null
    }

    @JvmStatic private external fun nativeLoad(modelPath: String, projectorPath: String, contextLength: Int): Int
    @JvmStatic private external fun nativeDescribe(jpeg: ByteArray, question: String, maxTokens: Int): String
    @JvmStatic private external fun nativeUnload()
}
