package com.samanramezani.aichattest.screen

import java.util.concurrent.Executors
import java.util.concurrent.ScheduledExecutorService
import java.util.concurrent.ScheduledFuture
import java.util.concurrent.TimeUnit

/** Independent local vision runtime; it never changes the active text-model/provider selection. */
object VisionNativeRuntime {
    private val lock = Any()
    private val lifecycleExecutor: ScheduledExecutorService =
        Executors.newSingleThreadScheduledExecutor { runnable ->
            Thread(runnable, "screen-vision-lifecycle").apply { isDaemon = true }
        }
    private var pendingIdleUnload: ScheduledFuture<*>? = null
    private val idleUnloadSeconds = 60L
    private var libraryLoaded = false
    @Volatile private var loadedModel: String? = null
    @Volatile private var loadedProjector: String? = null

    fun describe(modelPath: String, projectorPath: String, jpeg: ByteArray, question: String): String =
        synchronized(lock) {
            ensureLibraryLoadedLocked()
            cancelIdleUnloadLocked()
            if (loadedModel != modelPath || loadedProjector != projectorPath) {
                unloadNativeLocked()
                val result = nativeLoad(modelPath, projectorPath, 4096)
                if (result != 0) {
                    runCatching { nativeUnload() }
                    error("VISION_MODEL_LOAD_FAILED:" + result)
                }
                loadedModel = modelPath
                loadedProjector = projectorPath
            }
            try {
                nativeDescribe(jpeg, question, 192).also { scheduleIdleUnloadLocked() }
            } catch (failure: Throwable) {
                runCatching { nativeUnload() }
                loadedModel = null
                loadedProjector = null
                throw failure
            }
        }

    /** Queue native teardown off the UI/service callback thread; describe() and unload() serialize on lock. */
    fun unloadAsync() { lifecycleExecutor.execute { runCatching { unload() } } }

    fun unload() = synchronized(lock) {
        unloadNativeLocked()
    }

    private fun ensureLibraryLoadedLocked() {
        if (!libraryLoaded) {
            System.loadLibrary("ai_chat_runtime")
            libraryLoaded = true
        }
    }

    private fun scheduleIdleUnloadLocked() {
        cancelIdleUnloadLocked()
        pendingIdleUnload = lifecycleExecutor.schedule(
            { runCatching { unload() } },
            idleUnloadSeconds,
            TimeUnit.SECONDS,
        )
    }

    private fun cancelIdleUnloadLocked() {
        pendingIdleUnload?.cancel(false)
        pendingIdleUnload = null
    }

    private fun unloadNativeLocked() {
        cancelIdleUnloadLocked()
        try {
            if (loadedModel != null || loadedProjector != null) nativeUnload()
        } finally {
            loadedModel = null
            loadedProjector = null
        }
    }

    @JvmStatic private external fun nativeLoad(modelPath: String, projectorPath: String, contextLength: Int): Int
    @JvmStatic private external fun nativeDescribe(jpeg: ByteArray, question: String, maxTokens: Int): String
    @JvmStatic private external fun nativeUnload()
}
