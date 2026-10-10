package com.woogit.aicore.runtime.mnn

import com.woogit.aicore.domain.ChatMessage
import com.woogit.aicore.domain.GenerationRequest
import com.woogit.aicore.domain.GenerationResult
import com.woogit.aicore.domain.ModelDescriptor
import com.woogit.aicore.domain.ModelFormat
import com.woogit.aicore.domain.RuntimeInfo
import com.woogit.aicore.runtime.RuntimeAdapter
import com.woogit.aicore.runtime.RuntimeMetrics
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.system.measureTimeMillis

/** MNN runtime for exported MNN model packs. GGUF is deliberately rejected. */
class MnnAndroidRuntimeAdapter(private val cpuThreads: Int = 4) : RuntimeAdapter, RuntimeMetrics {
    private val mutex = Mutex()
    private val stopped = AtomicBoolean(false)
    @Volatile private var backend = "unloaded"
    @Volatile private var loadedConfig: String? = null
    @Volatile private var lastResult: GenerationResult? = null
    @Volatile private var lastLoadMs: Long? = null

    override suspend fun load(model: ModelDescriptor) = mutex.withLock {
        require(model.format == ModelFormat.MNN) { "MNN runtime requires an MNN model pack; GGUF is not supported directly." }
        val config = model.path.toFile()
        require(config.isFile && config.name.equals("config.json", ignoreCase = true)) {
            "MNN model path must point to config.json inside an exported MNN model directory."
        }
        require(File(config.parentFile, "llm.mnn").isFile && File(config.parentFile, "llm.mnn.weight").isFile) {
            "MNN model pack is incomplete: expected llm.mnn and llm.mnn.weight beside config.json."
        }
        if (loadedConfig == config.absolutePath && backend != "unloaded") return@withLock
        runCatching { MnnNative.nativeUnload() }
        var loadError: Throwable? = null
        val elapsed = measureTimeMillis {
            try {
                val selected = MnnNative.nativeLoad(config.absolutePath, cpuThreads)
                require(!selected.startsWith("ERROR:")) { "MNN load failed: $selected" }
                backend = selected
                loadedConfig = config.absolutePath
            } catch (t: Throwable) {
                loadError = t
                backend = "failed"
                loadedConfig = null
            }
        }
        lastLoadMs = elapsed
        loadError?.let { throw IllegalStateException("MNN could not load on OpenCL or CPU: ${it.message}", it) }
        lastResult = null
    }

    override suspend fun unload() = mutex.withLock {
        stopped.set(true)
        runCatching { MnnNative.nativeUnload() }
        backend = "unloaded"
        loadedConfig = null
        lastResult = null
        stopped.set(false)
    }

    override suspend fun generate(request: GenerationRequest, onToken: suspend (String) -> Unit): GenerationResult = mutex.withLock {
        check(loadedConfig != null && backend != "failed") { "MNN model is not loaded" }
        stopped.set(false)
        val roles = request.messages.map { message ->
            when (message.role) {
                ChatMessage.Role.SYSTEM -> "system"
                ChatMessage.Role.USER -> "user"
                ChatMessage.Role.ASSISTANT -> "assistant"
                ChatMessage.Role.TOOL -> "tool"
            }
        }.toTypedArray()
        val contents = request.messages.map(ChatMessage::content).toTypedArray()
        val started = System.nanoTime()
        val text = MnnNative.nativeGenerate(roles, contents, request.settings.maxNewTokens)
        val elapsed = (System.nanoTime() - started) / 1_000_000
        backend = runCatching { MnnNative.nativeBackend() }.getOrDefault(backend)
        if (text.startsWith("ERROR:")) {
            backend = runCatching { MnnNative.nativeBackend() }.getOrDefault("failed")
            throw IllegalStateException("MNN generation failed: ${text.removePrefix("ERROR:")}")
        }
        // Buffer before publishing so failed OpenCL output is never concatenated with a CPU retry.
        if (!stopped.get()) onToken(text)
        GenerationResult(text = text, generationTimeMs = elapsed, stopped = stopped.get()).also { lastResult = it }
    }

    override suspend fun stopGeneration() {
        stopped.set(true)
        runCatching { MnnNative.nativeStop() }
    }

    override fun runtimeInfo(): RuntimeInfo = RuntimeInfo(
        name = "MNN-LLM",
        version = "3.6.1",
        backend = backend,
        threads = if (backend == "cpu") cpuThreads else null,
        gpuLayers = null,
        contextLength = null,
    )
    override fun lastGeneration(): GenerationResult? = lastResult
    override fun lastLoadTimeMs(): Long? = lastLoadMs
}
