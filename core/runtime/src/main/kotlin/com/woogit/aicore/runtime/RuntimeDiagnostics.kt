package com.woogit.aicore.runtime

import com.woogit.aicore.domain.GenerationResult
import com.woogit.aicore.domain.InferenceSettings
import com.woogit.aicore.domain.ModelDescriptor
import com.woogit.aicore.domain.RuntimeInfo
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.io.File
import java.util.Locale

data class RuntimeTraceEvent(
    val type: Type,
    val message: String? = null,
    val timestampMs: Long = System.currentTimeMillis(),
) {
    enum class Type { MODEL_LOAD_COMPLETED, GENERATION_STARTED, FIRST_TOKEN, GENERATION_COMPLETED, GENERATION_STOPPED, GENERATION_FAILED }
}

data class OpenClKernelTiming(
    val opName: String,
    val kernelName: String,
    val executionMs: Double,
)

data class OpenClGpuProfile(
    val kernelCount: Int = 0,
    val totalKernelMs: Double? = null,
    val q6KMulMatMs: Double? = null,
    val attentionMs: Double? = null,
    val ropeMs: Double? = null,
    val rmsNormMs: Double? = null,
    val ffnMs: Double? = null,
    val softmaxMs: Double? = null,
    val kernelLaunchMs: Double? = null,
    val kernelSubmitMs: Double? = null,
    val syncMs: Double? = null,
    val memoryTransferMs: Double? = null,
    val topKernels: List<OpenClKernelTiming> = emptyList(),
) {
    fun reportLines(): List<String> {
        fun ms(v: Double?): String = v?.let { String.format(Locale.US, "%.3f ms", it) } ?: "N/A"
        return listOf(
            "Kernel count: $kernelCount",
            "MUL_MAT Q6_K: ${ms(q6KMulMatMs)}",
            "Attention: ${ms(attentionMs)}",
            "RoPE: ${ms(ropeMs)}",
            "RMSNorm: ${ms(rmsNormMs)}",
            "FFN: ${ms(ffnMs)}",
            "Softmax: ${ms(softmaxMs)}",
            "Kernel launch: ${ms(kernelLaunchMs)}",
            "Kernel submit: ${ms(kernelSubmitMs)}",
            "Kernel execution: ${ms(totalKernelMs)}",
            "Kernel completion/sync: ${ms(syncMs)}",
            "Memory transfer: ${ms(memoryTransferMs)}",
        )
    }
}

data class GpuDeviceProfile(
    val name: String,
    val description: String,
    val memoryFreeMiB: Double?,
    val memoryTotalMiB: Double?,
    val memoryKnown: Boolean,
) {
    val memoryUsedMiB: Double?
        get() = if (memoryKnown && memoryFreeMiB != null && memoryTotalMiB != null) {
            (memoryTotalMiB - memoryFreeMiB).coerceAtLeast(0.0)
        } else null
}

data class GpuWeightResidency(
    val gpuTensorMiB: Double?,
    val gpuTensors: Int?,
    val gpuBuffers: Int?,
    val hostTensorMiB: Double?,
    val cpuTensorMiB: Double?,
    val otherTensorMiB: Double?,
    val gpuFreeMiB: Double?,
    val gpuTotalMiB: Double?,
)

data class RuntimeDiagnosticsSnapshot(
    val model: ModelDescriptor? = null,
    val runtime: RuntimeInfo = RuntimeInfo("unknown", "unknown", null),
    val loadTimeMs: Long? = null,
    val generation: GenerationResult? = null,
    val settings: InferenceSettings? = null,
    val generationStartedAtMs: Long? = null,
    val trace: List<RuntimeTraceEvent> = emptyList(),
    val lastNativeEvent: String? = null,
    val openClProfile: OpenClGpuProfile? = null,
    val gpuDevice: GpuDeviceProfile? = null,
    val weightResidency: GpuWeightResidency? = null,
)

object RuntimeDiagnosticsStore {
    private const val MAX_NATIVE_DIAGNOSTIC_LINES = 1000
    private val markerFile = File(System.getProperty("java.io.tmpdir") ?: ".", "ai-chat-last-native-event.txt")
    private val preflightFile = File(System.getProperty("java.io.tmpdir") ?: ".", "ai-chat-model-preflight.txt")
    private val nativeTraceFile = File(System.getProperty("java.io.tmpdir") ?: ".", "ai-chat-native-trace.txt")
    private val state = MutableStateFlow(RuntimeDiagnosticsSnapshot(lastNativeEvent = readNativeDiagnostics(), openClProfile = readOpenClProfile(), gpuDevice = readGpuDevice(), weightResidency = readWeightResidency()))
    val snapshot: StateFlow<RuntimeDiagnosticsSnapshot> = state.asStateFlow()

    private fun lastNativeDiagnosticLines(text: String): String =
        text.lineSequence().toList().takeLast(MAX_NATIVE_DIAGNOSTIC_LINES).joinToString("\n")

    private fun readNativeDiagnostics(): String? = runCatching {
        val preflight = preflightFile.takeIf { it.isFile }?.readText()?.trim()?.takeIf { it.isNotBlank() }
        val marker = markerFile.takeIf { it.isFile }?.readText()?.trim()?.takeIf { it.isNotBlank() }
        val trace = nativeTraceFile.takeIf { it.isFile }?.readText()?.trim()?.takeIf { it.isNotBlank() }
        listOfNotNull(
            preflight,
            marker?.let { "LAST_NATIVE_EVENT=$it" },
            trace?.let { "===== NATIVE TRACE =====\n${lastNativeDiagnosticLines(it)}" },
        ).joinToString("\n===== NATIVE DIAGNOSTICS =====\n").let(::lastNativeDiagnosticLines).takeIf { it.isNotBlank() }
    }.getOrNull()

    fun refreshNativeEvent() {
        state.value = state.value.copy(lastNativeEvent = readNativeDiagnostics(), openClProfile = readOpenClProfile(), gpuDevice = readGpuDevice(), weightResidency = readWeightResidency())
    }

    fun recordTrace(type: RuntimeTraceEvent.Type, message: String? = null) {
        val current = state.value
        state.value = current.copy(trace = (current.trace + RuntimeTraceEvent(type, message)).takeLast(100))
    }

    fun recordNativeEvent(message: String) {
        state.value = state.value.copy(lastNativeEvent = lastNativeDiagnosticLines(message))
        runCatching {
            markerFile.parentFile?.mkdirs()
            markerFile.writeText(message)
        }
    }

    fun recordLoaded(model: ModelDescriptor, loadTimeMs: Long?, runtime: RuntimeInfo) {
        state.value = state.value.copy(model = model, runtime = runtime, loadTimeMs = loadTimeMs, generation = null, settings = null, openClProfile = readOpenClProfile(), gpuDevice = readGpuDevice(), weightResidency = readWeightResidency())
        recordTrace(RuntimeTraceEvent.Type.MODEL_LOAD_COMPLETED, loadTimeMs?.let { "loadMs=$it" })
    }

    fun recordGeneration(settings: InferenceSettings, result: GenerationResult, runtime: RuntimeInfo) {
        state.value = state.value.copy(runtime = runtime, generation = result, settings = settings, generationStartedAtMs = System.currentTimeMillis() - (result.generationTimeMs ?: 0L), openClProfile = readOpenClProfile(), gpuDevice = readGpuDevice(), weightResidency = readWeightResidency())
        recordTrace(if (result.stopped) RuntimeTraceEvent.Type.GENERATION_STOPPED else RuntimeTraceEvent.Type.GENERATION_COMPLETED, "generationMs=${result.generationTimeMs ?: "n/a"}")
    }

    private fun nativeLine(prefix: String): String? {
        val text = readNativeDiagnostics() ?: return null
        return text.lineSequence().toList().asReversed().firstOrNull { it.contains(prefix) }
    }

    private fun nativeField(line: String, key: String): String? =
        Regex("""\b${Regex.escape(key)}=([^\s]+)""").find(line)?.groupValues?.get(1)

    private fun readGpuDevice(): GpuDeviceProfile? = runCatching {
        val line = nativeLine("NATIVE_OPENCL_DEVICE") ?: return@runCatching null
        GpuDeviceProfile(
            name = nativeField(line, "name")?.replace('_', ' ') ?: "unknown",
            description = nativeField(line, "description")?.replace('_', ' ') ?: "unknown",
            memoryFreeMiB = nativeField(line, "memoryFreeMiB")?.toDoubleOrNull(),
            memoryTotalMiB = nativeField(line, "memoryTotalMiB")?.toDoubleOrNull(),
            memoryKnown = nativeField(line, "memoryKnown") == "1",
        )
    }.getOrNull()

    private fun readWeightResidency(): GpuWeightResidency? = runCatching {
        val line = nativeLine("NATIVE_WEIGHT_RESIDENCY") ?: return@runCatching null
        GpuWeightResidency(
            gpuTensorMiB = nativeField(line, "gpuTensorMiB")?.toDoubleOrNull(),
            gpuTensors = nativeField(line, "gpuTensors")?.toIntOrNull(),
            gpuBuffers = nativeField(line, "gpuBuffers")?.toIntOrNull(),
            hostTensorMiB = nativeField(line, "hostTensorMiB")?.toDoubleOrNull(),
            cpuTensorMiB = nativeField(line, "cpuTensorMiB")?.toDoubleOrNull(),
            otherTensorMiB = nativeField(line, "otherTensorMiB")?.toDoubleOrNull(),
            gpuFreeMiB = nativeField(line, "gpuFreeMiB")?.toDoubleOrNull(),
            gpuTotalMiB = nativeField(line, "gpuTotalMiB")?.toDoubleOrNull(),
        )
    }.getOrNull()

    private fun readOpenClProfile(): OpenClGpuProfile? = runCatching {
        val root = System.getenv("TMPDIR")?.takeIf { it.isNotBlank() }
            ?: System.getProperty("java.io.tmpdir")?.takeIf { it.isNotBlank() }
            ?: return@runCatching null
        val candidates = listOf(
            File(root, "cl_profiling.csv"),
            File("/data/user/0/com.samanramezani.aichattest/cache", "cl_profiling.csv"),
            File("/data/data/com.samanramezani.aichattest/cache", "cl_profiling.csv"),
            File(".", "cl_profiling.csv"),
        ).distinctBy { it.absolutePath }
        val file = candidates.firstOrNull { it.isFile && it.length() > 0L }
            ?: return@runCatching null
        readOpenClProfileFile(file)
    }.getOrNull()

    /**
     * Reads one real OpenCL profiling CSV exactly as produced by the native V4 writer.
     * Kept separate from path discovery so the file read + parse path can be tested
     * deterministically without requiring an OpenCL-capable CI device.
     */
    internal fun readOpenClProfileFile(
        file: File,
        transferFile: File = File(file.parentFile ?: File("."), "cl_transfer_profile.txt"),
    ): OpenClGpuProfile? = runCatching {
        if (!file.isFile || file.length() <= 0L) return@runCatching null
        val transferMs = transferFile.takeIf { it.isFile }?.readText()?.let { text ->
            Regex("""memory_transfer_ms=([0-9.+-Ee]+)""").find(text)?.groupValues?.get(1)?.toDoubleOrNull()
        }

        val rows = file.readLines().drop(1).mapNotNull { line ->
            val p = line.split(',')
            if (p.size < 10) return@mapNotNull null
            val exec = p[4].toDoubleOrNull() ?: return@mapNotNull null
            OpenClProfileRow(
                opName = p[0],
                kernelName = p[1],
                queueMs = p[2].toDoubleOrNull() ?: 0.0,
                submitMs = p[3].toDoubleOrNull() ?: 0.0,
                execMs = exec,
                completeMs = p[5].toDoubleOrNull() ?: 0.0,
            )
        }
        if (rows.isEmpty()) return@runCatching null
        fun sum(match: (OpenClProfileRow) -> Boolean): Double = rows.filter(match).sumOf { it.execMs }
        val q6 = sum {
            it.kernelName.contains("q6_k", true) &&
                (it.kernelName.contains("mul", true) || it.kernelName.contains("gemv", true) || it.kernelName.contains("gemm", true))
        }
        val attention = sum {
            val n = (it.opName + " " + it.kernelName).lowercase(Locale.US)
            n.contains("flash_attn") || n.contains("kqv") || n.contains("attn")
        }
        val rope = sum { it.opName.contains("rope", true) || it.kernelName.contains("rope", true) }
        val rms = sum { it.opName.contains("rms_norm", true) || it.kernelName.contains("rms_norm", true) }
        val ffn = sum {
            val n = (it.opName + " " + it.kernelName).lowercase(Locale.US)
            n.contains("ffn") || n.contains("glu")
        }
        val softmax = sum { it.opName.contains("softmax", true) || it.kernelName.contains("softmax", true) }
        OpenClGpuProfile(
            kernelCount = rows.size,
            totalKernelMs = rows.sumOf { it.execMs },
            q6KMulMatMs = q6,
            attentionMs = attention,
            ropeMs = rope,
            rmsNormMs = rms,
            ffnMs = ffn,
            softmaxMs = softmax,
            kernelLaunchMs = rows.sumOf { it.queueMs },
            kernelSubmitMs = rows.sumOf { it.submitMs },
            syncMs = rows.sumOf { it.completeMs },
            memoryTransferMs = transferMs,
            topKernels = rows.groupBy { it.kernelName }
                .map { (name, items) -> OpenClKernelTiming(items.first().opName, name, items.sumOf { it.execMs }) }
                .sortedByDescending { it.executionMs }
                .take(5),
        )
    }.getOrNull()

    private data class OpenClProfileRow(
        val opName: String,
        val kernelName: String,
        val queueMs: Double,
        val submitMs: Double,
        val execMs: Double,
        val completeMs: Double,
    )

    fun clearGeneration() {
        state.value = state.value.copy(generation = null, settings = null, generationStartedAtMs = null)
    }
}
