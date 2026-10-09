package com.woogit.aicore.runtime

import com.woogit.aicore.domain.GenerationResult
import com.woogit.aicore.domain.InferenceSettings
import com.woogit.aicore.domain.ModelDescriptor
import com.woogit.aicore.domain.RuntimeInfo
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.io.File

data class RuntimeTraceEvent(
    val type: Type,
    val message: String? = null,
    val timestampMs: Long = System.currentTimeMillis(),
) {
    enum class Type { MODEL_LOAD_COMPLETED, GENERATION_STARTED, FIRST_TOKEN, GENERATION_COMPLETED, GENERATION_STOPPED, GENERATION_FAILED }
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
    val gpuDevice: GpuDeviceProfile? = null,
    val weightResidency: GpuWeightResidency? = null,
)

object RuntimeDiagnosticsStore {
    private const val MAX_NATIVE_DIAGNOSTIC_LINES = 80
    private val markerFile = File(System.getProperty("java.io.tmpdir") ?: ".", "ai-chat-last-native-event.txt")
    private val preflightFile = File(System.getProperty("java.io.tmpdir") ?: ".", "ai-chat-model-preflight.txt")
    private val nativeTraceFile = File(System.getProperty("java.io.tmpdir") ?: ".", "ai-chat-native-trace.txt")
    private val state = MutableStateFlow(RuntimeDiagnosticsSnapshot(lastNativeEvent = readNativeDiagnostics(), gpuDevice = readGpuDevice(), weightResidency = readWeightResidency()))
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
        state.value = state.value.copy(lastNativeEvent = readNativeDiagnostics(), gpuDevice = readGpuDevice(), weightResidency = readWeightResidency())
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
        state.value = state.value.copy(model = model, runtime = runtime, loadTimeMs = loadTimeMs, generation = null, settings = null, gpuDevice = readGpuDevice(), weightResidency = readWeightResidency())
        recordTrace(RuntimeTraceEvent.Type.MODEL_LOAD_COMPLETED, loadTimeMs?.let { "loadMs=$it" })
    }

    fun recordGeneration(settings: InferenceSettings, result: GenerationResult, runtime: RuntimeInfo) {
        state.value = state.value.copy(runtime = runtime, generation = result, settings = settings, generationStartedAtMs = System.currentTimeMillis() - (result.generationTimeMs ?: 0L), gpuDevice = readGpuDevice(), weightResidency = readWeightResidency())
        recordTrace(if (result.stopped) RuntimeTraceEvent.Type.GENERATION_STOPPED else RuntimeTraceEvent.Type.GENERATION_COMPLETED, "generationMs=${result.generationTimeMs ?: "n/a"}")
    }

    private fun nativeLine(prefix: String): String? {
        val text = readNativeDiagnostics() ?: return null
        return text.lineSequence().toList().asReversed().firstOrNull { it.contains(prefix) }
    }

    private fun nativeField(line: String, key: String): String? =
        Regex("""\b${Regex.escape(key)}=([^\s]+)""").find(line)?.groupValues?.get(1)

    private fun readGpuDevice(): GpuDeviceProfile? = runCatching {
        val line = nativeLine("NATIVE_VULKAN_DEVICE") ?: return@runCatching null
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

    fun clearGeneration() {
        state.value = state.value.copy(generation = null, settings = null, generationStartedAtMs = null)
    }
}
