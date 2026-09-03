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

data class RuntimeDiagnosticsSnapshot(
    val model: ModelDescriptor? = null,
    val runtime: RuntimeInfo = RuntimeInfo("unknown", "unknown", null),
    val loadTimeMs: Long? = null,
    val generation: GenerationResult? = null,
    val settings: InferenceSettings? = null,
    val generationStartedAtMs: Long? = null,
    val trace: List<RuntimeTraceEvent> = emptyList(),
    val lastNativeEvent: String? = null,
)

object RuntimeDiagnosticsStore {
    private val markerFile = File(System.getProperty("java.io.tmpdir") ?: ".", "ai-chat-last-native-event.txt")
    private val preflightFile = File(System.getProperty("java.io.tmpdir") ?: ".", "ai-chat-model-preflight.txt")
    private val nativeTraceFile = File(System.getProperty("java.io.tmpdir") ?: ".", "ai-chat-native-trace.txt")
    private val state = MutableStateFlow(RuntimeDiagnosticsSnapshot(lastNativeEvent = readNativeDiagnostics()))
    val snapshot: StateFlow<RuntimeDiagnosticsSnapshot> = state.asStateFlow()

    private fun readNativeDiagnostics(): String? = runCatching {
        val preflight = preflightFile.takeIf { it.isFile }?.readText()?.trim()?.takeIf { it.isNotBlank() }
        val marker = markerFile.takeIf { it.isFile }?.readText()?.trim()?.takeIf { it.isNotBlank() }
        val trace = nativeTraceFile.takeIf { it.isFile }?.readText()?.trim()?.takeIf { it.isNotBlank() }
        listOfNotNull(
            preflight,
            marker?.let { "LAST_NATIVE_EVENT=$it" },
            trace?.let { "===== NATIVE TRACE =====\n$it" },
        ).joinToString("\n===== NATIVE DIAGNOSTICS =====\n").takeIf { it.isNotBlank() }
    }.getOrNull()

    fun refreshNativeEvent() {
        state.value = state.value.copy(lastNativeEvent = readNativeDiagnostics())
    }

    fun recordTrace(type: RuntimeTraceEvent.Type, message: String? = null) {
        val current = state.value
        state.value = current.copy(trace = (current.trace + RuntimeTraceEvent(type, message)).takeLast(100))
    }

    fun recordNativeEvent(message: String) {
        state.value = state.value.copy(lastNativeEvent = message)
        runCatching {
            markerFile.parentFile?.mkdirs()
            markerFile.writeText(message)
        }
    }

    fun recordLoaded(model: ModelDescriptor, loadTimeMs: Long?, runtime: RuntimeInfo) {
        state.value = state.value.copy(model = model, runtime = runtime, loadTimeMs = loadTimeMs, generation = null, settings = null)
        recordTrace(RuntimeTraceEvent.Type.MODEL_LOAD_COMPLETED, loadTimeMs?.let { "loadMs=$it" })
    }

    fun recordGeneration(settings: InferenceSettings, result: GenerationResult, runtime: RuntimeInfo) {
        state.value = state.value.copy(runtime = runtime, generation = result, settings = settings, generationStartedAtMs = System.currentTimeMillis() - (result.generationTimeMs ?: 0L))
        recordTrace(if (result.stopped) RuntimeTraceEvent.Type.GENERATION_STOPPED else RuntimeTraceEvent.Type.GENERATION_COMPLETED, "generationMs=${result.generationTimeMs ?: "n/a"}")
    }

    fun clearGeneration() {
        state.value = state.value.copy(generation = null, settings = null, generationStartedAtMs = null)
    }
}
