package com.woogit.aicore.runtime

import com.woogit.aicore.domain.GenerationResult
import com.woogit.aicore.domain.InferenceSettings
import com.woogit.aicore.domain.ModelDescriptor
import com.woogit.aicore.domain.RuntimeInfo
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

data class RuntimeTraceEvent(
    val type: Type,
    val message: String? = null,
    val timestampMs: Long = System.currentTimeMillis(),
) {
    enum class Type { MODEL_LOAD_COMPLETED, GENERATION_STARTED, FIRST_TOKEN, GENERATION_COMPLETED, GENERATION_STOPPED, GENERATION_FAILED }
}

/** Runtime-owned diagnostics. Values are populated only by real runtime operations. */
data class RuntimeDiagnosticsSnapshot(
    val model: ModelDescriptor? = null,
    val runtime: RuntimeInfo = RuntimeInfo("unknown", "unknown", null),
    val loadTimeMs: Long? = null,
    val generation: GenerationResult? = null,
    val settings: InferenceSettings? = null,
    val generationStartedAtMs: Long? = null,
    val trace: List<RuntimeTraceEvent> = emptyList(),
)

object RuntimeDiagnosticsStore {
    private val state = MutableStateFlow(RuntimeDiagnosticsSnapshot())
    val snapshot: StateFlow<RuntimeDiagnosticsSnapshot> = state.asStateFlow()

    fun recordTrace(type: RuntimeTraceEvent.Type, message: String? = null) {
        val current = state.value
        state.value = current.copy(trace = (current.trace + RuntimeTraceEvent(type, message)).takeLast(100))
    }

    fun recordLoaded(model: ModelDescriptor, loadTimeMs: Long?, runtime: RuntimeInfo) {
        state.value = state.value.copy(
            model = model,
            runtime = runtime,
            loadTimeMs = loadTimeMs,
            generation = null,
            settings = null,
        )
        recordTrace(RuntimeTraceEvent.Type.MODEL_LOAD_COMPLETED, loadTimeMs?.let { "loadMs=$it" })
    }

    fun recordGeneration(settings: InferenceSettings, result: GenerationResult, runtime: RuntimeInfo) {
        state.value = state.value.copy(
            runtime = runtime,
            generation = result,
            settings = settings,
            generationStartedAtMs = System.currentTimeMillis() - (result.generationTimeMs ?: 0L),
        )
        recordTrace(
            if (result.stopped) RuntimeTraceEvent.Type.GENERATION_STOPPED else RuntimeTraceEvent.Type.GENERATION_COMPLETED,
            "generationMs=${result.generationTimeMs ?: "n/a"}",
        )
    }

    fun clearGeneration() {
        state.value = state.value.copy(generation = null, settings = null, generationStartedAtMs = null)
    }
}
