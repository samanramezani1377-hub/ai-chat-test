package com.woogit.aicore.runtime

import com.woogit.aicore.domain.GenerationResult
import com.woogit.aicore.domain.InferenceSettings
import com.woogit.aicore.domain.ModelDescriptor
import com.woogit.aicore.domain.RuntimeInfo
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/** Runtime-owned diagnostics. Values are populated only by real runtime operations. */
data class RuntimeDiagnosticsSnapshot(
    val model: ModelDescriptor? = null,
    val runtime: RuntimeInfo = RuntimeInfo("unknown", "unknown", null),
    val loadTimeMs: Long? = null,
    val generation: GenerationResult? = null,
    val settings: InferenceSettings? = null,
    val generationStartedAtMs: Long? = null,
)

object RuntimeDiagnosticsStore {
    private val state = MutableStateFlow(RuntimeDiagnosticsSnapshot())
    val snapshot: StateFlow<RuntimeDiagnosticsSnapshot> = state.asStateFlow()

    fun recordLoaded(model: ModelDescriptor, loadTimeMs: Long?, runtime: RuntimeInfo) {
        state.value = state.value.copy(model = model, runtime = runtime, loadTimeMs = loadTimeMs, generation = null, settings = null)
    }

    fun recordGeneration(settings: InferenceSettings, result: GenerationResult, runtime: RuntimeInfo) {
        state.value = state.value.copy(
            runtime = runtime,
            generation = result,
            settings = settings,
            generationStartedAtMs = System.currentTimeMillis() - (result.generationTimeMs ?: 0L),
        )
    }

    fun clearGeneration() {
        state.value = state.value.copy(generation = null, settings = null, generationStartedAtMs = null)
    }
}
