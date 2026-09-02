package com.samanramezani.aichattest.ui.diagnostics

import com.woogit.aicore.actions.ActionTraceEvent
import com.woogit.aicore.domain.GenerationResult
import com.woogit.aicore.domain.InferenceSettings
import com.woogit.aicore.domain.ModelDescriptor
import com.woogit.aicore.domain.RuntimeInfo
import com.woogit.aicore.runtime.RuntimeTraceEvent

internal data class RuntimeDiagnostic(
    val model: ModelDescriptor?,
    val runtime: RuntimeInfo,
    val loadTimeMs: Long?,
    val generation: GenerationResult?,
    val settings: InferenceSettings?,
    val status: String,
    val error: String?,
    val rawError: String? = null,
    val executionId: String? = null,
    val actionTrace: List<ActionTraceEvent> = emptyList(),
    val runtimeTrace: List<RuntimeTraceEvent> = emptyList(),
)

internal fun RuntimeDiagnostic.report(): String = buildString {
    appendLine("AI Chat Test — Runtime Diagnostic Report")
    appendLine()
    appendLine("Model: ${model?.displayName ?: "N/A"}")
    appendLine("Format: ${model?.format ?: "N/A"}")
    appendLine("Quantization: ${model?.quantization ?: "N/A"}")
    appendLine("Runtime: ${runtime.name}")
    appendLine("Runtime Version: ${runtime.version}")
    appendLine("Backend: ${runtime.backend ?: "N/A"}")
    appendLine("Threads: ${runtime.threads ?: "N/A"}")
    appendLine("GPU Layers: ${runtime.gpuLayers ?: "N/A"}")
    appendLine("Context: ${runtime.contextLength ?: "N/A"}")
    appendLine()
    appendLine("Load Time: ${loadTimeMs?.let { "$it ms" } ?: "N/A"}")
    appendLine("TTFT: ${generation?.firstTokenTimeMs?.let { "$it ms" } ?: "N/A"}")
    appendLine("Generation Time: ${generation?.generationTimeMs?.let { "$it ms" } ?: "N/A"}")
    appendLine("Prompt Tokens: ${generation?.inputTokens ?: "N/A"}")
    appendLine("Output Tokens: ${generation?.outputTokens ?: "N/A"}")
    appendLine("Tokens/sec: ${tokensPerSecond(generation)?.let { "%.2f".format(it) } ?: "N/A"}")
    appendLine()
    val current = settings
    appendLine("Temperature: ${current?.temperature ?: "N/A"}")
    appendLine("Top-P: ${current?.topP ?: "N/A"}")
    appendLine("Top-K: ${current?.topK ?: "N/A"}")
    appendLine("Min-P: ${current?.minP ?: "N/A"}")
    appendLine("Repeat Penalty: ${current?.repeatPenalty ?: "N/A"}")
    appendLine("Max New Tokens: ${current?.maxNewTokens ?: "N/A"}")
    appendLine("Context Setting: ${current?.contextLength ?: "N/A"}")
    appendLine("Stop Sequences: ${current?.stopSequences?.size ?: "N/A"}")
    appendLine("Seed: ${current?.seed ?: "N/A"}")
    appendLine()
    appendLine("Status: $status")
    appendLine("Execution: ${executionId ?: "N/A"}")
    if (!error.isNullOrBlank()) appendLine("Error: $error")
    if (!rawError.isNullOrBlank()) appendLine("Raw Error: $rawError")
    appendLine()
    appendLine("Runtime Trace")
    if (runtimeTrace.isEmpty()) appendLine("N/A")
    else runtimeTrace.forEach { event -> appendLine("${event.timestampMs} | ${event.type} | ${event.message ?: ""}".trimEnd()) }
    appendLine()
    appendLine("Action Trace")
    if (actionTrace.isEmpty()) appendLine("N/A")
    else actionTrace.forEach { event -> appendLine("${event.timestampMs} | ${event.type} | ${event.message ?: ""}".trimEnd()) }
}

internal fun RuntimeDiagnostic.errorReport(): String = buildString {
    appendLine("AI Chat Test — Error Report")
    appendLine("Status: $status")
    appendLine("Execution: ${executionId ?: "N/A"}")
    appendLine("Model: ${model?.displayName ?: "N/A"}")
    appendLine("Runtime: ${runtime.name} ${runtime.version}")
    appendLine("Error: ${error ?: "N/A"}")
    appendLine("Raw Error: ${rawError ?: "N/A"}")
}

private fun tokensPerSecond(result: GenerationResult?): Double? {
    val tokens = result?.outputTokens ?: return null
    val timeMs = result.generationTimeMs ?: return null
    if (tokens <= 0 || timeMs <= 0) return null
    return tokens.toDouble() / (timeMs.toDouble() / 1000.0)
}
