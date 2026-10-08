package com.samanramezani.aichattest.ui.diagnostics

import com.woogit.aicore.actions.ActionTraceEvent
import com.woogit.aicore.domain.GenerationResult
import com.woogit.aicore.domain.InferenceSettings
import com.woogit.aicore.domain.ModelDescriptor
import com.woogit.aicore.domain.RuntimeInfo
import com.woogit.aicore.runtime.RuntimeTraceEvent
import com.woogit.aicore.runtime.OpenClGpuProfile
import com.samanramezani.aichattest.ui.errors.ErrorCenter

private const val RECENT_RUNTIME_EVENTS = 20
private const val RECENT_ACTION_EVENTS = 10

internal data class NativePerformance(
    val prefillMs: Long? = null,
    val promptTokens: Int? = null,
    val cacheStatus: String? = null,
    val cachedTokens: Int? = null,
    val reusedTokens: Int? = null,
    val newTokens: Int? = null,
    val cacheHitRatio: Double? = null,
    val generatedTokens: Int? = null,
    val generationMs: Long? = null,
    val decodeMs: Long? = null,
    val decodeTokensPerSec: Double? = null,
    val speculativeDraftTokens: Int? = null,
    val speculativeAcceptedTokens: Int? = null,
    val speculativeAcceptanceRate: Double? = null,
    val speculativeSteps: Int? = null,
    val speculativeMeanAcceptedPerStep: Double? = null,
    val profileDecodeMs: Long? = null,
    val profileLogitsSyncMs: Long? = null,
    val profileSamplingMs: Long? = null,
    val profileCallbackMs: Long? = null,
    val profileTokenSteps: Int? = null,
    val profileLogitsAccesses: Int? = null,
    val profileCallbackCalls: Int? = null,
    val profileAccountedMs: Long? = null,
    val profileUnaccountedMs: Long? = null,
    val profileDecodeWindowMs: Long? = null,
    val speculativeGenerationMs: Long? = null,
    val speculativeTokensPerSec: Double? = null,
    val speculativeFallbackCode: Int? = null,
) {
}

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
    val nativeDiagnostics: String? = null,
    val openClProfile: OpenClGpuProfile? = null,
    val gpuDevice: com.woogit.aicore.runtime.GpuDeviceProfile? = null,
    val weightResidency: com.woogit.aicore.runtime.GpuWeightResidency? = null,
) {
    val tensorResidencyLines: List<String>
        get() = nativeDiagnostics.orEmpty().lineSequence()
            .filter { it.contains("OPENGL_ES_TENSOR_RESIDENCY") }
            .distinct()
            .toList()

    val nonGpuTensorLines: List<String>
        get() = nativeDiagnostics.orEmpty().lineSequence()
            .filter { it.contains("OPENGL_ES_NON_GPU_TENSOR") }
            .distinct()
            .toList()

    val residencySummaryLines: List<String>
        get() = nativeDiagnostics.orEmpty().lineSequence()
            .filter { it.contains("OPENGL_ES_RESIDENCY_SUMMARY") }
            .distinct()
            .toList()
    val nativePerformance: NativePerformance
        get() = NativePerformance(
            prefillMs = nativeValue("prefillMs")?.toLongOrNull(),
            promptTokens = nativeValue("promptTokens")?.toIntOrNull(),
            cacheStatus = nativeCacheValue("status"),
            cachedTokens = nativeCacheValue("cachedTokens")?.toIntOrNull(),
            reusedTokens = nativeCacheValue("reusedTokens")?.toIntOrNull(),
            newTokens = nativeCacheValue("newTokens")?.toIntOrNull(),
            cacheHitRatio = nativeCacheValue("hitRatio")?.toDoubleOrNull(),
            generatedTokens = nativeValue("generatedTokens")?.toIntOrNull(),
            generationMs = nativeValue("generationMs")?.toLongOrNull(),
            decodeMs = nativeValue("decodeMs")?.toLongOrNull(),
            decodeTokensPerSec = nativeValue("decodeTokensPerSec")?.toDoubleOrNull(),
            speculativeDraftTokens = nativeValue("draftTokens")?.toIntOrNull(),
            speculativeAcceptedTokens = nativeValue("acceptedTokens")?.toIntOrNull(),
            speculativeAcceptanceRate = nativeValue("acceptanceRate")?.toDoubleOrNull(),
            speculativeSteps = nativeValue("steps")?.toIntOrNull(),
            speculativeMeanAcceptedPerStep = nativeValue("meanAcceptedPerStep")?.toDoubleOrNull(),
            profileDecodeMs = nativeProfileValue("decodeMs")?.toLongOrNull(),
            profileLogitsSyncMs = nativeProfileValue("logitsSyncMs")?.toLongOrNull(),
            profileSamplingMs = nativeProfileValue("samplingMs")?.toLongOrNull(),
            profileCallbackMs = nativeProfileValue("callbackMs")?.toLongOrNull(),
            profileTokenSteps = nativeProfileValue("tokenSteps")?.toIntOrNull(),
            profileLogitsAccesses = nativeProfileValue("logitsAccesses")?.toIntOrNull(),
            profileCallbackCalls = nativeProfileValue("callbackCalls")?.toIntOrNull(),
            profileAccountedMs = nativeProfileValue("accountedMs")?.toLongOrNull(),
            profileUnaccountedMs = nativeProfileValue("unaccountedMs")?.toLongOrNull(),
            profileDecodeWindowMs = nativeProfileValue("decodeWindowMs")?.toLongOrNull(),
            speculativeGenerationMs = nativeSpecValue("generationMs")?.toLongOrNull(),
            speculativeTokensPerSec = nativeSpecValue("tokensPerSec")?.toDoubleOrNull(),
            speculativeFallbackCode = nativeSpecValue("code")?.toIntOrNull(),
        )

    private fun nativeValue(key: String): String? {
        val text = nativeDiagnostics ?: return null
        val line = text.lineSequence().toList().asReversed().firstOrNull { it.contains("$key=") } ?: return null
        return Regex("""\b${Regex.escape(key)}=([^\s]+)""").find(line)?.groupValues?.get(1)
    }

    private fun nativeCacheValue(key: String): String? {
        val text = nativeDiagnostics ?: return null
        val line = text.lineSequence().toList().asReversed().firstOrNull { it.contains("NATIVE_KV_CACHE_RESULT") && it.contains("$key=") } ?: return null
        return Regex("""\b${Regex.escape(key)}=([^\s]+)""").find(line)?.groupValues?.get(1)
    }

    private fun nativeSpecValue(key: String): String? {
        val text = nativeDiagnostics ?: return null
        val lines = text.lineSequence().toList().asReversed()
        val line = lines.firstOrNull { it.contains("SPECULATIVE_PERFORMANCE") && it.contains("$key=") }
            ?: lines.firstOrNull { it.contains("SPECULATIVE_GENERATE_RETURNED") && it.contains("$key=") }
            ?: return null
        return Regex("""\b${Regex.escape(key)}=([^\s]+)""").find(line)?.groupValues?.get(1)
    }

    private fun nativeProfileValue(key: String): String? {
        val text = nativeDiagnostics ?: return null
        val line = text.lineSequence().toList().asReversed().firstOrNull { it.contains("NATIVE_PERF_PROFILE") && it.contains("$key=") } ?: return null
        return Regex("""\b${Regex.escape(key)}=([^\s]+)""").find(line)?.groupValues?.get(1)
    }
}

internal fun RuntimeDiagnostic.report(): String = buildString {
    val perf = nativePerformance
    appendLine("AI Chat Test — Runtime Diagnostic Report")
    appendLine()
    appendLine("===== EXECUTION =====")
    appendLine("Status: $status")
    appendLine("Execution: ${executionId ?: "N/A"}")
    appendLine("Model: ${model?.displayName ?: "N/A"}")
    appendLine("Format: ${model?.format ?: "N/A"}")
    appendLine("Quantization: ${model?.quantization ?: "N/A"}")
    appendLine("Runtime: ${runtime.name} ${runtime.version}")
    appendLine("Backend: ${runtime.backend ?: "N/A"}")
    appendLine("Threads: ${runtime.threads ?: "N/A"}")
    appendLine("GPU Layers: ${runtime.gpuLayers ?: "N/A"}")
    appendLine("Context: ${runtime.contextLength ?: "N/A"}")
    appendLine()
    appendLine("===== PERFORMANCE =====")
    appendLine("Load Time: ${loadTimeMs?.let { "$it ms" } ?: "N/A"}")
    appendLine("TTFT: ${generation?.firstTokenTimeMs?.let { "$it ms" } ?: "N/A"}")
    appendLine("Generation Time: ${perf.generationMs?.let { "$it ms" } ?: generation?.generationTimeMs?.let { "$it ms" } ?: "N/A"}")
    appendLine("Prompt Tokens: ${perf.promptTokens ?: generation?.inputTokens ?: "N/A"}")
    appendLine("Output Tokens: ${perf.generatedTokens ?: generation?.outputTokens ?: "N/A"}")
    appendLine("Tokens/sec: ${perf.decodeTokensPerSec?.let { "%.2f".format(it) } ?: tokensPerSecond(generation)?.let { "%.2f".format(it) } ?: "N/A"}")
    appendLine("Prefill Time: ${perf.prefillMs?.let { "$it ms" } ?: "N/A"}")
    appendLine("Decode Time: ${perf.decodeMs?.let { "$it ms" } ?: "N/A"}")
    appendLine("Decode Tokens/sec: ${perf.decodeTokensPerSec?.let { "%.2f".format(it) } ?: "N/A"}")
    appendLine()
    appendLine("===== GPU / MEMORY =====")
    appendLine("GPU: ${gpuDevice?.name ?: "N/A"}")
    appendLine("GPU Description: ${gpuDevice?.description ?: "N/A"}")
    appendLine("GPU Memory Total: ${gpuDevice?.memoryTotalMiB?.let { "%.1f MiB".format(it) } ?: "Driver did not report"}")
    appendLine("GPU Memory Free: ${gpuDevice?.memoryFreeMiB?.let { "%.1f MiB".format(it) } ?: "Driver did not report"}")
    appendLine("GPU Memory Used: ${gpuDevice?.memoryUsedMiB?.let { "%.1f MiB".format(it) } ?: "N/A"}")
    appendLine("GPU Resident Tensor Memory: ${weightResidency?.gpuTensorMiB?.let { "%.1f MiB".format(it) } ?: "N/A"}")
    appendLine("GPU Resident Tensors: ${weightResidency?.gpuTensors ?: "N/A"}")
    appendLine("GPU Resident Buffers: ${weightResidency?.gpuBuffers ?: "N/A"}")
    appendLine("Host Tensor Memory: ${weightResidency?.hostTensorMiB?.let { "%.1f MiB".format(it) } ?: "N/A"}")
    appendLine()
    appendLine("===== KV CACHE =====")
    appendLine("Status: ${perf.cacheStatus ?: "N/A"}")
    appendLine("Cached Tokens: ${perf.cachedTokens ?: "N/A"}")
    appendLine("Reused Tokens: ${perf.reusedTokens ?: "N/A"}")
    appendLine("New Tokens: ${perf.newTokens ?: "N/A"}")
    appendLine("Hit Ratio: ${perf.cacheHitRatio?.let { "%.1f%%".format(it) } ?: "N/A"}")
    appendLine()
    appendLine("===== DECODE BOTTLENECK PROFILE =====")
    appendLine("Decode: ${perf.profileDecodeMs?.let { "$it ms" } ?: "N/A"}")
    appendLine("Logits Sync: ${perf.profileLogitsSyncMs?.let { "$it ms" } ?: "N/A"}")
    appendLine("CPU Sampling: ${perf.profileSamplingMs?.let { "$it ms" } ?: "N/A"}")
    appendLine("JNI/UI Callback: ${perf.profileCallbackMs?.let { "$it ms" } ?: "N/A"}")
    appendLine("Token Steps: ${perf.profileTokenSteps ?: "N/A"}")
    appendLine("Logits Accesses: ${perf.profileLogitsAccesses ?: "N/A"}")
    appendLine("Callback Calls: ${perf.profileCallbackCalls ?: "N/A"}")
    appendLine("Accounted: ${perf.profileAccountedMs?.let { "$it ms" } ?: "N/A"}")
    appendLine("Unaccounted: ${perf.profileUnaccountedMs?.let { "$it ms" } ?: "N/A"}")
    appendLine("Decode Window: ${perf.profileDecodeWindowMs?.let { "$it ms" } ?: "N/A"}")
    appendLine()
    appendLine("===== SPECULATIVE DECODING =====")
    appendLine("Draft Tokens: ${perf.speculativeDraftTokens ?: "N/A"}")
    appendLine("Accepted Tokens: ${perf.speculativeAcceptedTokens ?: "N/A"}")
    appendLine("Acceptance Rate: ${perf.speculativeAcceptanceRate?.let { "%.1f%%".format(it) } ?: "N/A"}")
    appendLine("Steps: ${perf.speculativeSteps ?: "N/A"}")
    appendLine("Mean Accepted / Step: ${perf.speculativeMeanAcceptedPerStep?.let { "%.2f".format(it) } ?: "N/A"}")
    appendLine("Generation Time: ${perf.speculativeGenerationMs?.let { "$it ms" } ?: "N/A"}")
    appendLine("Tokens/sec: ${perf.speculativeTokensPerSec?.let { "%.2f".format(it) } ?: "N/A"}")
    appendLine("Fallback Code: ${perf.speculativeFallbackCode ?: "N/A"}")
    appendLine()
    appendLine("===== SETTINGS =====")
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
    appendLine("===== ERROR =====")
    if (!error.isNullOrBlank()) {
        val resolved = ErrorCenter.resolve(listOf(error, nativeDiagnostics).filterNotNull().joinToString("\n"))
        appendLine("Code: ${resolved.code}")
        appendLine("Title: ${resolved.title}")
        appendLine("Message: ${resolved.message}")
        appendLine("Recommended Action: ${resolved.action}")
        appendLine("Technical: ${resolved.technical}")
        appendLine("Original: $error")
    } else appendLine("N/A")
    if (!rawError.isNullOrBlank() && rawError != error) appendLine("Raw: $rawError")
    appendLine()
    appendLine("===== TENSOR RESIDENCY =====")
    residencySummaryLines.distinct().ifEmpty { listOf("No residency summary found.") }.forEach(::appendLine)
    val tensors = nativeDiagnostics.orEmpty().lineSequence()
        .filter { it.contains("OPENGL_ES_NON_GPU_TENSOR") || it.contains("OPENGL_ES_TENSOR_RESIDENCY") }
        .distinct()
        .take(200)
    val tensorLines = tensors.toList()
    if (tensorLines.isEmpty()) appendLine("No per-tensor residency entries found.") else tensorLines.forEach(::appendLine)
    appendLine()
    appendLine("===== RECENT EVENTS =====")
    uniqueRuntimeEvents().forEach(::appendLine)
    appendLine()
    appendLine("===== NATIVE DIAGNOSTIC EVENTS =====")
    importantNativeEvents(includeVerbose = false).forEach(::appendLine)
}

internal fun RuntimeDiagnostic.errorReport(): String = buildString {
    appendLine("AI Chat Test — Error Report")
    appendLine()
    appendLine("===== ERROR =====")
    appendLine("Status: $status")
    appendLine("Execution: ${executionId ?: "N/A"}")
    if (!error.isNullOrBlank()) {
        val resolved = ErrorCenter.resolve(listOf(error, nativeDiagnostics).filterNotNull().joinToString("\n"))
        appendLine("Code: ${resolved.code}")
        appendLine("Title: ${resolved.title}")
        appendLine("Message: ${resolved.message}")
        appendLine("Recommended Action: ${resolved.action}")
        appendLine("Technical: ${resolved.technical}")
    } else appendLine("Error: N/A")
    if (!rawError.isNullOrBlank() && rawError != error) appendLine("Raw: $rawError")
    appendLine()
    appendLine("===== FAILURE CONTEXT =====")
    appendLine("Model: ${model?.displayName ?: "N/A"}")
    appendLine("Runtime: ${runtime.name} ${runtime.version}")
    appendLine("Backend: ${runtime.backend ?: "N/A"}")
    appendLine("GPU Layers: ${runtime.gpuLayers ?: "N/A"}")
    appendLine("Context: ${runtime.contextLength ?: "N/A"}")
    appendLine()
    appendLine("===== RECENT EVENTS =====")
    uniqueRuntimeEvents().forEach(::appendLine)
    appendLine()
    appendLine("===== NATIVE FAILURE EVENTS =====")
    importantNativeEvents(includeVerbose = true).forEach(::appendLine)
}

private fun RuntimeDiagnostic.uniqueRuntimeEvents(): List<String> {
    data class EventLine(val timestamp: Long, val text: String, val key: String)
    val runtime = runtimeTrace.asSequence().map {
        EventLine(it.timestampMs, "${it.timestampMs} | ${it.type} | ${it.message ?: ""}".trimEnd(), "${it.type}|${it.message ?: ""}")
    }
    val actions = actionTrace.asSequence().map {
        EventLine(it.timestampMs, "${it.timestampMs} | ${it.type} | ${it.message ?: ""}".trimEnd(), "${it.type}|${it.message ?: ""}")
    }
    return (runtime + actions)
        .sortedBy { it.timestamp }
        .distinctBy { it.key }
        .takeLast(RECENT_RUNTIME_EVENTS + RECENT_ACTION_EVENTS)
        .map { it.text }
}

private fun RuntimeDiagnostic.importantNativeEvents(includeVerbose: Boolean): List<String> {
    val verbose = setOf("create_tensor: loading tensor", "unused tensor")
    return nativeDiagnostics.orEmpty().lineSequence()
        .filter { line ->
            val important = line.contains("OPENGL_ES_") || line.contains("MODEL_LOAD") ||
                line.contains("NATIVE_WEIGHT_RESIDENCY") || line.contains("NATIVE_KV_CACHE") ||
                line.contains("NATIVE_PERF") || line.contains("SPECULATIVE_") ||
                line.contains("ERROR") || line.contains("error") || line.contains("failed") || line.contains("FAILED")
            important && (includeVerbose || verbose.none { line.contains(it) })
        }
        .distinct()
        .toList()
        .takeLast(if (includeVerbose) 120 else 100)
}
private fun tokensPerSecond(result: GenerationResult?): Double? {
    val tokens = result?.outputTokens ?: return null
    val timeMs = result.generationTimeMs ?: return null
    if (tokens <= 0 || timeMs <= 0) return null
    return tokens.toDouble() / (timeMs.toDouble() / 1000.0)
}
