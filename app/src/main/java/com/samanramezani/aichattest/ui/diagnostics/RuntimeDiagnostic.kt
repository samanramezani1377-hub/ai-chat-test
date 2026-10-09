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
    val resolved = error?.takeIf { it.isNotBlank() }?.let {
        ErrorCenter.resolve(listOf(it, nativeDiagnostics).filterNotNull().joinToString("\n"))
    }
    val nativeLines = nativeDiagnostics.orEmpty().lineSequence()
        .map(String::trim)
        .filter(String::isNotBlank)
        .filterNot { it.contains("OPENGL_ES_TENSOR_RESIDENCY") || it.contains("create_tensor: loading tensor") || it.contains("unused tensor") }
        .toList()
    val diagnosticMarkers = listOf(
        "OPENGL_ES_BACKEND_INIT_FAILED", "OPENGL_ES_SHADER_COMPILE_FAILED",
        "OPENGL_ES_PROGRAM_LINK_FAILED", "OPENGL_ES_PROGRAM_INIT_GL_ERROR",
        "OPENGL_ES_BUFFER_ALLOCATION_FAILED", "OPENGL_ES_GPU_ONLY_UNSUPPORTED_OP",
        "OPENGL_ES_INIT_FAILED", "OPENGL_ES_RESIDENCY_SUMMARY",
        "NATIVE_WEIGHT_RESIDENCY", "CONTEXT_INIT_STARTED", "CONTEXT_INIT_RETURNED_FAILED",
        "CONTEXT_INIT_FAILED", "llama_init_from_model", "MODEL_LOAD_FAILURE",
        "failed to initialize", "OutOfMemory", "exception", "allocation failed"
    )
    val technicalLines = diagnosticMarkers.flatMap { marker ->
        nativeLines.filter { it.contains(marker, ignoreCase = true) }.takeLast(2)
    }.distinct().takeLast(10)
    val fallbackLines = nativeLines.filter {
        it.contains("error", ignoreCase = true) || it.contains("failed", ignoreCase = true) ||
            it.contains("exception", ignoreCase = true) || it.contains("outofmemory", ignoreCase = true)
    }.distinct().takeLast(6)
    val keyLines = (technicalLines.ifEmpty { fallbackLines }).takeLast(10)

    appendLine("AI Chat Test — گزارش عیب‌یابی (حداکثر ۵۰ خط)")
    appendLine("Status: $status | Execution: ${executionId ?: "N/A"}")
    appendLine("Model: ${model?.displayName ?: "N/A"} | Format: ${model?.format ?: "N/A"} | Quantization: ${model?.quantization ?: "N/A"}")
    appendLine("Runtime: ${runtime.name} ${runtime.version} | Backend: ${runtime.backend ?: "N/A"}")
    appendLine("Device: ${gpuDevice?.name ?: "N/A"}")
    appendLine("Threads: ${runtime.threads ?: "N/A"} | GPU Layers: ${runtime.gpuLayers ?: "N/A"} | Context: ${runtime.contextLength ?: "N/A"}")
    appendLine("Load: ${loadTimeMs?.let { "$it ms" } ?: "N/A"} | TTFT: ${generation?.firstTokenTimeMs?.let { "$it ms" } ?: "N/A"}")
    appendLine("Generation: ${perf.generationMs?.let { "$it ms" } ?: generation?.generationTimeMs?.let { "$it ms" } ?: "N/A"}")
    appendLine("Tokens: in=${perf.promptTokens ?: generation?.inputTokens ?: "N/A"} out=${perf.generatedTokens ?: generation?.outputTokens ?: "N/A"} speed=${perf.decodeTokensPerSec?.let { "%.2f tok/s".format(java.util.Locale.US, it) } ?: tokensPerSecond(generation)?.let { "%.2f tok/s".format(java.util.Locale.US, it) } ?: "N/A"}")
    appendLine("Prefill: ${perf.prefillMs?.let { "$it ms" } ?: "N/A"} | Decode: ${perf.decodeMs?.let { "$it ms" } ?: "N/A"}")
    appendLine("GPU tensors: ${weightResidency?.gpuTensors ?: "N/A"} / ${weightResidency?.gpuTensorMiB?.let { "%.1f MiB".format(java.util.Locale.US, it) } ?: "N/A"}")
    appendLine("Host tensors: ${weightResidency?.hostTensorMiB?.let { "%.1f MiB".format(java.util.Locale.US, it) } ?: "N/A"} | CPU tensors: ${weightResidency?.cpuTensorMiB?.let { "%.1f MiB".format(java.util.Locale.US, it) } ?: "N/A"}")
    appendLine("GPU memory: free=${gpuDevice?.memoryFreeMiB?.let { "%.1f MiB".format(java.util.Locale.US, it) } ?: "N/A"} total=${gpuDevice?.memoryTotalMiB?.let { "%.1f MiB".format(java.util.Locale.US, it) } ?: "N/A"}")
    appendLine("KV cache: ${perf.cacheStatus ?: "N/A"} | cached=${perf.cachedTokens ?: "N/A"} reused=${perf.reusedTokens ?: "N/A"} new=${perf.newTokens ?: "N/A"}")
    appendLine("Settings: temp=${settings?.temperature ?: "N/A"} topP=${settings?.topP ?: "N/A"} topK=${settings?.topK ?: "N/A"} minP=${settings?.minP ?: "N/A"}")
    appendLine("Generation config: maxTokens=${settings?.maxNewTokens ?: "N/A"} context=${settings?.contextLength ?: "N/A"} repeatPenalty=${settings?.repeatPenalty ?: "N/A"} seed=${settings?.seed ?: "N/A"}")
    if (resolved != null) {
        appendLine("Error: ${resolved.code} | ${resolved.title}")
        appendLine("Meaning: ${resolved.message}")
        appendLine("Suggested action: ${resolved.action}")
    } else appendLine("Error: N/A")
    val cleanError = (rawError ?: error)?.replace(Regex("\\s+"), " ")?.take(220)
    if (!cleanError.isNullOrBlank()) appendLine("Raw error: $cleanError")
    appendLine("----- Key native diagnostics -----")
    if (keyLines.isEmpty()) appendLine("No specific native failure marker captured.")
    else keyLines.forEach { appendLine(it.take(260)) }
    appendLine("----- Recent lifecycle events -----")
    val events = uniqueRuntimeEvents().takeLast(4)
    if (events.isEmpty()) appendLine("No runtime events captured.")
    else events.forEach { appendLine(it.take(180)) }
}.lineSequence().take(50).joinToString("\n")

private fun RuntimeDiagnostic.failedPhaseLog(): List<String> {
    val lines = nativeDiagnostics.orEmpty().lineSequence().map(String::trim)
        .filter(String::isNotBlank).toList()
    if (lines.isEmpty()) return emptyList()

    val failureIndex = lines.indexOfLast { line ->
        line.contains("CONTEXT_INIT_FAILED", true) ||
            line.contains("CONTEXT_INIT_RETURNED_FAILED", true) ||
            line.contains("MODEL_LOAD_RETURNED_FAILED", true) ||
            line.contains("MODEL_LOAD_FAILURE", true) ||
            line.contains("OPENGL_ES_BACKEND_INIT_FAILED", true) ||
            line.contains("OPENGL_ES_INIT_FAILED", true) ||
            line.contains("SHADER_COMPILE_FAILED", true) ||
            line.contains("PROGRAM_LINK_FAILED", true) ||
            line.contains("BUFFER_ALLOCATION_FAILED", true) ||
            line.contains("OutOfMemory", true) ||
            line.contains("failed to initialize", true)
    }
    if (failureIndex < 0) return emptyList()

    val phaseStartMarkers = listOf(
        "NATIVE_LOAD_STARTED", "MODEL_LOAD_STARTED", "CONTEXT_INIT_STARTED",
        "OPENGL_ES_BACKEND_INITIALIZATION_STARTED", "ACTIVATION_LLAMA_BACKEND_INIT_STARTED"
    )
    val startIndex = (0..failureIndex).lastOrNull { index ->
        phaseStartMarkers.any { lines[index].contains(it, ignoreCase = true) }
    } ?: maxOf(0, failureIndex - 49)
    // The UI must never dump a 999-line native trace by default. Keep the phase
    // boundary, failure markers and the most recent distinct diagnostic evidence.
    val phaseLines = lines.subList(startIndex, lines.size)
    val keyLines = phaseLines.filter { line ->
        line.contains("NATIVE_FATAL_", true) ||
            line.contains("CONTEXT_INIT", true) ||
            line.contains("MODEL_LOAD_FAILURE", true) ||
            line.contains("OPENGL_ES_") && (
                line.contains("FAILED", true) || line.contains("ERROR", true) ||
                    line.contains("UNSUPPORTED", true)
            ) ||
            line.contains("SIGABRT", true) ||
            line.contains("OutOfMemory", true) ||
            line.contains("failed to initialize", true) ||
            line.contains("llama_graph_n_input_tensors", true) ||
            line.contains(" is used by node ", true) ||
            line.contains("NATIVE_CHECKPOINT", true)
    }.distinct()
    val tail = phaseLines.asReversed().distinct().take(12).asReversed()
    return (keyLines + tail).distinct().takeLast(30)
}

internal fun RuntimeDiagnostic.logLines(): List<String> {
    val all = nativeDiagnostics.orEmpty().lineSequence()
        .map(String::trim).filter(String::isNotBlank).toList()
    if (all.isEmpty()) return emptyList()

    val failed = error != null || status.equals("FAILED", true) || status.equals("ERROR", true)
    if (failed) {
        val concise = failedPhaseLog()
        if (concise.isNotEmpty()) return concise
        return importantNativeEvents(includeVerbose = false).takeLast(30)
    }

    // Completed phases are summarized. A hard cap also protects against malformed
    // traces that lack phase markers or contain repetitive backend output.
    return compactCompletedPhases(all).distinct().takeLast(30)
}

private data class NativePhase(
    val name: String,
    val startIndex: Int,
    val startLine: String,
)

private fun compactCompletedPhases(lines: List<String>): List<String> {
    val startTokens = listOf(
        "NATIVE_LOAD_STARTED" to "بارگذاری Native",
        "MODEL_LOAD_STARTED" to "بارگذاری وزن‌های مدل",
        "CONTEXT_INIT_STARTED" to "ساخت Context",
        "OPENGL_ES_BACKEND_INITIALIZATION_STARTED" to "راه‌اندازی Backend گرافیکی",
        "ACTIVATION_LLAMA_BACKEND_INIT_STARTED" to "راه‌اندازی Backend مدل",
        "GENERATION_STARTED" to "تولید پاسخ",
        "NATIVE_KV_CACHE_STARTED" to "آماده‌سازی KV Cache",
        "TOKENIZATION_STARTED" to "توکن‌سازی ورودی",
        "PREFILL_STARTED" to "پردازش ورودی (Prefill)",
        "DECODE_STARTED" to "تولید توکن (Decode)",
    )
    val failureTokens = listOf(
        "FAILED", "ERROR", "EXCEPTION", "OUTOFMEMORY", "ALLOCATION_FAILED",
        "UNSUPPORTED_OP", "CONTEXT_INIT_RETURNED_FAILED", "MODEL_LOAD_FAILURE",
        "failed to initialize", "allocation failed"
    )
    val successTokens = listOf(
        "RETURNED_SUCCESS", "COMPLETED", "SUCCESS", "MODEL_READY",
        "TARGET_RUNTIME_READY", "BACKEND_INITIALIZED", "INITIALIZATION_SUCCEEDED",
        "GENERATION_FINISHED", "GENERATION_COMPLETED", "PREFILL_COMPLETED",
        "DECODE_COMPLETED", "KV_CACHE_READY", "TOKENIZATION_COMPLETED"
    )
    val excludedVerbose = listOf(
        "OPENGL_ES_TENSOR_RESIDENCY", "create_tensor: loading tensor", "unused tensor"
    )
    val starts = lines.mapIndexedNotNull { index, line ->
        startTokens.firstOrNull { line.contains(it.first, ignoreCase = true) }
            ?.let { NativePhase(it.second, index, line) }
            ?: Regex("""\b([A-Z][A-Z0-9_]*(?:STARTED|BEGIN))\b""")
                .find(line)?.groupValues?.get(1)
                ?.takeIf { token ->
                    listOf("MODEL", "CONTEXT", "BACKEND", "GENERATION", "TOKEN", "PREFILL",
                        "DECODE", "CACHE", "RUNTIME", "LOAD", "INITIALIZATION", "ACTIVATION")
                        .any { token.contains(it) }
                }
                ?.let { token ->
                    val label = token.removeSuffix("_STARTED").removeSuffix("_BEGIN")
                        .lowercase(java.util.Locale.ROOT).replace("_", " ")
                    NativePhase(label, index, line)
                }
    }.distinctBy { it.startIndex }
    if (starts.isEmpty()) {
        // No phase boundaries were recorded. Keep useful milestone/error lines,
        // while dropping repetitive per-tensor chatter.
        return lines.filterNot { line -> excludedVerbose.any { line.contains(it, true) } }
            .filter { line ->
                listOf("READY", "SUCCESS", "COMPLETED", "FAILED", "ERROR", "MODEL_LOAD",
                    "CONTEXT_INIT", "OPENGL_ES_BACKEND", "OPENGL_ES_DEVICE",
                    "RESIDENCY_SUMMARY", "NATIVE_WEIGHT_RESIDENCY", "GENERATION",
                    "KV_CACHE", "PREFILL", "DECODE", "TOKENIZATION")
                    .any { line.contains(it, true) }
            }.distinct().takeLast(50)
    }

    val output = mutableListOf<String>()
    var cursor = 0
    for (phaseIndex in starts.indices) {
        val phase = starts[phaseIndex]
        val nextStart = starts.getOrNull(phaseIndex + 1)?.startIndex ?: lines.size
        if (cursor < phase.startIndex) {
            output += summarizeLooseLines(lines.subList(cursor, phase.startIndex), excludedVerbose)
        }
        val endIndex = nextStart
        val chunk = lines.subList(phase.startIndex, endIndex)
        val failureIndex = chunk.indexOfFirst { line ->
            failureTokens.any { token -> line.contains(token, ignoreCase = true) }
        }
        val completed = chunk.any { line ->
            successTokens.any { token -> line.contains(token, ignoreCase = true) }
        } || phaseIndex < starts.lastIndex // the next phase starting implies this phase returned control
        if (failureIndex >= 0 || !completed) {
            // An unfinished phase is the evidence: never truncate or summarize it.
            output += "⛔ مرحله «${phase.name}» تکمیل نشد؛ لاگ کامل مرحله:"
            output += chunk
        } else {
            val duration = chunk.firstNotNullOfOrNull { line ->
                Regex("""(?:duration|elapsed|time|took)=([0-9]+(?:\.[0-9]+)?\s*(?:ms|s))""", RegexOption.IGNORE_CASE)
                    .find(line)?.groupValues?.get(1)
            }
            val detail = chunk.asSequence()
                .filterNot { line -> excludedVerbose.any { line.contains(it, true) } }
                .filter { line ->
                    successTokens.any { line.contains(it, true) } ||
                        line.contains("RESIDENCY_SUMMARY", true) ||
                        line.contains("NATIVE_WEIGHT_RESIDENCY", true) ||
                        line.contains("OPENGL_ES_DEVICE", true) ||
                        line.contains("tokens=", true) || line.contains("MiB", true) ||
                        line.contains("ms", true)
                }
                .distinct().toList().takeLast(3)
            output += "✓ مرحله «${phase.name}» تکمیل شد${duration?.let { "؛ زمان: $it" } ?: ""}."
            detail.forEach { output += "  • ${it.take(240)}" }
        }
        cursor = endIndex
    }
    if (cursor < lines.size) output += summarizeLooseLines(lines.subList(cursor, lines.size), excludedVerbose)
    return output.distinct().takeIf { it.isNotEmpty() } ?: lines.takeLast(50)
}

private fun summarizeLooseLines(lines: List<String>, excludedVerbose: List<String>): List<String> =
    lines.asSequence()
        .filterNot { line -> excludedVerbose.any { line.contains(it, true) } }
        .filter { line ->
            listOf("FAILED", "ERROR", "EXCEPTION", "READY", "SUCCESS", "COMPLETED",
                "MODEL_LOAD", "CONTEXT_INIT", "OPENGL_ES_BACKEND", "OPENGL_ES_DEVICE",
                "RESIDENCY_SUMMARY", "NATIVE_WEIGHT_RESIDENCY", "GENERATION",
                "KV_CACHE", "PREFILL", "DECODE", "TOKENIZATION")
                .any { line.contains(it, true) }
        }
        .distinct()
        .toList()

internal fun RuntimeDiagnostic.errorReport(): String = report()

/** Full raw trace is intentionally available only through the explicit raw-log copy action. */
internal fun RuntimeDiagnostic.fullNativeLogLines(): List<String> =
    nativeDiagnostics.orEmpty().lineSequence()
        .map(String::trim)
        .filter(String::isNotBlank)
        .takeLast(1000)
        .toList()

private fun RuntimeDiagnostic.uniqueRuntimeEvents(): List<String> {
    data class EventLine(val timestamp: Long, val text: String, val key: String)
    val runtime = runtimeTrace.asSequence().map {
        EventLine(it.timestampMs, "${it.timestampMs} | ${it.type} | ${it.message ?: ""}".trimEnd(), "${it.type}|${it.message ?: ""}")
    }
    val actions = actionTrace.asSequence().map {
        EventLine(it.timestampMs, "${it.timestampMs} | ${it.type} | ${it.message ?: ""}".trimEnd(), "${it.type}|${it.message ?: ""}")
    }
    val events = (runtime + actions)
        .sortedBy { it.timestamp }
        .distinctBy { it.key }
        .toList()

    val limit = RECENT_RUNTIME_EVENTS + RECENT_ACTION_EVENTS
    val recent = if (events.size <= limit) events else events.subList(events.size - limit, events.size)
    return recent.map { it.text }
}

private fun RuntimeDiagnostic.importantNativeEvents(includeVerbose: Boolean): List<String> {
    val verbose = setOf("create_tensor: loading tensor", "unused tensor")
    val limit = if (includeVerbose) 120 else 100
    val lines = nativeDiagnostics.orEmpty().lineSequence()
        .filter { line ->
            val important = line.contains("OPENGL_ES_") || line.contains("MODEL_LOAD") ||
                line.contains("NATIVE_WEIGHT_RESIDENCY") || line.contains("NATIVE_KV_CACHE") ||
                line.contains("NATIVE_PERF") || line.contains("SPECULATIVE_") ||
                line.contains("ERROR") || line.contains("error") || line.contains("failed") || line.contains("FAILED")
            important && (includeVerbose || verbose.none { line.contains(it) })
        }
        .distinct()
        .toList()

    if (lines.size <= limit) return lines
    return lines.subList(lines.size - limit, lines.size)
}

private fun tokensPerSecond(result: GenerationResult?): Double? {
    val tokens = result?.outputTokens ?: return null
    val timeMs = result.generationTimeMs ?: return null
    if (tokens <= 0 || timeMs <= 0) return null
    return tokens.toDouble() / (timeMs.toDouble() / 1000.0)
}
