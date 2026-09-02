package com.woogit.aicore.observability

import java.util.UUID
import java.util.concurrent.ConcurrentLinkedDeque

/**
 * The single observability boundary for the application.
 * Components emit events/errors here; presentation layers only query/subscribe.
 */
class CentralObservability(
    private val errorStore: ErrorReportStore = InMemoryErrorReportStore(),
    private val traceStore: ExecutionTraceStore = InMemoryExecutionTraceStore(),
    private val redactor: LogRedactor = DefaultLogRedactor,
) {
    suspend fun error(
        appVersion: String,
        component: String,
        errorCode: String,
        userMessageFa: String,
        rawMachineError: String,
        severity: Severity = Severity.ERROR,
        eventId: String = UUID.randomUUID().toString(),
        taskId: String? = null,
        actionId: String? = null,
        trace: String? = null,
        relatedLogs: List<String> = emptyList(),
        executionState: String? = null,
        recoveryState: String? = null,
        verificationState: String? = null,
        runtimeInfo: String? = null,
        modelInfo: String? = null,
    ): ErrorReport {
        val report = ErrorReportBuilder(appVersion, redactor).build(
            eventId = eventId,
            component = component,
            errorCode = errorCode,
            userMessageFa = userMessageFa,
            rawMachineError = rawMachineError,
            severity = severity,
            taskId = taskId,
            actionId = actionId,
            trace = trace,
            relatedLogs = relatedLogs,
            executionState = executionState,
            recoveryState = recoveryState,
            verificationState = verificationState,
            runtimeInfo = runtimeInfo,
            modelInfo = modelInfo,
        )
        errorStore.append(report)
        return report
    }

    suspend fun trace(event: ExecutionTraceEvent) {
        traceStore.append(event.copy(message = event.message?.let(redactor::redact)))
    }

    suspend fun trace(
        executionId: String,
        phase: ExecutionTraceEvent.Phase,
        taskId: String? = null,
        actionId: String? = null,
        message: String? = null,
    ): ExecutionTraceEvent {
        val event = ExecutionTraceEvent(
            executionId = executionId,
            taskId = taskId,
            actionId = actionId,
            phase = phase,
            message = message,
        )
        trace(event)
        return event
    }

    suspend fun errors(limit: Int? = null): List<ErrorReport> = errorStore.recent(limit)
    suspend fun traces(executionId: String): List<ExecutionTraceEvent> = traceStore.forExecution(executionId)

    suspend fun reportForAgent(reportId: String): String? =
        errorStore.find(reportId)?.let(ErrorReportFormatter::forAgent)

    suspend fun copyableReport(reportId: String): String? = reportForAgent(reportId)
}

interface ErrorReportStore {
    suspend fun append(report: ErrorReport)
    suspend fun recent(limit: Int? = null): List<ErrorReport>
    suspend fun find(reportId: String): ErrorReport?
}

class InMemoryErrorReportStore : ErrorReportStore {
    private val reports = ConcurrentLinkedDeque<ErrorReport>()

    override suspend fun append(report: ErrorReport) { reports.addLast(report) }

    override suspend fun recent(limit: Int?): List<ErrorReport> {
        val values = reports.toList().asReversed()
        return if (limit == null) values else values.take(limit.coerceAtLeast(0))
    }

    override suspend fun find(reportId: String): ErrorReport? = reports.firstOrNull { it.reportId == reportId }
}

object ErrorReportFormatter {
    fun forAgent(report: ErrorReport): String = buildString {
        appendLine("AI Chat Test Error Report")
        appendLine("Report ID: ${report.reportId}")
        appendLine("App Version: ${report.appVersion}")
        appendLine("Timestamp: ${report.timestamp}")
        appendLine("Event ID: ${report.eventId}")
        appendLine("Task ID: ${report.taskId ?: "-"}")
        appendLine("Action ID: ${report.actionId ?: "-"}")
        appendLine("Component: ${report.component}")
        appendLine("Error Code: ${report.errorCode}")
        appendLine("Severity: ${report.severity}")
        appendLine("User-facing Error: ${report.userMessageFa}")
        appendLine("Raw Machine Error: ${report.rawMachineError}")
        appendLine("Execution State: ${report.executionState ?: "-"}")
        appendLine("Recovery State: ${report.recoveryState ?: "-"}")
        appendLine("Verification State: ${report.verificationState ?: "-"}")
        appendLine("Runtime: ${report.runtimeInfo ?: "-"}")
        appendLine("Model: ${report.modelInfo ?: "-"}")
        appendLine("Redaction: ${report.redactionStatus}")
        if (!report.trace.isNullOrBlank()) appendLine("Trace:\n${report.trace}")
        if (report.relatedLogs.isNotEmpty()) appendLine("Related Logs:\n${report.relatedLogs.joinToString("\n")}")
    }
}

object DefaultLogRedactor : LogRedactor {
    private val patterns = listOf(
        Regex("(?i)(bearer\\s+)[A-Za-z0-9._~+/-]+") to "$1[REDACTED]",
        Regex("(?i)((?:api[-_ ]?key|access[-_ ]?token|refresh[-_ ]?token|authorization|password|secret)\\s*[:=]\\s*)[^\\s,;]+") to "$1[REDACTED]",
        Regex("(?i)(x-api-key\\s*[:=]\\s*)[^\\s,;]+") to "$1[REDACTED]",
    )

    override fun redact(value: String): String = patterns.fold(value) { text, (pattern, replacement) ->
        pattern.replace(text, replacement)
    }
}
