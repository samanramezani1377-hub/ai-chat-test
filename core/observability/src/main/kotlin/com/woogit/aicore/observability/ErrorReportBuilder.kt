package com.woogit.aicore.observability

/** Builds a user-facing Persian report with mandatory secret redaction. */
class ErrorReportBuilder(
    private val appVersion: String,
    private val redactor: LogRedactor = DefaultLogRedactor,
) {
    fun build(
        eventId: String,
        component: String,
        errorCode: String,
        userMessageFa: String,
        rawMachineError: String,
        severity: Severity = Severity.ERROR,
        taskId: String? = null,
        actionId: String? = null,
        trace: String? = null,
        relatedLogs: List<String> = emptyList(),
        executionState: String? = null,
        recoveryState: String? = null,
        verificationState: String? = null,
        runtimeInfo: String? = null,
        modelInfo: String? = null,
    ): ErrorReport = ErrorReport(
        appVersion = appVersion,
        severity = severity,
        component = component,
        errorCode = errorCode,
        eventId = eventId,
        taskId = taskId,
        actionId = actionId,
        userMessageFa = userMessageFa,
        rawMachineError = redactor.redact(rawMachineError),
        trace = trace?.let(redactor::redact),
        relatedLogs = relatedLogs.map(redactor::redact),
        executionState = executionState,
        recoveryState = recoveryState,
        verificationState = verificationState,
        runtimeInfo = runtimeInfo,
        modelInfo = modelInfo,
        redactionStatus = RedactionStatus.REDACTED,
    )
}

fun interface LogRedactor {
    fun redact(value: String): String
}

/** Kept for source compatibility; secure callers should use the default redactor. */
object NoOpLogRedactor : LogRedactor {
    override fun redact(value: String): String = value
}
