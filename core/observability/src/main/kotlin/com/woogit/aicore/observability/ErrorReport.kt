package com.woogit.aicore.observability

import java.time.Instant
import java.util.UUID

data class ErrorReport(
    val reportId: String = UUID.randomUUID().toString(),
    val appVersion: String,
    val timestamp: Instant = Instant.now(),
    val severity: Severity,
    val component: String,
    val errorCode: String,
    val eventId: String,
    val taskId: String? = null,
    val actionId: String? = null,
    val userMessageFa: String,
    val rawMachineError: String,
    val trace: String? = null,
    val relatedLogs: List<String> = emptyList(),
    val executionState: String? = null,
    val recoveryState: String? = null,
    val verificationState: String? = null,
    val runtimeInfo: String? = null,
    val modelInfo: String? = null,
    val redactionStatus: RedactionStatus
)

enum class Severity { INFO, WARNING, ERROR, FATAL }
enum class RedactionStatus { NOT_PROCESSED, REDACTED }

interface ErrorReporter {
    fun record(report: ErrorReport)
    fun recent(limit: Int? = null): List<ErrorReport>
}

class InMemoryErrorReporter : ErrorReporter {
    private val reports = java.util.concurrent.ConcurrentLinkedDeque<ErrorReport>()

    override fun record(report: ErrorReport) {
        reports.addLast(report)
    }

    override fun recent(limit: Int?): List<ErrorReport> {
        val values = reports.toList().asReversed()
        return if (limit == null) values else values.take(limit.coerceAtLeast(0))
    }
}
