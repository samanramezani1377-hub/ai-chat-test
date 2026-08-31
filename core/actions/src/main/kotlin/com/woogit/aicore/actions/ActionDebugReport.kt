package com.woogit.aicore.actions

/**
 * Development-only aggregate view. UI may render this or hide it without changing the core.
 * The raw machine errors remain available for copying to an agent.
 */
data class ActionDebugReport(
    val executionId: String,
    val trace: List<ActionTraceEvent>,
    val errors: List<ActionErrorLog>
) {
    fun asCopyText(): String = buildString {
        appendLine("executionId=$executionId")
        appendLine("TRACE")
        trace.forEach { appendLine("${it.timestampMs} | ${it.type} | ${it.message.orEmpty()}") }
        appendLine("ERRORS")
        errors.forEach {
            appendLine("${it.timestampMs} | ${it.actionId.orEmpty()} | ${it.userMessageFa} | ${it.rawMachineError}")
        }
    }
}

class ActionDebugReportService(
    private val traceSink: InMemoryActionTraceSink,
    private val errorRepository: ActionErrorLogRepository
) {
    suspend fun report(executionId: String): ActionDebugReport = ActionDebugReport(
        executionId = executionId,
        trace = traceSink.snapshot().filter { it.executionId == executionId },
        errors = errorRepository.forExecution(executionId)
    )
}
