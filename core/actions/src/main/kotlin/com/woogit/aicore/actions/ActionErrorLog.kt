package com.woogit.aicore.actions

/** Central development error record. The raw machine error is preserved separately from the user message. */
data class ActionErrorLog(
    val executionId: String?,
    val actionId: String?,
    val userMessageFa: String,
    val rawMachineError: String,
    val timestampMs: Long = System.currentTimeMillis()
)

fun interface ActionErrorLogSink {
    suspend fun record(error: ActionErrorLog)
}

class InMemoryActionErrorLogSink : ActionErrorLogSink {
    private val errors = mutableListOf<ActionErrorLog>()
    override suspend fun record(error: ActionErrorLog) {
        synchronized(errors) { errors += error }
    }
    fun snapshot(): List<ActionErrorLog> = synchronized(errors) { errors.toList() }
}
