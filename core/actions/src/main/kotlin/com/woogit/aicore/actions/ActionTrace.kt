package com.woogit.aicore.actions

/** Development trace for the complete action lifecycle. Keep raw technical details here. */
enum class ActionTraceType {
    PREPARED,
    VALIDATED,
    APPROVAL_REQUESTED,
    APPROVED,
    REJECTED,
    EXECUTION_STARTED,
    EXECUTION_COMPLETED,
    VERIFICATION_FAILED,
    EXECUTION_FAILED,
    RETRY_REQUESTED
}

data class ActionTraceEvent(
    val executionId: String,
    val type: ActionTraceType,
    val message: String? = null,
    val timestampMs: Long = System.currentTimeMillis()
)

fun interface ActionTraceSink {
    suspend fun record(event: ActionTraceEvent)
}

class InMemoryActionTraceSink : ActionTraceSink {
    private val events = mutableListOf<ActionTraceEvent>()
    override suspend fun record(event: ActionTraceEvent) {
        synchronized(events) { events += event }
    }
    fun snapshot(): List<ActionTraceEvent> = synchronized(events) { events.toList() }
}
