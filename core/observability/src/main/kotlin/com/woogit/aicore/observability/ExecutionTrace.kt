package com.woogit.aicore.observability

import java.time.Instant
import java.util.UUID

/** Immutable event emitted while an operation moves through its lifecycle. */
data class ExecutionTraceEvent(
    val eventId: String = UUID.randomUUID().toString(),
    val executionId: String,
    val taskId: String? = null,
    val actionId: String? = null,
    val phase: Phase,
    val message: String? = null,
    val timestamp: Instant = Instant.now()
) {
    enum class Phase {
        PREPARED,
        VALIDATED,
        APPROVAL_REQUIRED,
        APPROVED,
        REJECTED,
        EXECUTING,
        EXECUTED,
        VERIFYING,
        VERIFIED,
        FAILED,
        RECOVERY_REQUIRED,
        RECOVERED
    }
}

interface ExecutionTraceStore {
    suspend fun append(event: ExecutionTraceEvent)
    suspend fun forExecution(executionId: String): List<ExecutionTraceEvent>
}

class InMemoryExecutionTraceStore : ExecutionTraceStore {
    private val events = mutableListOf<ExecutionTraceEvent>()

    override suspend fun append(event: ExecutionTraceEvent) {
        synchronized(events) {
            events += event
        }
    }

    override suspend fun forExecution(executionId: String): List<ExecutionTraceEvent> = synchronized(events) {
        events.filter { it.executionId == executionId }
    }
}
