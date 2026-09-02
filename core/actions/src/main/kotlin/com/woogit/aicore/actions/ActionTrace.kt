package com.woogit.aicore.actions

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/** Single lifecycle trace contract for action execution. */
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
    RETRY_REQUESTED,
    RETRY_REJECTED,
    RECOVERY_REQUIRED,
    RECOVERED
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

object ActionTraceStore {
    private val state = MutableStateFlow<List<ActionTraceEvent>>(emptyList())
    val events: StateFlow<List<ActionTraceEvent>> = state.asStateFlow()

    fun record(event: ActionTraceEvent) {
        state.value = state.value + event
    }
}

class InMemoryActionTraceSink : ActionTraceSink {
    private val events = mutableListOf<ActionTraceEvent>()
    override suspend fun record(event: ActionTraceEvent) {
        synchronized(events) { events += event }
        ActionTraceStore.record(event)
    }
    fun snapshot(): List<ActionTraceEvent> = synchronized(events) { events.toList() }
}
