package com.woogit.aicore.actions

interface ActionErrorLogRepository : ActionErrorLogSink {
    suspend fun all(): List<ActionErrorLog>
    suspend fun forExecution(executionId: String): List<ActionErrorLog>
}

class InMemoryActionErrorLogRepository : ActionErrorLogRepository {
    private val errors = mutableListOf<ActionErrorLog>()

    override suspend fun record(error: ActionErrorLog) {
        synchronized(errors) { errors += error }
    }

    override suspend fun all(): List<ActionErrorLog> = synchronized(errors) { errors.toList() }

    override suspend fun forExecution(executionId: String): List<ActionErrorLog> =
        synchronized(errors) { errors.filter { it.executionId == executionId } }
}
