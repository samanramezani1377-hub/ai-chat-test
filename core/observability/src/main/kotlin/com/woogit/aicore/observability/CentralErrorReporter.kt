package com.woogit.aicore.observability

interface CentralErrorReporter {
    suspend fun report(error: ErrorReport)
}

class InMemoryCentralErrorReporter : CentralErrorReporter {
    private val reports = mutableListOf<ErrorReport>()

    override suspend fun report(error: ErrorReport) {
        synchronized(reports) {
            reports += error
        }
    }

    fun all(): List<ErrorReport> = synchronized(reports) {
        reports.toList()
    }
}
