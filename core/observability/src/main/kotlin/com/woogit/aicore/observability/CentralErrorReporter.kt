package com.woogit.aicore.observability

interface CentralErrorReporter {
    suspend fun report(error: ErrorReport)
}

class InMemoryCentralErrorReporter : CentralErrorReporter {
    private val reports = mutableListOf<ErrorReport>()

    @Synchronized
    override suspend fun report(error: ErrorReport) {
        reports += error
    }

    @Synchronized
    fun all(): List<ErrorReport> = reports.toList()
}
