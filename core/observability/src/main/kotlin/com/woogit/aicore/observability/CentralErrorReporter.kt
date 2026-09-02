package com.woogit.aicore.observability

/** Compatibility contract; CentralObservability is the only production implementation. */
interface CentralErrorReporter {
    suspend fun report(error: ErrorReport)
}

/** In-memory compatibility adapter for tests and legacy callers. */
@Deprecated("Use CentralObservability directly")
class InMemoryCentralErrorReporter : CentralErrorReporter {
    private val delegate = InMemoryErrorReportStore()

    override suspend fun report(error: ErrorReport) {
        delegate.append(error)
    }

    suspend fun all(): List<ErrorReport> = delegate.recent()
}
