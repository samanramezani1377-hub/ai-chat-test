package com.samanramezani.aichattest

import com.woogit.aicore.actions.ActionErrorLog
import com.woogit.aicore.actions.ActionErrorLogSink
import com.woogit.aicore.actions.ActionTraceEvent
import com.woogit.aicore.actions.ActionTraceSink
import com.woogit.aicore.observability.CentralObservability
import com.woogit.aicore.observability.ExecutionTraceEvent
import com.woogit.aicore.observability.Severity

/** Bridges the legacy action contracts into the single Central Observability boundary. */
class CentralActionTraceSink(
    private val central: CentralObservability,
    private val local: com.woogit.aicore.actions.InMemoryActionTraceSink = com.woogit.aicore.actions.InMemoryActionTraceSink(),
) : ActionTraceSink {
    override suspend fun record(event: ActionTraceEvent) {
        local.record(event)
        central.trace(
            executionId = event.executionId,
            phase = event.type.toCentralPhase(),
            message = event.message,
        )
    }

    fun snapshot(): List<ActionTraceEvent> = local.snapshot()
}

class CentralActionErrorSink(
    private val central: CentralObservability,
    private val local: com.woogit.aicore.actions.InMemoryActionErrorLogSink = com.woogit.aicore.actions.InMemoryActionErrorLogSink(),
    private val appVersion: String,
) : ActionErrorLogSink {
    override suspend fun record(error: ActionErrorLog) {
        local.record(error)
        central.error(
            appVersion = appVersion,
            component = "Action",
            errorCode = "ACTION_ERROR",
            userMessageFa = error.userMessageFa,
            rawMachineError = error.rawMachineError,
            actionId = error.actionId,
            executionState = "FAILED",
        )
    }

    fun snapshot(): List<ActionErrorLog> = local.snapshot()
}

private fun com.woogit.aicore.actions.ActionTraceType.toCentralPhase(): ExecutionTraceEvent.Phase = when (this) {
    com.woogit.aicore.actions.ActionTraceType.PREPARED -> ExecutionTraceEvent.Phase.PREPARED
    com.woogit.aicore.actions.ActionTraceType.VALIDATED -> ExecutionTraceEvent.Phase.VALIDATED
    com.woogit.aicore.actions.ActionTraceType.APPROVAL_REQUESTED -> ExecutionTraceEvent.Phase.APPROVAL_REQUIRED
    com.woogit.aicore.actions.ActionTraceType.APPROVED -> ExecutionTraceEvent.Phase.APPROVED
    com.woogit.aicore.actions.ActionTraceType.REJECTED -> ExecutionTraceEvent.Phase.REJECTED
    com.woogit.aicore.actions.ActionTraceType.EXECUTION_STARTED -> ExecutionTraceEvent.Phase.EXECUTING
    com.woogit.aicore.actions.ActionTraceType.EXECUTION_COMPLETED -> ExecutionTraceEvent.Phase.EXECUTED
    com.woogit.aicore.actions.ActionTraceType.VERIFICATION_FAILED -> ExecutionTraceEvent.Phase.FAILED
    com.woogit.aicore.actions.ActionTraceType.EXECUTION_FAILED -> ExecutionTraceEvent.Phase.FAILED
    com.woogit.aicore.actions.ActionTraceType.RETRY_REQUESTED -> ExecutionTraceEvent.Phase.RECOVERY_REQUIRED
    com.woogit.aicore.actions.ActionTraceType.RETRY_REJECTED -> ExecutionTraceEvent.Phase.FAILED
    com.woogit.aicore.actions.ActionTraceType.RECOVERY_REQUIRED -> ExecutionTraceEvent.Phase.RECOVERY_REQUIRED
    com.woogit.aicore.actions.ActionTraceType.RECOVERED -> ExecutionTraceEvent.Phase.RECOVERED
}
