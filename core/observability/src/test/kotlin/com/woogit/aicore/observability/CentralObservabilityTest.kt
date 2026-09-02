package com.woogit.aicore.observability

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class CentralObservabilityTest {
    @Test
    fun recordsRedactedErrorAndTraceAndCanFormatAgentReport() = kotlinx.coroutines.test.runTest {
        val central = CentralObservability()
        val trace = central.trace("exec-1", ExecutionTraceEvent.Phase.EXECUTING, taskId = "task-1", actionId = "file.write", message = "Authorization: Bearer secret-token")
        val report = central.error(
            appVersion = "test",
            component = "Action",
            errorCode = "WRITE_FAILED",
            userMessageFa = "نوشتن فایل انجام نشد.",
            rawMachineError = "api_key=super-secret",
            taskId = "task-1",
            actionId = "file.write",
            trace = "Bearer another-secret",
        )

        assertEquals("exec-1", trace.executionId)
        assertEquals(1, central.traces("exec-1").size)
        assertEquals(1, central.errors().size)
        assertTrue(report.rawMachineError.contains("[REDACTED]"))
        assertTrue(report.trace!!.contains("[REDACTED]"))
        assertTrue(central.copyableReport(report.reportId)!!.contains("Event ID: ${report.eventId}"))
    }
}
