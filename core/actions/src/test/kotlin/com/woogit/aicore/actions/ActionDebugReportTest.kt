package com.woogit.aicore.actions

import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals

class ActionDebugReportTest {
    @Test
    fun reportContainsTraceAndRawErrorsAndProducesCopyableText() = kotlinx.coroutines.test.runTest {
        val trace = InMemoryActionTraceSink()
        val errors = InMemoryActionErrorLogRepository()
        val executionId = "exec-42"

        trace.record(ActionTraceEvent(executionId, ActionTraceType.EXECUTION_STARTED, "write"))
        trace.record(ActionTraceEvent(executionId, ActionTraceType.EXECUTION_FAILED, "HTTP 500"))
        errors.record(ActionErrorLog(executionId, "write", "اجرای عملیات با خطا مواجه شد.", "HTTP 500 / rest_server_error"))
        errors.record(ActionErrorLog("other", "read", "خطای دیگر", "HTTP 404"))

        val report = ActionDebugReportService(trace, errors).report(executionId)
        val copied = report.asCopyText()

        assertEquals(executionId, report.executionId)
        assertEquals(2, report.trace.size)
        assertEquals(1, report.errors.size)
        assertContains(copied, "exec-42")
        assertContains(copied, "EXECUTION_FAILED")
        assertContains(copied, "HTTP 500 / rest_server_error")
        assertContains(copied, "اجرای عملیات با خطا مواجه شد.")
    }
}
