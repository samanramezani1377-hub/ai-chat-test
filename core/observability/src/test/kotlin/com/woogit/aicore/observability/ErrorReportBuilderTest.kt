package com.woogit.aicore.observability

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ErrorReportBuilderTest {
    @Test
    fun builderKeepsUserAndMachineErrorsTogether() {
        val report = ErrorReportBuilder("0.1.0").build(
            eventId = "event-1",
            component = "runtime",
            errorCode = "RUNTIME_FAILED",
            userMessageFa = "اجرای مدل ناموفق بود.",
            rawMachineError = "native runtime error"
        )

        assertEquals("اجرای مدل ناموفق بود.", report.userMessageFa)
        assertEquals("native runtime error", report.rawMachineError)
        assertTrue(report.redactionStatus == RedactionStatus.REDACTED)
    }
}
