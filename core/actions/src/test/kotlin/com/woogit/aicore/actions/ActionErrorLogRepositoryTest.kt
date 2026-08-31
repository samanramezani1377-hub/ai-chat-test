package com.woogit.aicore.actions

import kotlin.test.Test
import kotlin.test.assertEquals

class ActionErrorLogRepositoryTest {
    @Test
    fun errorsCanBeQueriedGloballyAndByExecution() = kotlinx.coroutines.test.runTest {
        val repository = InMemoryActionErrorLogRepository()
        repository.record(ActionErrorLog("exec-1", "write", "خطای اول", "raw-1"))
        repository.record(ActionErrorLog("exec-2", "delete", "خطای دوم", "raw-2"))
        repository.record(ActionErrorLog("exec-1", "write", "خطای سوم", "raw-3"))

        assertEquals(3, repository.all().size)
        assertEquals(2, repository.forExecution("exec-1").size)
        assertEquals("raw-2", repository.forExecution("exec-2").single().rawMachineError)
    }
}
