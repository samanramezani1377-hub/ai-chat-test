package com.woogit.aicore.runtime.android

import com.woogit.aicore.domain.ModelError
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertTrue

class RuntimeErrorMapperTest {
    @Test
    fun unsupportedOpenClGpuIsReportedWithoutSuggestingCpuFallback() {
        val error = RuntimeErrorMapper.nativeLoadFailure(5)

        assertIs<ModelError.RuntimeUnavailable>(error)
        assertTrue(error.message.contains("No supported OpenCL GPU was registered"))
        assertTrue(error.message.contains("CPU fallback is disabled"))
        assertTrue(error.message.contains("vendor OpenCL driver may be unavailable"))
    }

    @Test
    fun failedGpuResidencyIsReportedAsRuntimeUnavailable() {
        val error = RuntimeErrorMapper.nativeLoadFailure(6)

        assertIs<ModelError.RuntimeUnavailable>(error)
        assertTrue(error.message.contains("fully resident on the OpenCL GPU"))
        assertTrue(error.message.contains("CPU weight fallback is disabled"))
    }

    @Test
    fun unknownNativeFailureRemainsAUsefulLoadError() {
        val error = RuntimeErrorMapper.nativeLoadFailure(1)

        assertIs<ModelError.LoadFailed>(error)
        assertEquals("llama.cpp failed to load the model (code=1)", error.message)
    }
}
