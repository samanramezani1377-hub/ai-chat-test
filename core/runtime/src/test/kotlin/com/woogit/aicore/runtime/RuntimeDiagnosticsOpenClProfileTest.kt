package com.woogit.aicore.runtime

import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class RuntimeDiagnosticsOpenClProfileTest {
    @Test
    fun `writes real-shaped profiling file and reads non-N-A profile data`() {
        val dir = Files.createTempDirectory("ai-chat-opencl-profile-test-").toFile()
        try {
            val profileFile = dir.resolve("cl_profiling.csv")
            val transferFile = dir.resolve("cl_transfer_profile.txt")

            profileFile.writeText(
                """
                op name, kernel name, queue (ms), submit (ms), exec duration (ms), complete (ms), total (ms), global size, local size, output size
                MUL_MAT,gemv_q6_k_f32_flat,0.120,0.030,4.500,0.080,4.730,1x1x1,1x1x1,1x1x1x1
                FLASH_ATTN,flash_attn_f32,0.100,0.020,2.250,0.050,2.420,1x1x1,1x1x1,1x1x1x1
                ROPE,rope_f32,0.050,0.010,0.750,0.020,0.830,1x1x1,1x1x1,1x1x1x1
                """.trimIndent() + "\n",
            )
            transferFile.writeText("memory_transfer_ms=1.75\n")

            val profile = RuntimeDiagnosticsStore.readOpenClProfileFile(profileFile, transferFile)

            assertNotNull(profile)
            assertEquals(3, profile.kernelCount)
            assertEquals(7.5, profile.totalKernelMs, 1e-9)
            assertEquals(4.5, profile.q6KMulMatMs, 1e-9)
            assertEquals(2.25, profile.attentionMs, 1e-9)
            assertEquals(0.75, profile.ropeMs, 1e-9)
            assertEquals(0.27, profile.kernelLaunchMs, 1e-9)
            assertEquals(0.06, profile.kernelSubmitMs, 1e-9)
            assertEquals(0.15, profile.syncMs, 1e-9)
            assertEquals(1.75, profile.memoryTransferMs, 1e-9)
            assertTrue(profile.topKernels.isNotEmpty())
            assertEquals("gemv_q6_k_f32_flat", profile.topKernels.first().kernelName)
            assertTrue(profile.reportLines().none { it.endsWith("N/A") })
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun `ignores malformed rows but returns null when no valid kernel row exists`() {
        val dir = Files.createTempDirectory("ai-chat-opencl-profile-invalid-").toFile()
        try {
            val profileFile = dir.resolve("cl_profiling.csv")
            profileFile.writeText(
                """
                op name, kernel name, queue (ms), submit (ms), exec duration (ms), complete (ms), total (ms), global size, local size, output size
                malformed,row
                also,not,a,number,here
                """.trimIndent() + "\n",
            )

            assertTrue(RuntimeDiagnosticsStore.readOpenClProfileFile(profileFile) == null)
        } finally {
            dir.deleteRecursively()
        }
    }
}
