package com.woogit.aicore.runtime.android

import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class RuntimeLoadConfigurationTest {
    @Test
    fun reusesLoadedModelOnlyWhenGpuPreferenceAlsoMatches() {
        assertTrue(
            shouldSkipNativeModelLoad(
                loadedModelPath = "/models/model.gguf",
                requestedModelPath = "/models/model.gguf",
                loadedContextLength = 8192,
                loadedGpuLayersMode = -1,
                requestedGpuLayersMode = -1,
            ),
        )
        assertFalse(
            shouldSkipNativeModelLoad(
                loadedModelPath = "/models/model.gguf",
                requestedModelPath = "/models/model.gguf",
                loadedContextLength = 8192,
                loadedGpuLayersMode = -1,
                requestedGpuLayersMode = 0,
            ),
        )
        assertFalse(
            shouldSkipNativeModelLoad(
                loadedModelPath = "/models/old.gguf",
                requestedModelPath = "/models/model.gguf",
                loadedContextLength = 8192,
                loadedGpuLayersMode = -1,
                requestedGpuLayersMode = -1,
            ),
        )
        assertFalse(
            shouldSkipNativeModelLoad(
                loadedModelPath = "/models/model.gguf",
                requestedModelPath = "/models/model.gguf",
                loadedContextLength = null,
                loadedGpuLayersMode = -1,
                requestedGpuLayersMode = -1,
            ),
        )
    }
}
