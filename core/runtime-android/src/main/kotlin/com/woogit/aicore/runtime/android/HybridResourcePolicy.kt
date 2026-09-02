package com.woogit.aicore.runtime.android

/** Conservative default resource policy for public Android devices. */
internal object HybridResourcePolicy {
    data class Plan(
        val gpuLayers: Int,
        val cpuThreads: Int,
    )

    fun choose(blockCount: Long, hasGpu: Boolean, availableProcessors: Int, bigCoreCount: Int): Plan {
        val cpuThreads = when {
            bigCoreCount > 0 -> bigCoreCount.coerceIn(1, 4)
            availableProcessors > 0 -> (availableProcessors / 2).coerceIn(1, 4)
            else -> 2
        }
        if (!hasGpu || blockCount <= 0L) return Plan(0, cpuThreads)

        // Keep a meaningful CPU share and avoid occupying the whole GPU with a small model.
        val conservative = maxOf(1L, blockCount / 3L)
        val capped = minOf(conservative, 12L, blockCount)
        return Plan(capped.toInt(), cpuThreads)
    }
}
