package com.woogit.aicore.domain

/**
 * Ready-to-use inference profiles shared by UI and runtime-facing code.
 *
 * Selecting a profile is only a settings operation; it does not execute
 * generation or actions until the resulting settings are applied by the UI.
 */
enum class InferencePreset(
    val title: String,
    val description: String,
    val settings: InferenceSettings,
) {
    FAST(
        title = "سریع",
        description = "پاسخ سریع‌تر و مصرف کمتر؛ مناسب گفت‌وگوی روزمره و دستگاه‌های ضعیف‌تر.",
        settings = InferenceSettings(temperature = 0.7, maxNewTokens = 256, topK = 32, topP = 0.9, minP = 0.0, repeatPenalty = 1.1, seed = null, stopSequences = emptyList(), contextLength = 2048, recentMessages = 6, maxActionSteps = 2),
    ),
    STANDARD(
        title = "استاندارد",
        description = "تعادل پیشنهادی بین کیفیت، سرعت و مصرف منابع؛ گزینه مناسب برای استفاده معمول.",
        settings = InferenceSettings(temperature = 0.7, maxNewTokens = 512, topK = 40, topP = 0.9, minP = 0.0, repeatPenalty = 1.1, seed = null, stopSequences = emptyList(), contextLength = 8192, recentMessages = 10, maxActionSteps = 4),
    ),
    DEEP(
        title = "عمیق",
        description = "فضای بیشتر برای پاسخ‌های طولانی و چندمرحله‌ای؛ کندتر و پرمصرف‌تر.",
        settings = InferenceSettings(temperature = 0.65, maxNewTokens = 1024, topK = 50, topP = 0.92, minP = 0.0, repeatPenalty = 1.1, seed = null, stopSequences = emptyList(), contextLength = 8192, recentMessages = 20, maxActionSteps = 6),
    ),
}

fun defaultInferenceSettings(): InferenceSettings = InferencePreset.STANDARD.settings

fun InferenceSettings.matchingPreset(): InferencePreset? =
    InferencePreset.values().firstOrNull { it.settings == this }
