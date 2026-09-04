package com.woogit.aicore.domain

enum class ApiProtocol { OPENAI_CHAT, ANTHROPIC_MESSAGES }

data class ApiProviderConfig(
    val providerId: String = "deepseek",
    val baseUrl: String = "https://api.deepseek.com",
    val apiKey: String = "",
    val model: String = "deepseek-v4-flash",
    val protocol: ApiProtocol = ApiProtocol.OPENAI_CHAT,
    val reasoningEffort: String? = "high",
    val thinkingEnabled: Boolean = true,
    val timeoutMs: Long = 120_000L,
) {
    fun normalizedBaseUrl(): String = baseUrl.trim().trimEnd('/')
}

object ApiProviderPresets {
    val deepSeek = ApiProviderConfig()

    val openAi = ApiProviderConfig(
        providerId = "openai",
        baseUrl = "https://api.openai.com/v1",
        model = "gpt-5-mini",
        reasoningEffort = null,
        thinkingEnabled = false,
    )

    fun openAiCompatible(providerId: String, baseUrl: String, model: String, apiKey: String = "") =
        ApiProviderConfig(providerId = providerId, baseUrl = baseUrl, model = model, apiKey = apiKey)

    fun anthropic(providerId: String, baseUrl: String, model: String, apiKey: String = "") =
        ApiProviderConfig(providerId = providerId, baseUrl = baseUrl, model = model, apiKey = apiKey, protocol = ApiProtocol.ANTHROPIC_MESSAGES, thinkingEnabled = false, reasoningEffort = null)
}
