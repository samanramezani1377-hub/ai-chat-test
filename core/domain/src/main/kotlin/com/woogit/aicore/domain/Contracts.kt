package com.woogit.aicore.domain

interface ModelRuntime {
    suspend fun load(model: ModelDescriptor)
    suspend fun unload()
    suspend fun generate(request: GenerationRequest, onToken: suspend (String) -> Unit): GenerationResult
    suspend fun stopGeneration()
    fun runtimeInfo(): RuntimeInfo
}

data class ModelDescriptor(
    val id: String,
    val path: String,
    val format: String,
    val quantization: String? = null
)

data class GenerationRequest(
    val messages: List<ChatMessage>,
    val settings: InferenceSettings
)

data class GenerationResult(
    val inputTokens: Long?,
    val outputTokens: Long?,
    val firstTokenTimeMs: Long?,
    val generationTimeMs: Long?,
    val stopped: Boolean
)

data class ChatMessage(val role: Role, val content: String) {
    enum class Role { SYSTEM, USER, ASSISTANT, TOOL }
}

data class InferenceSettings(
    val temperature: Double,
    val maxNewTokens: Int,
    val topK: Int? = null,
    val topP: Double? = null,
    val minP: Double? = null,
    val repeatPenalty: Double? = null,
    val seed: Long? = null,
    val stopSequences: List<String> = emptyList(),
    val contextLength: Int? = null
)

data class RuntimeInfo(val name: String, val version: String, val backend: String?)

interface Action<in I, out O> {
    val id: String
    val risk: RiskLevel
    suspend fun execute(input: I): O
}

enum class RiskLevel { LOW, NORMAL, SENSITIVE }

interface ActionRegistry {
    fun register(category: String, action: Action<Any, Any>)
    fun find(actionId: String): Action<Any, Any>?
    fun categories(): Set<String>
}

interface CapabilityProvider {
    fun supports(capability: String): Boolean
}

interface Verifier<in O> {
    suspend fun verify(result: O): VerificationResult
}

data class VerificationResult(val success: Boolean, val evidence: String? = null)
