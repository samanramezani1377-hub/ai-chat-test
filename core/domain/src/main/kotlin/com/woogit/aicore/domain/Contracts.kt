package com.woogit.aicore.domain

interface ModelRuntime {
    suspend fun load(model: ModelDescriptor)
    suspend fun unload()
    suspend fun generate(request: GenerationRequest, onToken: suspend (String) -> Unit): GenerationResult
    suspend fun stopGeneration()
    fun runtimeInfo(): RuntimeInfo
}

data class GenerationRequest(
    val messages: List<ChatMessage>,
    val settings: InferenceSettings
)

data class GenerationResult(
    val text: String,
    val inputTokens: Long? = null,
    val outputTokens: Long? = null,
    val firstTokenTimeMs: Long? = null,
    val generationTimeMs: Long? = null,
    val stopped: Boolean = false
)

data class ChatMessage(val role: Role, val content: String) {
    enum class Role { SYSTEM, USER, ASSISTANT, TOOL }
}

data class InferenceSettings(
    val temperature: Double = 0.7,
    val maxNewTokens: Int = 512,
    val topK: Int? = null,
    val topP: Double? = null,
    val minP: Double? = null,
    val repeatPenalty: Double? = null,
    val seed: Long? = null,
    val stopSequences: List<String> = emptyList(),
    val contextLength: Int? = null
)

data class RuntimeInfo(
    val name: String,
    val version: String,
    val backend: String?
)

enum class ActionArgumentType { STRING, INTEGER, NUMBER, BOOLEAN, OBJECT, ARRAY }

data class ActionArgument(
    val name: String,
    val type: ActionArgumentType,
    val required: Boolean = true,
    val maxLength: Int? = null,
)

data class ActionSchema(
    val version: Int = 1,
    val arguments: List<ActionArgument> = emptyList(),
)

interface Action<in I, out O> {
    val id: String
    val risk: RiskLevel
    val schema: ActionSchema get() = ActionSchema()
    val permission: String get() = "action:$id"
    val confirmationRequired: Boolean get() = risk == RiskLevel.SENSITIVE
    val stateChanging: Boolean get() = risk != RiskLevel.LOW
    val reversible: Boolean get() = false
    val idempotent: Boolean get() = !stateChanging
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
