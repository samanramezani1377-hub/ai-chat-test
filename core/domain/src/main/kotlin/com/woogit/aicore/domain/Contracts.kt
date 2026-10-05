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
    /** Balanced default for local instruction-following models. */
    val temperature: Double = 0.7,
    /** Practical response size that keeps mobile inference responsive. */
    val maxNewTokens: Int = 512,
    /** Common local-LLM sampling defaults; nullable means explicitly disabled only when chosen by the user. */
    val topK: Int? = 40,
    val topP: Double? = 0.9,
    val minP: Double? = 0.0,
    val repeatPenalty: Double? = 1.1,
    /** Null means use a non-deterministic seed. */
    val seed: Long? = null,
    /** No forced stop strings by default; the model/runtime decides when generation naturally ends. */
    val stopSequences: List<String> = emptyList(),
    /** Conservative local default; runtime capability may further constrain this value. */
    val contextLength: Int? = 8192,
    /** Number of recent conversation messages included in generated context. */
    val recentMessages: Int = 10,
    /** Maximum number of real Action steps the Agent may execute for one request. */
    val maxActionSteps: Int = 4,
) {
    init {
        require(recentMessages >= 0) { "recentMessages must be non-negative" }
        require(maxActionSteps >= 0) { "maxActionSteps must be non-negative" }
    }
}

data class RuntimeInfo(
    val name: String,
    val version: String,
    val backend: String?,
    val threads: Int? = null,
    val gpuLayers: Int? = null,
    val contextLength: Int? = null,
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
