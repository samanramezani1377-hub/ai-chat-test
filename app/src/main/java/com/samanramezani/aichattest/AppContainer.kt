package com.samanramezani.aichattest

import android.content.Context
import android.os.Build
import com.woogit.aicore.actions.ActionExecutionService
import com.woogit.aicore.actions.ActionExecutionState
import com.woogit.aicore.actions.ActionLifecycle
import com.woogit.aicore.actions.ApprovalController
import com.woogit.aicore.actions.DefaultActionRegistry
import com.woogit.aicore.actions.DefaultActionRetryPolicy
import com.woogit.aicore.actions.InMemoryActionCheckpointStore
import com.woogit.aicore.actions.registerBuiltinFileActions
import com.woogit.aicore.actions.registerProviderActions
import com.woogit.aicore.agent.ActionExecutionOutcome
import com.woogit.aicore.agent.ActionPlan
import com.woogit.aicore.agent.ActionPlanCoordinator
import com.woogit.aicore.agent.AgentEvent
import com.woogit.aicore.agent.AgentOrchestrator
import com.woogit.aicore.agent.AgentSession
import com.woogit.aicore.agent.ProtocolActionIntentPlanner
import com.woogit.aicore.conversation.ConversationHistoryRepository
import com.woogit.aicore.conversation.ConversationMessage
import com.woogit.aicore.conversation.ConversationStore
import com.woogit.aicore.conversation.DefaultContextProvider
import com.woogit.aicore.conversation.InMemoryConversationHistoryRepository
import com.woogit.aicore.domain.ActionRegistry
import com.woogit.aicore.domain.CapabilityProvider
import com.woogit.aicore.domain.InferenceSettings
import com.woogit.aicore.domain.ModelResult
import com.woogit.aicore.domain.Verifier
import com.woogit.aicore.runtime.RuntimeAdapter
import com.woogit.aicore.runtime.android.LlamaCppAndroidRuntimeAdapter
import java.nio.file.Files

/** Application composition root. Implementations are wired here, never inside UI screens. */
class AppContainer(context: Context? = null) {
    private val appContext = context?.applicationContext

    val modelRuntime: RuntimeAdapter = LlamaCppAndroidRuntimeAdapter()
    val conversationHistory: ConversationHistoryRepository = appContext?.let { AndroidConversationHistoryRepository(it) } ?: InMemoryConversationHistoryRepository()
    val modelManager: AndroidModelManager? = appContext?.let {
        val directory = it.filesDir.toPath().resolve("models")
        Files.createDirectories(directory)
        AndroidModelManager(it.contentResolver, directory, modelRuntime)
    }

    val actionRegistry: ActionRegistry = DefaultActionRegistry().also { registry ->
        if (appContext != null) {
            val workspace = appContext.filesDir.toPath().resolve("workspace")
            Files.createDirectories(workspace)
            registry.registerBuiltinFileActions(workspace)
            registry.registerProviderActions(
                modelInfo = {
                    when (val active = modelManager?.activeModel()) {
                        is ModelResult.Success -> active.value?.let { "name=${it.displayName}, quantization=${it.quantization}, sizeBytes=${it.sizeBytes}" } ?: "no-active-model"
                        is ModelResult.Failure -> "unavailable: ${active.error.message}"
                        null -> "no-model-manager"
                    }
                },
                performanceStats = {
                    val info = modelRuntime.runtimeInfo()
                    "runtime=${info.name}, backend=${info.backend ?: "unknown"}"
                },
                deviceInfo = { "manufacturer=${Build.MANUFACTURER}, model=${Build.MODEL}, sdk=${Build.VERSION.SDK_INT}" },
            )
        }
    }

    private val capabilityProvider: CapabilityProvider = object : CapabilityProvider {
        override fun supports(capability: String): Boolean = capability in setOf("filesystem", "runtime", "device", "utility")
    }
    private val checkpointStore = InMemoryActionCheckpointStore()
    private val lifecycle = ActionLifecycle(actionRegistry, capabilityProvider, checkpointStore, retryPolicy = DefaultActionRetryPolicy(2))
    private val verifier = Verifier<Any> { result -> com.woogit.aicore.domain.VerificationResult(true, result.toString()) }
    private val coordinator = ActionPlanCoordinator(lifecycle, ProtocolActionIntentPlanner())

    private fun executionOutcome(state: ActionExecutionState): ActionExecutionOutcome = when (state) {
        is ActionExecutionState.Completed -> ActionExecutionOutcome(true, state.verification.success, state.verification.evidence ?: "Action completed", state.verification.evidence)
        is ActionExecutionState.Failed -> ActionExecutionOutcome(false, false, state.message, errorCode = "ACTION_FAILED")
        else -> ActionExecutionOutcome(false, false, "Action was not executed", errorCode = "NOT_EXECUTED")
    }

    private val actionExecutor: suspend (ActionPlan) -> ActionExecutionOutcome = { plan ->
        executionOutcome(lifecycle.executeApproved(plan.prepared.executionId, verifier))
    }

    fun approveAndExecute(executionId: String): ActionExecutionOutcome = runCatching {
        executionOutcome(lifecycle.approve(executionId).let { lifecycle.executeApproved(it.executionId, verifier) })
    }.getOrElse { ActionExecutionOutcome(false, false, it.message ?: "Approval failed", errorCode = "APPROVAL_FAILED") }

    fun reject(executionId: String): Boolean = runCatching {
        val prepared = lifecycle.checkpoint(executionId)
        lifecycle.reject(prepared)
        true
    }.getOrDefault(false)

    fun createAgentSession(conversationId: String, eventSink: suspend (AgentEvent) -> Unit = {}): AgentSession? {
        if (appContext == null || modelManager == null) return null
        val store = HistoryConversationStore(conversationHistory, conversationId)
        return AgentSession(
            orchestrator = AgentOrchestrator(contextProvider = DefaultContextProvider(store), runtime = modelRuntime),
            conversationStore = store,
            actionPlanCoordinator = coordinator,
            actionExecutor = actionExecutor,
            eventSink = eventSink,
        )
    }

    val actionExecutionService = ActionExecutionService(lifecycle, ApprovalController(lifecycle))
}

private class HistoryConversationStore(private val repository: ConversationHistoryRepository, private val conversationId: String) : ConversationStore {
    override suspend fun append(message: ConversationMessage) { check(repository.append(conversationId, message, message.timestampEpochMs)) { "Conversation does not exist: $conversationId" } }
    override suspend fun recent(limit: Int): List<ConversationMessage> = repository.messages(conversationId).takeLast(limit)
    override suspend fun summary(): String? = null
    override suspend fun replaceSummary(summary: String) = Unit
}
