package com.samanramezani.aichattest

import android.content.Context
import android.os.Build
import com.woogit.aicore.actions.ActionExecutionService
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
import com.woogit.aicore.agent.AgentOrchestrator
import com.woogit.aicore.agent.AgentSession
import com.woogit.aicore.agent.ProtocolActionIntentPlanner
import com.woogit.aicore.conversation.ConversationStore
import com.woogit.aicore.conversation.InMemoryConversationStore
import com.woogit.aicore.conversation.ConversationHistoryRepository
import com.woogit.aicore.conversation.InMemoryConversationHistoryRepository
import com.woogit.aicore.domain.ActionRegistry
import com.woogit.aicore.domain.CapabilityProvider
import com.woogit.aicore.domain.Verifier
import com.woogit.aicore.domain.ModelResult
import com.woogit.aicore.runtime.RuntimeAdapter
import com.woogit.aicore.runtime.android.LlamaCppAndroidRuntimeAdapter
import java.nio.file.Files

/** Application composition root. Implementations are wired here, never inside UI screens. */
class AppContainer(context: Context? = null) {
    private val appContext = context?.applicationContext

    val modelRuntime: RuntimeAdapter = LlamaCppAndroidRuntimeAdapter()

    val actionRegistry: ActionRegistry = DefaultActionRegistry().also { registry ->
        if (appContext != null) {
            val workspace = appContext.filesDir.toPath().resolve("workspace")
            Files.createDirectories(workspace)
            registry.registerBuiltinFileActions(workspace)
            registry.registerProviderActions(
                modelInfo = {
                    val active = modelManager?.activeModel()
                    when (active) {
                        is ModelResult.Success -> active.value?.let {
                            "name=${it.displayName}, quantization=${it.quantization}, sizeBytes=${it.sizeBytes}"
                        } ?: "no-active-model"
                        is ModelResult.Failure -> "unavailable: ${active.error.message}"
                        null -> "no-model-manager"
                    }
                },
                performanceStats = { "runtime=${modelRuntime.runtimeInfo().name}, backend=${modelRuntime.runtimeInfo().backend ?: "unknown"}" },
                deviceInfo = {
                    "manufacturer=${Build.MANUFACTURER}, model=${Build.MODEL}, sdk=${Build.VERSION.SDK_INT}"
                },
            )
        }
    }

    val conversationHistory: ConversationHistoryRepository = appContext?.let {
        AndroidConversationHistoryRepository(it)
    } ?: InMemoryConversationHistoryRepository()

    val modelManager: AndroidModelManager? = appContext?.let {
        val directory = it.filesDir.toPath().resolve("models")
        Files.createDirectories(directory)
        AndroidModelManager(it.contentResolver, directory, modelRuntime)
    }

    private val agentConversation: ConversationStore = InMemoryConversationStore()
    private val capabilityProvider: CapabilityProvider = object : CapabilityProvider {
        override fun supports(capability: String): Boolean = capability in setOf("filesystem", "runtime", "device", "utility")
    }
    private val checkpointStore = InMemoryActionCheckpointStore()
    private val lifecycle = ActionLifecycle(
        registry = actionRegistry,
        capabilityProvider = capabilityProvider,
        checkpointStore = checkpointStore,
        retryPolicy = DefaultActionRetryPolicy(2),
    )
    private val verifier = Verifier<Any> { result ->
        com.woogit.aicore.domain.VerificationResult(true, result.toString())
    }
    private val coordinator = ActionPlanCoordinator(lifecycle, ProtocolActionIntentPlanner())
    private val actionExecutor: suspend (ActionPlan) -> ActionExecutionOutcome = { plan ->
        val execution = lifecycle.executeApproved(plan.prepared.executionId, verifier)
        when (execution) {
            is com.woogit.aicore.actions.ActionExecutionState.Completed -> ActionExecutionOutcome(
                success = true,
                verified = execution.verification.success,
                message = execution.verification.evidence ?: "Action completed",
                data = execution.verification.evidence,
            )
            is com.woogit.aicore.actions.ActionExecutionState.Failed -> ActionExecutionOutcome(
                success = false,
                verified = false,
                message = execution.message,
                errorCode = "ACTION_FAILED",
            )
            else -> ActionExecutionOutcome(false, false, "Action was not executed", errorCode = "NOT_EXECUTED")
        }
    }

    val agentSession: AgentSession? = appContext?.let {
        AgentSession(
            orchestrator = AgentOrchestrator(
                contextProvider = com.woogit.aicore.conversation.DefaultContextProvider(agentConversation),
                runtime = modelRuntime,
            ),
            conversationStore = agentConversation,
            actionPlanCoordinator = coordinator,
            actionExecutor = actionExecutor,
        )
    }

    val actionExecutionService = ActionExecutionService(lifecycle, ApprovalController(lifecycle))
}
