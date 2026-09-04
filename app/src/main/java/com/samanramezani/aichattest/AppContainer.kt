package com.samanramezani.aichattest

import android.content.Context
import android.os.Build
import com.woogit.aicore.actions.ActionExecutionService
import com.woogit.aicore.actions.ActionExecutionState
import com.woogit.aicore.actions.ActionLifecycle
import com.woogit.aicore.actions.ApprovalController
import com.woogit.aicore.actions.BuiltinActionVerifier
import com.woogit.aicore.actions.DefaultActionRegistry
import com.woogit.aicore.actions.DefaultActionRetryPolicy
import com.woogit.aicore.actions.InMemoryActionCheckpointStore
import com.woogit.aicore.actions.PreparedAction
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
import com.woogit.aicore.domain.ApiProviderConfig
import com.woogit.aicore.domain.ApiProviderConfigStore
import com.woogit.aicore.domain.CapabilityProvider
import com.woogit.aicore.domain.ModelResult
import com.woogit.aicore.domain.VerificationResult
import com.woogit.aicore.domain.Verifier
import com.woogit.aicore.observability.CentralObservability
import com.woogit.aicore.observability.ErrorReport
import com.woogit.aicore.observability.ExecutionTraceEvent
import com.woogit.aicore.runtime.RemoteApiRuntimeAdapter
import com.woogit.aicore.runtime.RuntimeAdapter
import com.woogit.aicore.runtime.RuntimeMetrics
import java.nio.file.Files

/** Application composition root. Inference is remote/API based; no local model runtime is wired. */
class AppContainer(context: Context? = null) {
    companion object {
        @Volatile var latest: AppContainer? = null
            private set
    }

    init { latest = this }

    private val appContext = context?.applicationContext
    private val workspaceRoot = appContext?.filesDir?.toPath()?.resolve("workspace")
    private val appVersion = runCatching { BuildConfig.VERSION_NAME }.getOrDefault("unknown")

    val observability: CentralObservability = CentralObservability(
        errorStore = appContext?.let { AndroidErrorReportStore(it) } ?: com.woogit.aicore.observability.InMemoryErrorReportStore(),
        traceStore = appContext?.let { AndroidExecutionTraceStore(it) } ?: com.woogit.aicore.observability.InMemoryExecutionTraceStore(),
    )

    val modelRuntime: RuntimeAdapter = RemoteApiRuntimeAdapter { ApiProviderConfigStore.current }
    val conversationHistory: ConversationHistoryRepository = appContext?.let { AndroidConversationHistoryRepository(it) } ?: InMemoryConversationHistoryRepository()
    val modelManager: AndroidModelManager? = appContext?.let { AndroidModelManager(it.contentResolver, it.filesDir.toPath().resolve("models"), modelRuntime) }

    fun apiConfig(): ApiProviderConfig = ApiProviderConfigStore.current
    fun updateApiConfig(config: ApiProviderConfig) { ApiProviderConfigStore.current = config }

    val actionRegistry: ActionRegistry = DefaultActionRegistry().also { registry ->
        if (workspaceRoot != null) {
            Files.createDirectories(workspaceRoot)
            registry.registerBuiltinFileActions(workspaceRoot)
            registry.registerProviderActions(
                modelInfo = {
                    val config = ApiProviderConfigStore.current
                    "provider=${config.providerId}, model=${config.model}"
                },
                performanceStats = {
                    val info = modelRuntime.runtimeInfo()
                    val metrics = (modelRuntime as? RuntimeMetrics)?.lastGeneration()
                    if (metrics == null) "runtime=${info.name}, backend=${info.backend ?: "unknown"}, generation=none"
                    else "runtime=${info.name}, backend=${info.backend ?: "unknown"}, generationMs=${metrics.generationTimeMs ?: "n/a"}, firstTokenMs=${metrics.firstTokenTimeMs ?: "n/a"}, outputTokens=${metrics.outputTokens ?: "n/a"}, stopped=${metrics.stopped}"
                },
                deviceInfo = { "manufacturer=${Build.MANUFACTURER}, model=${Build.MODEL}, sdk=${Build.VERSION.SDK_INT}" },
            )
        }
    }

    private val capabilityProvider: CapabilityProvider = object : CapabilityProvider {
        override fun supports(capability: String): Boolean = capability in setOf("filesystem", "runtime", "device", "utility", "network")
    }
    private val checkpointStore = appContext?.let { AndroidActionCheckpointStore(it) } ?: InMemoryActionCheckpointStore()
    private val actionTraceSink = CentralActionTraceSink(observability)
    private val actionErrorLogSink = CentralActionErrorSink(observability, appVersion = appVersion)
    private val verifier = object : Verifier<Any> {
        override suspend fun verify(result: Any): VerificationResult {
            val text = result.toString()
            return if (text.isBlank()) VerificationResult(false, "Action returned an empty result") else VerificationResult(true, text)
        }
    }
    private val lifecycle = ActionLifecycle(
        actionRegistry,
        capabilityProvider,
        checkpointStore,
        retryPolicy = DefaultActionRetryPolicy(2),
        traceSink = actionTraceSink,
        errorLogSink = actionErrorLogSink,
        actionResultVerifier = workspaceRoot?.let { BuiltinActionVerifier(it) },
    )
    private val coordinator = ActionPlanCoordinator(lifecycle, ProtocolActionIntentPlanner())

    private fun executionOutcome(state: ActionExecutionState): ActionExecutionOutcome = when (state) {
        is ActionExecutionState.Completed -> ActionExecutionOutcome(true, state.verification.success, state.verification.evidence ?: "Action completed", state.verification.evidence)
        is ActionExecutionState.Failed -> ActionExecutionOutcome(false, false, state.message, errorCode = "ACTION_FAILED")
        ActionExecutionState.Unknown -> ActionExecutionOutcome(false, false, "Action execution state is unknown; reconciliation is required", errorCode = "ACTION_UNKNOWN")
        else -> ActionExecutionOutcome(false, false, "Action was not executed", errorCode = "NOT_EXECUTED")
    }

    private val actionExecutor: suspend (ActionPlan) -> ActionExecutionOutcome = { plan -> lifecycle.executeApproved(plan.prepared.executionId, verifier).let(::executionOutcome) }
    private val actionRetryExecutor: suspend (ActionPlan) -> ActionExecutionOutcome = { plan -> lifecycle.retryFailed(plan.prepared.executionId, verifier).let(::executionOutcome) }

    suspend fun pendingApproval(conversationId: String): PreparedAction? = lifecycle.pendingApprovals(conversationId).maxByOrNull { it.executionId }
    suspend fun executionCheckpoint(executionId: String): PreparedAction? = runCatching { lifecycle.checkpoint(executionId) }.getOrNull()
    suspend fun reconcileInterruptedActions(): List<PreparedAction> = lifecycle.markInterruptedExecutionsUnknown()

    suspend fun retryAction(executionId: String, conversationId: String? = null): ActionExecutionOutcome = runCatching {
        val prepared = lifecycle.checkpoint(executionId)
        if (conversationId != null) require(prepared.conversationId == conversationId) { "Retry belongs to another conversation" }
        lifecycle.retryFailed(executionId, verifier).let(::executionOutcome)
    }.getOrElse {
        reportError("Action", "RETRY_FAILED", "تلاش مجدد عملیات ناموفق بود.", it, executionId = executionId)
        ActionExecutionOutcome(false, false, it.message ?: "Retry failed", errorCode = "RETRY_FAILED")
    }

    fun actionTraces() = actionTraceSink.snapshot()
    fun actionErrors() = actionErrorLogSink.snapshot()

    suspend fun approveAndExecute(executionId: String, conversationId: String? = null): ActionExecutionOutcome = runCatching {
        val prepared = lifecycle.checkpoint(executionId)
        if (conversationId != null) require(prepared.conversationId == conversationId) { "Approval belongs to another conversation" }
        val approved = lifecycle.approve(executionId)
        lifecycle.executeApproved(approved.executionId, verifier).let(::executionOutcome)
    }.getOrElse {
        reportError("Approval", "APPROVAL_FAILED", "تأیید و اجرای عملیات انجام نشد.", it, executionId = executionId)
        ActionExecutionOutcome(false, false, it.message ?: "Approval failed", errorCode = "APPROVAL_FAILED")
    }

    suspend fun reject(executionId: String, conversationId: String? = null): Boolean = runCatching {
        val prepared = lifecycle.checkpoint(executionId)
        if (conversationId != null) require(prepared.conversationId == conversationId) { "Approval belongs to another conversation" }
        check(prepared.state == ActionExecutionState.AwaitingApproval) { "Action is not awaiting approval" }
        lifecycle.reject(prepared)
        true
    }.getOrElse {
        reportError("Approval", "REJECT_FAILED", "رد کردن عملیات انجام نشد.", it)
        false
    }

    suspend fun recentErrors(limit: Int = 100): List<ErrorReport> = observability.errors(limit)
    suspend fun copyErrorReport(reportId: String): String? = observability.copyableReport(reportId)
    suspend fun traceForExecution(executionId: String): List<ExecutionTraceEvent> = observability.traces(executionId)

    suspend fun recordRuntimeFailure(message: String, raw: String = message, taskId: String? = null) {
        reportError("Runtime", "RUNTIME_FAILED", message.ifBlank { "اجرای API با خطا مواجه شد." }, IllegalStateException(raw), taskId = taskId)
    }

    fun createAgentSession(conversationId: String, eventSink: suspend (AgentEvent) -> Unit = {}): AgentSession? {
        if (appContext == null) return null
        val store = HistoryConversationStore(conversationHistory, conversationId)
        val contextProvider = DefaultContextProvider(conversationStore = store, systemContext = { null }, persistentTaskContext = { null }, workspaceContext = { workspaceRoot?.toString() })
        return AgentSession(
            orchestrator = AgentOrchestrator(contextProvider = contextProvider, runtime = modelRuntime),
            conversationStore = store,
            actionPlanCoordinator = coordinator,
            actionExecutor = actionExecutor,
            actionRetryExecutor = actionRetryExecutor,
            eventSink = { event ->
                when (event) {
                    AgentEvent.Started -> observability.trace("agent:$conversationId", ExecutionTraceEvent.Phase.PREPARED, taskId = conversationId, message = "Agent started")
                    is AgentEvent.ActionPrepared -> observability.trace(event.executionId, ExecutionTraceEvent.Phase.PREPARED, taskId = conversationId, actionId = event.actionId)
                    is AgentEvent.ApprovalRequired -> observability.trace(event.executionId, ExecutionTraceEvent.Phase.APPROVAL_REQUIRED, taskId = conversationId, message = "Approval required")
                    is AgentEvent.ActionExecuted -> observability.trace(event.executionId, ExecutionTraceEvent.Phase.EXECUTED, taskId = conversationId)
                    is AgentEvent.Failed -> reportError("Agent", "AGENT_FAILED", "عامل هوشمند با خطا متوقف شد.", IllegalStateException(event.message), taskId = conversationId)
                    AgentEvent.Completed -> observability.trace("agent:$conversationId", ExecutionTraceEvent.Phase.VERIFIED, taskId = conversationId, message = "Agent completed")
                    is AgentEvent.Token -> Unit
                }
                eventSink(event)
            },
            conversationId = conversationId,
            contextProvider = contextProvider,
        )
    }

    val actionExecutionService = ActionExecutionService(lifecycle, ApprovalController(lifecycle))

    private suspend fun reportError(component: String, code: String, userMessageFa: String, throwable: Throwable, actionId: String? = null, taskId: String? = null, executionId: String? = null) {
        val trace = executionId?.let { observability.traces(it) }?.joinToString("\n") { "${it.timestamp} ${it.phase}: ${it.message ?: ""}" }
        observability.error(appVersion = appVersion, component = component, errorCode = code, userMessageFa = userMessageFa, rawMachineError = throwable.stackTraceToString(), actionId = actionId, taskId = taskId, trace = trace, runtimeInfo = modelRuntime.runtimeInfo().toString(), modelInfo = ApiProviderConfigStore.current.toString())
    }
}

private class HistoryConversationStore(private val repository: ConversationHistoryRepository, private val conversationId: String) : ConversationStore {
    override suspend fun append(message: ConversationMessage) { check(repository.append(conversationId, message, message.timestampEpochMs)) { "Conversation does not exist: $conversationId" } }
    override suspend fun recent(limit: Int): List<ConversationMessage> = repository.messages(conversationId).takeLast(limit)
    override suspend fun summary(): String? = null
    override suspend fun replaceSummary(summary: String) = Unit
}
