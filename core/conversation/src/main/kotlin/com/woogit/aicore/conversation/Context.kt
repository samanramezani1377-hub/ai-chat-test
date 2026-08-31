package com.woogit.aicore.conversation

/** Context supplied to the model/agent for one turn. */
data class ConversationContext(
    val system: String?,
    val persistentTask: String?,
    val summary: String?,
    val recentMessages: List<ConversationMessage>,
    val workspace: String?
)

interface ContextProvider {
    suspend fun build(requestedRecentMessages: Int = 10): ConversationContext
}

class DefaultContextProvider(
    private val conversationStore: ConversationStore,
    private val systemContext: () -> String?,
    private val persistentTaskContext: () -> String?,
    private val workspaceContext: () -> String?
) : ContextProvider {
    override suspend fun build(requestedRecentMessages: Int): ConversationContext {
        require(requestedRecentMessages >= 0) { "requestedRecentMessages must be non-negative" }
        return ConversationContext(
            system = systemContext(),
            persistentTask = persistentTaskContext(),
            summary = conversationStore.summary(),
            recentMessages = conversationStore.recent(requestedRecentMessages),
            workspace = workspaceContext()
        )
    }
}
