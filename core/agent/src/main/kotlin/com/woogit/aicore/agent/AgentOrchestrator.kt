package com.woogit.aicore.agent

import com.woogit.aicore.conversation.ContextProvider
import com.woogit.aicore.conversation.ConversationContext
import com.woogit.aicore.domain.GenerationRequest
import com.woogit.aicore.domain.GenerationResult
import com.woogit.aicore.domain.InferenceSettings
import com.woogit.aicore.domain.ModelRuntime

interface ContextToGenerationRequest {
    fun create(context: ConversationContext, settings: InferenceSettings): GenerationRequest
}

class DefaultContextToGenerationRequest : ContextToGenerationRequest {
    override fun create(context: ConversationContext, settings: InferenceSettings): GenerationRequest {
        val messages = buildList {
            context.system?.takeIf { it.isNotBlank() }?.let {
                add(com.woogit.aicore.domain.ChatMessage(com.woogit.aicore.domain.ChatMessage.Role.SYSTEM, it))
            }
            context.persistentTask?.takeIf { it.isNotBlank() }?.let {
                add(com.woogit.aicore.domain.ChatMessage(com.woogit.aicore.domain.ChatMessage.Role.SYSTEM, it))
            }
            context.summary?.takeIf { it.isNotBlank() }?.let {
                add(com.woogit.aicore.domain.ChatMessage(com.woogit.aicore.domain.ChatMessage.Role.SYSTEM, it))
            }
            context.workspace?.takeIf { it.isNotBlank() }?.let {
                add(com.woogit.aicore.domain.ChatMessage(com.woogit.aicore.domain.ChatMessage.Role.SYSTEM, it))
            }
            context.recentMessages.forEach { message ->
                val role = when (message.role) {
                    com.woogit.aicore.conversation.ConversationMessage.Role.SYSTEM -> com.woogit.aicore.domain.ChatMessage.Role.SYSTEM
                    com.woogit.aicore.conversation.ConversationMessage.Role.USER -> com.woogit.aicore.domain.ChatMessage.Role.USER
                    com.woogit.aicore.conversation.ConversationMessage.Role.ASSISTANT -> com.woogit.aicore.domain.ChatMessage.Role.ASSISTANT
                    com.woogit.aicore.conversation.ConversationMessage.Role.TOOL -> com.woogit.aicore.domain.ChatMessage.Role.TOOL
                }
                add(com.woogit.aicore.domain.ChatMessage(role, message.content))
            }
        }
        return GenerationRequest(messages = messages, settings = settings)
    }
}

class AgentOrchestrator(
    private val contextProvider: ContextProvider,
    private val runtime: ModelRuntime,
    private val requestFactory: ContextToGenerationRequest = DefaultContextToGenerationRequest()
) {
    suspend fun generate(
        settings: InferenceSettings,
        requestedRecentMessages: Int = 10,
        onToken: suspend (String) -> Unit
    ): GenerationResult {
        val context = contextProvider.build(requestedRecentMessages)
        val request = requestFactory.create(context, settings)
        return runtime.generate(request, onToken)
    }
}
