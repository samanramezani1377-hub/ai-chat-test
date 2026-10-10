package com.woogit.aicore.actions

import com.woogit.aicore.domain.Action
import com.woogit.aicore.domain.ActionRegistry

class DefaultActionRegistry : ActionRegistry {
    private val actions = linkedMapOf<String, Action<Any, Any>>()
    private val categoriesByAction = linkedMapOf<String, String>()

    @Synchronized
    override fun register(category: String, action: Action<Any, Any>) {
        require(category.isNotBlank()) { "category must not be blank" }
        require(action.id.isNotBlank()) { "action id must not be blank" }
        require(action.id !in actions) { "Action already registered: ${action.id}" }
        actions[action.id] = action
        categoriesByAction[action.id] = category
    }

    @Synchronized
    override fun find(actionId: String): Action<Any, Any>? = actions[actionId]

    @Synchronized
    override fun all(): List<Action<Any, Any>> = actions.values.toList()

    @Synchronized
    override fun categories(): Set<String> = categoriesByAction.values.toSet()
}
