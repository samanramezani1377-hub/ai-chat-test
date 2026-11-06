package com.woogit.aicore.actions

import com.woogit.aicore.domain.Action
import com.woogit.aicore.domain.ActionRegistry
import java.util.concurrent.ConcurrentHashMap

class InMemoryActionRegistry : ActionRegistry {
    private val actions = ConcurrentHashMap<String, Action<Any, Any>>()
    private val categoryMap = ConcurrentHashMap<String, MutableSet<String>>()

    override fun register(category: String, action: Action<Any, Any>) {
        require(category.isNotBlank()) { "Action category must not be blank" }
        require(action.id.isNotBlank()) { "Action id must not be blank" }
        check(actions.putIfAbsent(action.id, action) == null) { "Action already registered: ${action.id}" }
        categoryMap.computeIfAbsent(category) { ConcurrentHashMap.newKeySet() }.add(action.id)
    }

    override fun find(actionId: String): Action<Any, Any>? = actions[actionId]

    override fun all(): List<Action<Any, Any>> = actions.values.sortedBy { it.id }

    override fun categories(): Set<String> = categoryMap.keys.toSet()
}
