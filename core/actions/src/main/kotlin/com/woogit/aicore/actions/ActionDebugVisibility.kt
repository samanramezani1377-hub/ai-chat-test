package com.woogit.aicore.actions

/**
 * Controls visibility of the development-only full error surface.
 * Hiding it changes presentation only; trace/error data remains available to the core.
 */
interface ActionDebugVisibility {
    val fullErrorDetailsVisible: Boolean
}

data class MutableActionDebugVisibility(
    override var fullErrorDetailsVisible: Boolean = true
) : ActionDebugVisibility

class HiddenActionDebugVisibility : ActionDebugVisibility {
    override val fullErrorDetailsVisible: Boolean = false
}
