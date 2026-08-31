package com.woogit.aicore.actions

import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class ActionDebugVisibilityTest {
    @Test
    fun debugDetailsCanBeHiddenWithoutRemovingTheDebugCapability() {
        val visibility = MutableActionDebugVisibility()
        assertTrue(visibility.fullErrorDetailsVisible)

        visibility.fullErrorDetailsVisible = false
        assertFalse(visibility.fullErrorDetailsVisible)

        visibility.fullErrorDetailsVisible = true
        assertTrue(visibility.fullErrorDetailsVisible)
    }

    @Test
    fun hiddenVisibilityOnlyControlsPresentation() {
        val visibility: ActionDebugVisibility = HiddenActionDebugVisibility()
        assertFalse(visibility.fullErrorDetailsVisible)
    }
}
