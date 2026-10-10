package com.woogit.aicore.agent

import com.woogit.aicore.actions.CalculateAction
import com.woogit.aicore.actions.DefaultActionRegistry
import com.woogit.aicore.actions.GetTimeAction
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class ActionToolPromptTest {
    @Test
    fun promptListsOnlyActionsActuallyRegisteredAndTheirArguments() {
        val registry = DefaultActionRegistry().apply {
            register("utility", CalculateAction())
            register("utility", GetTimeAction())
        }

        val prompt = ActionToolPrompt.build(registry)

        assertContains(prompt, "calculate")
        assertContains(prompt, "expression: string, required, maxLength=512")
        assertContains(prompt, "get_time")
        assertFalse("create_file" in prompt)
        assertContains(prompt, """"version":1,"actionId":"call-1","action":"calculate"""")
        assertContains(prompt, "فقط یک شیء JSON")
    }

    @Test
    fun emptyRegistryExplicitlyDisablesActionRequests() {
        val prompt = ActionToolPrompt.build(DefaultActionRegistry())

        assertContains(prompt, "هیچ ابزاری در Registry ثبت نشده است")
        assertTrue(prompt.contains("هیچ ActionRequest تولید نکن"))
    }
}
