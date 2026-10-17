package com.samanramezani.aichattest.ui

import androidx.compose.ui.test.assertExists
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithText
import org.junit.Rule
import org.junit.Test

class SettingsScreenTest {
    @get:Rule
    val composeRule = createComposeRule()

    @Test
    fun emptyModelState_exposesImportActionAndEmptyState() {
        composeRule.setContent {
            SettingsScreen(
                models = emptyList(),
                active = null,
                error = null,
                onImport = {},
                onRefresh = {},
                onActivate = {},
                onDeactivate = null,
                onDelete = {},
            )
        }

        composeRule.onNodeWithText("تنظیمات").assertExists()
        composeRule.onNodeWithText("افزودن مدل").assertExists()
        composeRule.onNodeWithText("هنوز بستهٔ ONNX Runtime GenAI وارد نشده است.").assertExists()
    }

    @Test
    fun unavailableInferenceState_isHonestAboutMissingCapability() {
        composeRule.setContent {
            SettingsScreen(
                models = emptyList(),
                active = null,
                error = null,
                onImport = {},
                onRefresh = {},
                onActivate = {},
                onDeactivate = null,
                onDelete = {},
            )
        }

        composeRule.onNodeWithText("استنتاج").performClick()
        composeRule.onNodeWithText("خارج از دسترس").assertExists()
    }
}
