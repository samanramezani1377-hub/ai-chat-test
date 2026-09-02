package com.samanramezani.aichattest.ui

import androidx.compose.runtime.Composable
import com.woogit.aicore.domain.ModelDescriptor

/** Compatibility overload for the pre-refactor MainScreen signature. */
@Composable
internal fun SettingsScreen(
    models: List<ModelDescriptor>,
    active: ModelDescriptor?,
    runtimeStatus: String,
    error: String?,
    onImport: () -> Unit,
    onDeactivate: () -> Unit,
    onRefresh: () -> Unit,
) {
    SettingsScreen(
        models = models,
        active = active,
        error = error,
        onImport = onImport,
        onRefresh = onRefresh,
        onActivate = {},
        onDeactivate = onDeactivate,
        onDelete = {},
    )
}
