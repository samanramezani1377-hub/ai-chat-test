package com.samanramezani.aichattest.ui

import androidx.compose.runtime.Composable
import com.woogit.aicore.domain.ModelDescriptor

/** Compatibility bridge for the current MainScreen wiring while the UI layer is being consolidated. */
@Composable
internal fun SettingsScreen(
    models: List<ModelDescriptor>,
    active: ModelDescriptor?,
    runtimeStatus: String,
    error: String?,
    onImport: () -> Unit,
    onRefresh: () -> Unit,
    onActivate: (String) -> Unit,
    onDeactivate: () -> Unit,
    onDelete: (String) -> Unit,
) {
    SettingsScreen(
        models = models,
        active = active,
        error = error,
        onImport = onImport,
        onRefresh = onRefresh,
        onActivate = onActivate,
        onDeactivate = onDeactivate,
        onDelete = onDelete,
    )
}
