package com.samanramezani.aichattest.ui

import androidx.compose.runtime.Composable
import com.woogit.aicore.domain.ModelDescriptor

/**
 * Keeps MainScreen's existing wiring while the settings UI remains a single,
 * user-facing screen. GPU-only is a runtime policy, not a user setting.
 */
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
