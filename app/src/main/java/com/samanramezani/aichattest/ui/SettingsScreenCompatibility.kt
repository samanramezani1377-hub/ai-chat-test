package com.samanramezani.aichattest.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
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
    Column(Modifier.fillMaxWidth()) {
        Column(Modifier.weight(1f)) {
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
        HorizontalDivider()
        Surface(Modifier.fillMaxWidth(), tonalElevation = 2.dp) {
            Column(
                Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 10.dp),
                verticalArrangement = Arrangement.spacedBy(6.dp),
            ) {
                GpuDebugControls()
            }
        }
    }
}
