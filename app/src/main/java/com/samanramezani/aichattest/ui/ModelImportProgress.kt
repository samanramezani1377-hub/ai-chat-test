package com.samanramezani.aichattest.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.woogit.aicore.domain.ModelDescriptor

/** Settings entry point with explicit UI state for a model import in progress. */
@Composable
fun SettingsScreen(
    models: List<ModelDescriptor>,
    active: ModelDescriptor?,
    error: String?,
    importBusy: Boolean,
    onImport: () -> Unit,
    onRefresh: () -> Unit,
    onActivate: (String) -> Unit,
    onDeactivate: (() -> Unit)? = null,
    onDelete: (String) -> Unit,
) {
    Column(verticalArrangement = Arrangement.spacedBy(UiTokens.itemGap)) {
        if (importBusy) {
            UiSurface {
                Column(
                    Modifier.fillMaxWidth().padding(UiTokens.compactPadding),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    Row(
                        Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(10.dp),
                    ) {
                        CircularProgressIndicator(Modifier.size(22.dp), strokeWidth = 2.dp)
                        Column(Modifier.weight(1f)) {
                            Text("در حال وارد کردن مدل…", style = MaterialTheme.typography.titleMedium)
                            Text(
                                "فایل در حال کپی و ثبت در فضای امن برنامه است. لطفاً برنامه را نبندید.",
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                    }
                    LinearProgressIndicator(Modifier.fillMaxWidth())
                }
            }
        }
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
}
