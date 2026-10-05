package com.samanramezani.aichattest.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.woogit.aicore.domain.ModelDescriptor

@Composable
fun SettingsScreen(
    models: List<ModelDescriptor>, active: ModelDescriptor?, error: String?, onImport: () -> Unit,
    draftModels: List<ModelDescriptor>, activeDraft: ModelDescriptor?, onImportDraft: () -> Unit,
    onAssignDraft: (String?) -> Unit, onDeleteDraft: (String) -> Unit,
    onRefresh: () -> Unit, onActivate: (String) -> Unit, onDeactivate: (() -> Unit)?, onDelete: (String) -> Unit,
) {
    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        SettingsScreen(models, active, error, onImport, onRefresh, onActivate, onDeactivate, onDelete)
        Surface(
            Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 4.dp),
            shape = RoundedCornerShape(20.dp),
            color = MaterialTheme.colorScheme.surfaceVariant
        ) {
            Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("مدل درفت", style = MaterialTheme.typography.titleMedium)
                Text(
                    if (activeDraft == null) "هیچ مدل درفتی به مدل فعال متصل نیست."
                    else "درفت فعال: " + activeDraft.displayName,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedButton(onClick = onImportDraft, enabled = active != null, modifier = Modifier.weight(1f)) {
                        Text("افزودن درفت")
                    }
                    if (activeDraft != null) {
                        Button(onClick = { onAssignDraft(null) }, modifier = Modifier.weight(1f)) {
                            Text("غیرفعال")
                        }
                    }
                }
                draftModels.forEach { draft ->
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        Button(
                            onClick = { onAssignDraft(draft.id) },
                            enabled = active != null && activeDraft?.id != draft.id,
                            modifier = Modifier.weight(1f)
                        ) { Text(if (activeDraft?.id == draft.id) "متصل" else "اتصال " + draft.displayName) }
                        OutlinedButton(onClick = { onDeleteDraft(draft.id) }) { Text("حذف") }
                    }
                }
            }
        }
    }
}
