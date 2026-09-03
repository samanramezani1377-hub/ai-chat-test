package com.samanramezani.aichattest.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.samanramezani.aichattest.AppContainer
import com.woogit.aicore.runtime.android.LlamaCppAndroidRuntimeAdapter

@Composable
internal fun GpuDebugControls() {
    val runtime = AppContainer.latest?.modelRuntime as? LlamaCppAndroidRuntimeAdapter ?: return
    var selected by remember { mutableIntStateOf(runtime.gpuLayers()) }

    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Text("حالت GPU برای دیباگ موقت", style = MaterialTheme.typography.titleMedium)
        Text("فقط برای تست کرش. انتخاب بعدیِ فعال‌سازی مدل اعمال می‌شود.", color = MaterialTheme.colorScheme.onSurfaceVariant, style = MaterialTheme.typography.bodySmall)
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            listOf(
                LlamaCppAndroidRuntimeAdapter.GPU_LAYERS_CPU_ONLY to "۰ · CPU",
                LlamaCppAndroidRuntimeAdapter.GPU_LAYERS_70 to "۷۰ · GPU",
                LlamaCppAndroidRuntimeAdapter.GPU_LAYERS_MAX to "۹۹ · GPU Max",
            ).forEach { (value, label) ->
                val selectedNow = selected == value
                if (selectedNow) {
                    Button(onClick = { runtime.setGpuLayers(value); selected = value }, modifier = Modifier.weight(1f).heightIn(min = 48.dp)) { Text(label) }
                } else {
                    OutlinedButton(onClick = { runtime.setGpuLayers(value); selected = value }, modifier = Modifier.weight(1f).heightIn(min = 48.dp)) { Text(label) }
                }
            }
        }
    }
}
