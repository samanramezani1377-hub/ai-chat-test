package com.samanramezani.aichattest.ui

import android.content.Context
import android.os.Handler
import android.os.Looper
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.samanramezani.aichattest.screen.ScreenVisionModelStore
import kotlinx.coroutines.launch

@Composable
internal fun ScreenVisionModelControls() {
    val context = androidx.compose.ui.platform.LocalContext.current
    val scope = rememberCoroutineScope()
    var files by remember { mutableStateOf(ScreenVisionModelStore.fileNames(context)) }
    var busy by remember { mutableStateOf(false) }
    var status by remember { mutableStateOf<String?>(null) }
    val mainHandler = remember { Handler(Looper.getMainLooper()) }

    val modelPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) scope.launch {
            busy = true
            try {
                val file = ScreenVisionModelStore.import(context, uri, projector = false)
                files = ScreenVisionModelStore.fileNames(context)
                status = "مدل بینایی وارد شد: " + file.name
            } catch (t: Throwable) { status = t.message ?: "وارد کردن مدل ناموفق بود." }
            finally { busy = false }
        }
    }
    val projectorPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) scope.launch {
            busy = true
            try {
                val file = ScreenVisionModelStore.import(context, uri, projector = true)
                files = ScreenVisionModelStore.fileNames(context)
                status = "Projector وارد شد: " + file.name
            } catch (t: Throwable) { status = t.message ?: "وارد کردن projector ناموفق بود." }
            finally { busy = false }
        }
    }

    Surface(Modifier.fillMaxWidth(), shape = RoundedCornerShape(18.dp), color = MaterialTheme.colorScheme.surfaceVariant) {
        Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(9.dp)) {
            Text("مدل بینایی محلی", style = MaterialTheme.typography.titleMedium)
            Text("برای فهم واقعی تصویر، فایل مدل Vision و فایل mmproj متناظر لازم است. این مدل مستقل از مدل متنی فعلی اجرا می‌شود.", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            Text("مدل: " + (files.first ?: "وارد نشده"), style = MaterialTheme.typography.bodySmall)
            Text("Projector: " + (files.second ?: "وارد نشده"), style = MaterialTheme.typography.bodySmall)
            Button(
                onClick = {
                    busy = true
                    status = "دانلود مدل کوچک SmolVLM آغاز شد…"
                    scope.launch {
                        try {
                            ScreenVisionModelStore.downloadRecommended(context) { progress ->
                                mainHandler.post { status = progress }
                            }
                            files = ScreenVisionModelStore.fileNames(context)
                            status = "مدل و projector با کنترل SHA-256 آماده شدند."
                        } catch (t: Throwable) {
                            status = t.message ?: "دانلود مدل بینایی ناموفق بود."
                        } finally { busy = false }
                    }
                },
                enabled = !busy,
                modifier = Modifier.fillMaxWidth(),
            ) { Text(if (busy) "در حال آماده‌سازی مدل…" else "دانلود مدل پیشنهادی کوچک (حدود ۲۸۰ مگابایت)") }
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedButton(onClick = { modelPicker.launch(arrayOf("application/octet-stream", "application/*")) }, enabled = !busy, modifier = Modifier.weight(1f)) { Text("واردکردن مدل") }
                OutlinedButton(onClick = { projectorPicker.launch(arrayOf("application/octet-stream", "application/*")) }, enabled = !busy, modifier = Modifier.weight(1f)) { Text("واردکردن mmproj") }
            }
            Text("فایل‌های مدل و mmproj باید از یک نسخه و یک خانواده باشند. تحلیل صفحه فقط هنگام ارسال سؤال اجرا می‌شود.", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            status?.let { Text(it, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.secondary) }
        }
    }
}
