package com.samanramezani.aichattest.ui

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.provider.Settings
import android.media.projection.MediaProjectionManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.samanramezani.aichattest.screen.FloatingAssistantService
import com.samanramezani.aichattest.screen.ScreenSessionService

@Composable
internal fun ScreenAssistantSettings() {
    val context = androidx.compose.ui.platform.LocalContext.current
    val prefs = remember(context) { context.getSharedPreferences("screen_assistant", Context.MODE_PRIVATE) }
    var overlayEnabled by remember { mutableStateOf(prefs.getBoolean("overlay_enabled", false)) }
    var captureRunning by remember { mutableStateOf(ScreenSessionService.isRunning) }
    var status by remember { mutableStateOf<String?>(null) }

    DisposableEffect(prefs) {
        val listener = android.content.SharedPreferences.OnSharedPreferenceChangeListener { _, key ->
            if (key == "capture_running") {
                captureRunning = prefs.getBoolean("capture_running", false)
            }
        }
        prefs.registerOnSharedPreferenceChangeListener(listener)
        onDispose { prefs.unregisterOnSharedPreferenceChangeListener(listener) }
    }

    fun enableOverlay() {
        if (!Settings.canDrawOverlays(context)) return
        ContextCompat.startForegroundService(context, Intent(context, FloatingAssistantService::class.java).setAction(FloatingAssistantService.ACTION_ENABLE))
        prefs.edit().putBoolean("overlay_enabled", true).apply()
        overlayEnabled = true
        status = "حباب دینا فعال شد."
    }

    val overlayPermission = rememberLauncherForActivityResult(ActivityResultContracts.StartActivityForResult()) {
        if (Settings.canDrawOverlays(context)) enableOverlay() else {
            overlayEnabled = false
            prefs.edit().putBoolean("overlay_enabled", false).apply()
            status = "مجوز نمایش روی برنامه‌های دیگر داده نشد."
        }
    }
    val projectionPermission = rememberLauncherForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        val data = result.data
        if (result.resultCode == android.app.Activity.RESULT_OK && data != null) {
            val service = Intent(context, ScreenSessionService::class.java)
                .setAction(ScreenSessionService.ACTION_START)
                .putExtra(ScreenSessionService.EXTRA_RESULT_CODE, result.resultCode)
                .putExtra(ScreenSessionService.EXTRA_RESULT_DATA, data)
            ContextCompat.startForegroundService(context, service)
            status = "درخواست شروع نشست ارسال شد؛ پس از آماده‌شدن سرویس وضعیت فعال نمایش داده می‌شود."
        } else { captureRunning = false; status = "اجازه مشاهده صفحه داده نشد." }
    }

    Column(verticalArrangement = Arrangement.spacedBy(14.dp)) {
        Text("حباب شناور دینا", style = MaterialTheme.typography.titleLarge)
        Text("حباب روی برنامه‌های دیگر نمایش داده می‌شود و قابل جابه‌جایی است. با لمس آن یک پنجره گفت‌وگوی شناور باز می‌شود؛ برنامه اصلی بسته یا به پیش‌زمینه آورده نمی‌شود.", color = MaterialTheme.colorScheme.onSurfaceVariant)
        Surface(Modifier.fillMaxWidth(), shape = RoundedCornerShape(18.dp), color = MaterialTheme.colorScheme.surfaceVariant) {
            Row(Modifier.fillMaxWidth().padding(14.dp), horizontalArrangement = Arrangement.SpaceBetween) {
                Column(Modifier.weight(1f)) {
                    Text("فعال‌بودن حباب", style = MaterialTheme.typography.titleMedium)
                    Text("برای نمایش روی سایر برنامه‌ها مجوز Android لازم است.", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
                Switch(checked = overlayEnabled, onCheckedChange = { enabled ->
                    if (!enabled) {
                        context.stopService(Intent(context, FloatingAssistantService::class.java).setAction(FloatingAssistantService.ACTION_DISABLE))
                        prefs.edit().putBoolean("overlay_enabled", false).apply()
                        overlayEnabled = false
                        status = "حباب خاموش شد."
                    } else if (Settings.canDrawOverlays(context)) enableOverlay() else {
                        val intent = Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION, Uri.parse("package:" + context.packageName))
                        overlayPermission.launch(intent)
                    }
                })
            }
        }

        ScreenVisionModelControls()
        HorizontalDivider()
        Text("مشاهده صفحه", style = MaterialTheme.typography.titleLarge)
        Text("وقتی پنجره مجوز Android باز شد، «کل صفحه / Entire screen» را انتخاب کن. تا وقتی نشست فعال است دریافت فریم‌ها ادامه دارد؛ فقط آخرین تصویر کم‌حجم در حافظه موقت می‌ماند و تحلیل Vision فقط هنگام سؤال مرتبط اجرا می‌شود.", color = MaterialTheme.colorScheme.onSurfaceVariant)
        Surface(Modifier.fillMaxWidth(), shape = RoundedCornerShape(18.dp), color = MaterialTheme.colorScheme.surfaceVariant) {
            Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text(if (captureRunning) "نشست صفحه فعال است" else "نشست صفحه متوقف است", style = MaterialTheme.typography.titleMedium)
                Button(onClick = {
                    val manager = context.getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
                    projectionPermission.launch(manager.createScreenCaptureIntent())
                }, enabled = !captureRunning, modifier = Modifier.fillMaxWidth()) { Text("شروع مشاهده صفحه") }
                OutlinedButton(onClick = {
                    context.startService(Intent(context, ScreenSessionService::class.java).setAction(ScreenSessionService.ACTION_STOP))
                    captureRunning = false
                    status = "مشاهده صفحه متوقف شد و فریم موقت پاک شد."
                }, enabled = captureRunning, modifier = Modifier.fillMaxWidth()) { Text("توقف مشاهده صفحه") }
                Text("Android برای هر نشست جدید مجوز سیستمی جداگانه می‌خواهد. در صورت توقف سیستم‌عامل، نشست را دوباره شروع کن.", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
        status?.let { Text(it, color = MaterialTheme.colorScheme.secondary, style = MaterialTheme.typography.bodyMedium) }
    }
}
