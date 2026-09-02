package com.samanramezani.aichattest.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import com.woogit.aicore.domain.InferenceSettings
import com.woogit.aicore.domain.InferenceSettingsStore
import com.woogit.aicore.domain.ModelDescriptor

private val aiSections = listOf("مدل", "استنتاج", "زمینه", "عامل")
private val appSections = listOf("فضای کار", "لاگ و عیب‌یابی", "عملکرد", "امنیت و تأیید")

@Composable
fun SettingsScreen(
    models: List<ModelDescriptor>, active: ModelDescriptor?, error: String?, onImport: () -> Unit,
    onRefresh: () -> Unit, onActivate: (String) -> Unit, onDeactivate: (() -> Unit)? = null, onDelete: (String) -> Unit,
) {
    var section by remember { mutableStateOf("مدل") }
    LazyColumn(Modifier.fillMaxSize().padding(horizontal = UiTokens.pagePadding), contentPadding = PaddingValues(top = UiTokens.sectionGap, bottom = 36.dp), verticalArrangement = Arrangement.spacedBy(UiTokens.sectionGap)) {
        item { Column(verticalArrangement = Arrangement.spacedBy(4.dp)) { Text("تنظیمات", style = MaterialTheme.typography.headlineMedium); Text("کنترل‌های واقعی برنامه و Runtime محلی", color = MaterialTheme.colorScheme.onSurfaceVariant) } }
        item { SettingsGroup("هوش مصنوعی", aiSections, section) { section = it } }
        item { SettingsGroup("برنامه", appSections, section) { section = it } }
        item { HorizontalDivider() }
        item { when (section) {
            "مدل" -> ModelManagement(active, models, error, onImport, onRefresh, onActivate, onDeactivate, onDelete)
            "استنتاج" -> InferenceControls()
            "زمینه" -> UnavailableSection("زمینه", "مدیریت System Context، Summary، پیام‌های اخیر و Workspace Context هنوز کنترل تغییرپذیر مستقلی در Core ندارد.")
            "عامل" -> UnavailableSection("عامل", "چرخه آماده‌سازی، تأیید، اجرا، شکست و Recovery از Action API واقعی کنترل می‌شود.")
            "فضای کار" -> UnavailableSection("فضای کار", "Workspace محل اجرای عملیات واقعی است؛ تنظیم مستقل قابل تغییر در Core فعلی تعریف نشده است.")
            "لاگ و عیب‌یابی" -> DiagnosticsSettingsInfo()
            "عملکرد" -> PerformanceSettingsInfo()
            "امنیت و تأیید" -> UnavailableSection("امنیت و تأیید", "تأیید Actionهای حساس فقط در وضعیت واقعی نیازمند تأیید فعال می‌شود.")
        } }
    }
}

@Composable
private fun InferenceControls() {
    var settings by remember { mutableStateOf(InferenceSettingsStore.current) }
    fun update(value: InferenceSettings) {
        settings = value
        InferenceSettingsStore.current = value
    }

    Column(verticalArrangement = Arrangement.spacedBy(UiTokens.itemGap)) {
        Text("استنتاج", style = MaterialTheme.typography.titleLarge)
        Text("این مقادیر مستقیماً برای Generation بعدی به Runtime محلی ارسال می‌شوند.", color = MaterialTheme.colorScheme.onSurfaceVariant)
        UiSurface {
            Column(Modifier.padding(UiTokens.compactPadding), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Text("Temperature: %.2f".format(settings.temperature))
                Slider(value = settings.temperature.toFloat(), onValueChange = { update(settings.copy(temperature = it.toDouble())) }, valueRange = 0f..2f)
                Text("Top-P: %.2f".format(settings.topP ?: 0.9))
                Slider(value = (settings.topP ?: 0.9).toFloat(), onValueChange = { update(settings.copy(topP = it.toDouble())) }, valueRange = 0f..1f)
                Text("Top-K: ${settings.topK ?: 40}")
                Slider(value = (settings.topK ?: 40).toFloat(), onValueChange = { update(settings.copy(topK = it.toInt())) }, valueRange = 0f..128f)
                Text("Min-P: %.2f".format(settings.minP ?: 0.0))
                Slider(value = (settings.minP ?: 0.0).toFloat(), onValueChange = { update(settings.copy(minP = it.toDouble())) }, valueRange = 0f..1f)
                Text("Repeat Penalty: %.2f".format(settings.repeatPenalty ?: 1.1))
                Slider(value = (settings.repeatPenalty ?: 1.1).toFloat(), onValueChange = { update(settings.copy(repeatPenalty = it.toDouble())) }, valueRange = 0.8f..2f)
                Text("Max New Tokens: ${settings.maxNewTokens}")
                Slider(value = settings.maxNewTokens.toFloat(), onValueChange = { update(settings.copy(maxNewTokens = it.toInt().coerceAtLeast(1))) }, valueRange = 64f..2048f, steps = 31)
                Text("Context Length: ${settings.contextLength ?: 4096}")
                Slider(value = (settings.contextLength ?: 4096).toFloat(), onValueChange = { update(settings.copy(contextLength = it.toInt().coerceAtLeast(256))) }, valueRange = 256f..8192f, steps = 31)
            }
        }
        Button(onClick = { update(InferenceSettings()) }, Modifier.fillMaxWidth().heightIn(min = UiTokens.minimumTouchTarget)) { Text("بازنشانی مقادیر پیش‌فرض") }
    }
}

@Composable
private fun DiagnosticsSettingsInfo() {
    UiSurface {
        Column(Modifier.padding(UiTokens.compactPadding), verticalArrangement = Arrangement.spacedBy(7.dp)) {
            Text("لاگ و عیب‌یابی", style = MaterialTheme.typography.titleLarge)
            Text("گزارش Runtime، TTFT، زمان تولید، تنظیمات، Runtime Trace و Action Trace از داده واقعی جمع می‌شوند.")
            Text("گزارش کامل و گزارش خطا از صفحه عیب‌یابی قابل کپی هستند.", color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}

@Composable
private fun PerformanceSettingsInfo() {
    UiSurface {
        Column(Modifier.padding(UiTokens.compactPadding), verticalArrangement = Arrangement.spacedBy(7.dp)) {
            Text("عملکرد", style = MaterialTheme.typography.titleLarge)
            Text("Performance بدون مقدار ساختگی از Runtime اندازه‌گیری می‌شود.")
            Text("Load Time، TTFT، Generation Time، Token Count و Tokens/sec در صورت ارائه واقعی Runtime نمایش داده می‌شوند.", color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}

@Composable private fun SettingsGroup(title: String, sections: List<String>, selected: String, onSelect: (String) -> Unit) {
    UiSurface { Column(Modifier.padding(vertical = 8.dp)) { Text(title, style = MaterialTheme.typography.titleMedium, modifier = Modifier.padding(horizontal = UiTokens.compactPadding, vertical = 8.dp)); sections.forEach { name -> TextButton(onClick = { onSelect(name) }, modifier = Modifier.fillMaxWidth().heightIn(min = UiTokens.minimumTouchTarget).semantics { contentDescription = if (selected == name) "$name، انتخاب شده" else name }, contentPadding = PaddingValues(horizontal = UiTokens.compactPadding, vertical = 8.dp)) { Text(if (selected == name) "● $name" else name, Modifier.fillMaxWidth()) } } } }
}

@Composable private fun ModelManagement(active: ModelDescriptor?, models: List<ModelDescriptor>, error: String?, onImport: () -> Unit, onRefresh: () -> Unit, onActivate: (String) -> Unit, onDeactivate: (() -> Unit)?, onDelete: (String) -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(UiTokens.itemGap)) { Text("مدل", style = MaterialTheme.typography.titleLarge); UiSurface { Column(Modifier.padding(UiTokens.compactPadding), verticalArrangement = Arrangement.spacedBy(4.dp)) { Text("مدل فعلی", style = MaterialTheme.typography.labelLarge); Text(active?.displayName ?: "مدلی فعال نیست", style = MaterialTheme.typography.titleMedium); Text(if (active != null) "آماده استفاده در گفت‌وگو" else "برای شروع، یک مدل محلی وارد و فعال کنید.", color = MaterialTheme.colorScheme.onSurfaceVariant); if (active != null && onDeactivate != null) OutlinedButton(onClick = onDeactivate, Modifier.fillMaxWidth().heightIn(min = UiTokens.minimumTouchTarget)) { Text("خارج‌کردن مدل از حافظه") } } }; Button(onClick = onImport, Modifier.fillMaxWidth().heightIn(min = UiTokens.minimumTouchTarget)) { Text("انتخاب فایل مدل از گوشی") }; Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(UiTokens.itemGap)) { Text("مدیریت مدل‌ها", style = MaterialTheme.typography.titleMedium, modifier = Modifier.weight(1f)); TextButton(onClick = onRefresh, Modifier.heightIn(min = UiTokens.minimumTouchTarget)) { Text("به‌روزرسانی") } }; if (models.isEmpty()) EmptyState() else models.forEach { model -> ModelCard(model, model.id == active?.id, onActivate, onDelete) }; if (error != null) ErrorSurface(error) }
}

@Composable private fun EmptyState() { UiSurface { Column(Modifier.padding(UiTokens.compactPadding), verticalArrangement = Arrangement.spacedBy(4.dp)) { Text("مدل محلی واردشده‌ای وجود ندارد.", style = MaterialTheme.typography.titleMedium); Text("از انتخاب فایل مدل برای واردکردن یک فایل GGUF استفاده کنید.", color = MaterialTheme.colorScheme.onSurfaceVariant) } } }
@Composable private fun ErrorSurface(message: String) { Surface(Modifier.fillMaxWidth(), shape = MaterialTheme.shapes.medium, color = MaterialTheme.colorScheme.errorContainer) { Text(message, Modifier.padding(UiTokens.compactPadding), color = MaterialTheme.colorScheme.onErrorContainer) } }
@Composable private fun ModelCard(model: ModelDescriptor, active: Boolean, onActivate: (String) -> Unit, onDelete: (String) -> Unit) { var expanded by remember { mutableStateOf(false) }; UiSurface { Column(Modifier.padding(UiTokens.compactPadding), verticalArrangement = Arrangement.spacedBy(7.dp)) { Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(UiTokens.itemGap)) { Column(Modifier.weight(1f)) { Text(model.displayName, style = MaterialTheme.typography.titleMedium); Text("GGUF · ${model.quantization}", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant) }; Text(if (active) "فعال" else "غیرفعال", style = MaterialTheme.typography.labelLarge) }; Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) { if (!active) OutlinedButton(onClick = { onActivate(model.id) }, Modifier.heightIn(min = UiTokens.minimumTouchTarget)) { Text("فعال‌سازی") }; TextButton(onClick = { expanded = !expanded }, Modifier.heightIn(min = UiTokens.minimumTouchTarget)) { Text(if (expanded) "بستن" else "جزئیات") }; TextButton(onClick = { onDelete(model.id) }, Modifier.heightIn(min = UiTokens.minimumTouchTarget), colors = ButtonDefaults.textButtonColors(contentColor = MaterialTheme.colorScheme.error)) { Text("حذف") } }; if (expanded) { HorizontalDivider(); Text("شناسه مدل: ${model.id}", style = MaterialTheme.typography.bodySmall); Text("وضعیت: ${model.state}", style = MaterialTheme.typography.bodySmall); Text("کمّیت‌سازی: ${model.quantization}", style = MaterialTheme.typography.bodySmall) } } } }
@Composable private fun UnavailableSection(title: String, message: String) { UiSurface { Column(Modifier.padding(UiTokens.compactPadding), verticalArrangement = Arrangement.spacedBy(7.dp)) { Text(title, style = MaterialTheme.typography.titleLarge); Text("اطلاعات فعلی", style = MaterialTheme.typography.labelLarge); Text(message, color = MaterialTheme.colorScheme.onSurfaceVariant) } } }
