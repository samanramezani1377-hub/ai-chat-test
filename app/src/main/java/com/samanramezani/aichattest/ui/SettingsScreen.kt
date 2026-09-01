package com.samanramezani.aichattest.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.woogit.aicore.domain.ModelDescriptor

@Composable
fun SettingsScreen(
    models: List<ModelDescriptor>,
    active: ModelDescriptor?,
    error: String?,
    onImport: () -> Unit,
    onRefresh: () -> Unit,
    onActivate: (String) -> Unit,
    onDelete: (String) -> Unit,
) {
    var section by remember { mutableStateOf("مدل") }
    LazyColumn(
        Modifier.fillMaxSize().padding(20.dp),
        contentPadding = PaddingValues(bottom = 28.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        item { Text("تنظیمات", style = MaterialTheme.typography.headlineMedium) }
        item { Text("هوش مصنوعی", style = MaterialTheme.typography.titleLarge) }
        item { SettingsTab("مدل", section == "مدل") { section = "مدل" } }
        item { SettingsTab("استنتاج", section == "استنتاج") { section = "استنتاج" } }
        item { SettingsTab("زمینه", section == "زمینه") { section = "زمینه" } }
        item { SettingsTab("عامل", section == "عامل") { section = "عامل" } }
        item { HorizontalDivider() }
        item { Text("فضای کار", style = MaterialTheme.typography.titleMedium) }
        item { Text("لاگ و عیب‌یابی", style = MaterialTheme.typography.titleMedium) }
        item { Text("عملکرد", style = MaterialTheme.typography.titleMedium) }
        item { Text("امنیت و تأیید", style = MaterialTheme.typography.titleMedium) }
        item { HorizontalDivider() }
        when (section) {
            "مدل" -> {
                item { Text("مدل فعلی", style = MaterialTheme.typography.titleMedium) }
                item { Text(active?.displayName ?: "مدلی فعال نیست", color = MaterialTheme.colorScheme.onSurfaceVariant) }
                item { Button(onClick = onImport, Modifier.fillMaxWidth()) { Text("انتخاب فایل مدل از گوشی") } }
                item { Text("مدیریت مدل‌ها", style = MaterialTheme.typography.titleMedium) }
                if (models.isEmpty()) item { Text("مدل محلی واردشده‌ای وجود ندارد.", color = MaterialTheme.colorScheme.onSurfaceVariant) }
                items(models, key = { it.id }) { model -> ModelCard(model, model.id == active?.id, onActivate, onDelete) }
                item { TextButton(onClick = onRefresh, Modifier.fillMaxWidth()) { Text("به‌روزرسانی") } }
                if (error != null) item { Text(error, color = MaterialTheme.colorScheme.error) }
            }
            "استنتاج" -> item { UnavailableSection("استنتاج", "تنظیمات استنتاج تا زمانی که API واقعی Runtime برای آن‌ها در دسترس نباشد، تغییرپذیر نیستند.") }
            "زمینه" -> item { UnavailableSection("زمینه", "تنظیمات زمینه فعلاً قابلیت قابل‌تغییر متصل به Core ندارند.") }
            "عامل" -> item { UnavailableSection("عامل", "تنظیمات عامل فعلاً قابلیت قابل‌تغییر متصل به Core ندارند.") }
        }
    }
}

@Composable private fun SettingsTab(text: String, selected: Boolean, onClick: () -> Unit) {
    TextButton(onClick = onClick, Modifier.fillMaxWidth()) { Text(if (selected) "● $text" else text, Modifier.fillMaxWidth()) }
}

@Composable private fun ModelCard(model: ModelDescriptor, active: Boolean, onActivate: (String) -> Unit, onDelete: (String) -> Unit) {
    var expanded by remember { mutableStateOf(false) }
    ElevatedCard(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(model.displayName, style = MaterialTheme.typography.titleMedium)
            Text("GGUF · ${model.quantization}")
            Text(if (active) "آماده و فعال" else "وضعیت: ${model.state}", style = MaterialTheme.typography.labelMedium)
            Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                if (!active) OutlinedButton(onClick = { onActivate(model.id) }) { Text("فعال‌سازی") }
                OutlinedButton(onClick = { expanded = !expanded }) { Text(if (expanded) "بستن جزئیات" else "جزئیات") }
                TextButton(onClick = { onDelete(model.id) }, colors = ButtonDefaults.textButtonColors(contentColor = MaterialTheme.colorScheme.error)) { Text("حذف") }
            }
            if (expanded) Text("شناسه مدل: ${model.id}\nوضعیت: ${model.state}\nکمّیت‌سازی: ${model.quantization}", style = MaterialTheme.typography.bodySmall)
        }
    }
}

@Composable private fun UnavailableSection(title: String, message: String) {
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) { Text(title, style = MaterialTheme.typography.titleMedium); Text("خارج از دسترس", style = MaterialTheme.typography.labelMedium); Text(message, color = MaterialTheme.colorScheme.onSurfaceVariant) }
}
