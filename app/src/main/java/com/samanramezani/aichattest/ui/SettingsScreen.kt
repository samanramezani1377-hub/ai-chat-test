package com.samanramezani.aichattest.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
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
        Modifier.fillMaxWidth().padding(UiTokens.pagePadding),
        contentPadding = PaddingValues(bottom = 32.dp),
        verticalArrangement = Arrangement.spacedBy(UiTokens.sectionGap),
    ) {
        item { Text("تنظیمات", style = MaterialTheme.typography.headlineMedium) }
        item { Text("هوش مصنوعی", style = MaterialTheme.typography.titleLarge) }
        item {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                listOf("مدل", "استنتاج", "زمینه", "عامل").forEach { name ->
                    SettingsTab(name, section == name) { section = name }
                }
            }
        }
        item { HorizontalDivider() }
        item { SettingsCategory("فضای کار", "مسیر نمایش اجرای جاری و Timeline در صفحه «فضای کار» قرار دارد.") }
        item { SettingsCategory("لاگ و عیب‌یابی", "گزارش‌های واقعی Execution و خطا از صفحه «عیب‌یابی» قابل مشاهده‌اند.") }
        item { SettingsCategory("عملکرد", "Metric فقط زمانی نمایش داده می‌شود که Runtime داده واقعی ارائه کند.") }
        item { SettingsCategory("امنیت و تأیید", "کنترل تأیید فقط برای عملیات واقعی که نیازمند تأیید هستند نمایش داده می‌شود.") }
        item { HorizontalDivider() }
        when (section) {
            "مدل" -> {
                item { UiSection("مدل") {
                    UiStatusRow("مدل فعلی", active?.displayName ?: "مدلی فعال نیست")
                    Button(
                        onClick = onImport,
                        modifier = Modifier.fillMaxWidth().heightIn(min = UiTokens.minimumTouchTarget),
                    ) { Text("بارگذاری مدل از گوشی") }
                    Text("مدیریت مدل‌ها", style = MaterialTheme.typography.titleMedium)
                } }
                if (models.isEmpty()) {
                    item { Text("مدل محلی واردشده‌ای وجود ندارد.", color = MaterialTheme.colorScheme.onSurfaceVariant) }
                }
                items(models, key = { it.id }) { model ->
                    ModelCard(model, model.id == active?.id, onActivate, onDelete)
                }
                item { TextButton(onClick = onRefresh, Modifier.fillMaxWidth().heightIn(min = UiTokens.minimumTouchTarget)) { Text("به‌روزرسانی فهرست مدل‌ها") } }
                if (error != null) item { Text(error, color = MaterialTheme.colorScheme.error) }
            }
            "استنتاج" -> item { UnavailableSection("استنتاج", "پارامترهای دما، Top-K، Top-P، Min-P، جریمه تکرار و سایر گزینه‌ها تا اتصال کنترل واقعی به Runtime، قابل ویرایش نیستند.") }
            "زمینه" -> item { UnavailableSection("زمینه", "مدیریت System Context، Summary و پیام‌های اخیر هنوز کنترل تغییرپذیر متصل به Core ندارد.") }
            "عامل" -> item { UnavailableSection("عامل", "کنترل آماده‌سازی، تأیید، اجرا و Recovery فقط با Action API واقعی ارائه می‌شود.") }
        }
    }
}

@Composable
private fun SettingsTab(text: String, selected: Boolean, onClick: () -> Unit) {
    TextButton(
        onClick = onClick,
        modifier = Modifier.heightIn(min = UiTokens.minimumTouchTarget).semantics { contentDescription = "تنظیمات $text" },
    ) { Text(if (selected) "● $text" else text) }
}

@Composable
private fun SettingsCategory(title: String, description: String) {
    UiSurface {
        Column(Modifier.padding(UiTokens.compactPadding), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(title, style = MaterialTheme.typography.titleMedium)
            Text(description, color = MaterialTheme.colorScheme.onSurfaceVariant, style = MaterialTheme.typography.bodySmall)
        }
    }
}

@Composable
private fun ModelCard(
    model: ModelDescriptor,
    active: Boolean,
    onActivate: (String) -> Unit,
    onDelete: (String) -> Unit,
) {
    var expanded by remember { mutableStateOf(false) }
    UiSurface {
        Column(Modifier.padding(UiTokens.compactPadding), verticalArrangement = Arrangement.spacedBy(7.dp)) {
            Text(model.displayName, style = MaterialTheme.typography.titleMedium)
            Text("GGUF · ${model.quantization}", style = MaterialTheme.typography.bodySmall)
            Text(
                if (active) "● فعال و آماده" else "وضعیت: ${model.state}",
                style = MaterialTheme.typography.labelMedium,
                color = if (active) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                if (!active) {
                    OutlinedButton(onClick = { onActivate(model.id) }, Modifier.heightIn(min = UiTokens.minimumTouchTarget)) { Text("فعال‌سازی") }
                }
                OutlinedButton(onClick = { expanded = !expanded }, Modifier.heightIn(min = UiTokens.minimumTouchTarget)) {
                    Text(if (expanded) "بستن جزئیات" else "جزئیات")
                }
                TextButton(
                    onClick = { onDelete(model.id) },
                    modifier = Modifier.heightIn(min = UiTokens.minimumTouchTarget),
                    colors = ButtonDefaults.textButtonColors(contentColor = MaterialTheme.colorScheme.error),
                ) { Text("حذف") }
            }
            if (expanded) {
                Text("شناسه مدل: ${model.id}", style = MaterialTheme.typography.bodySmall)
                Text("وضعیت: ${model.state}", style = MaterialTheme.typography.bodySmall)
                Text("کمّیت‌سازی: ${model.quantization}", style = MaterialTheme.typography.bodySmall)
            }
        }
    }
}

@Composable
private fun UnavailableSection(title: String, message: String) {
    UiSurface {
        Column(Modifier.padding(UiTokens.compactPadding), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(title, style = MaterialTheme.typography.titleMedium)
            Text("خارج از دسترس", style = MaterialTheme.typography.labelMedium)
            Text(message, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}
