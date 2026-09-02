package com.samanramezani.aichattest.ui

import android.content.Context
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import com.woogit.aicore.domain.InferenceSettings
import com.woogit.aicore.domain.InferenceSettingsStore
import com.woogit.aicore.domain.ModelDescriptor

private const val PREFS = "inference_settings"
private const val KEY_SCHEMA_VERSION = "schema_version"
private const val CURRENT_SCHEMA_VERSION = 2
private const val KEY_TEMPERATURE = "temperature"
private const val KEY_TOP_P = "top_p"
private const val KEY_TOP_K = "top_k"
private const val KEY_MIN_P = "min_p"
private const val KEY_REPEAT = "repeat_penalty"
private const val KEY_MAX_TOKENS = "max_new_tokens"
private const val KEY_CONTEXT = "context_length"
private const val KEY_RECENT_MESSAGES = "recent_messages"
private const val KEY_SEED = "seed"
private const val KEY_STOPS = "stop_sequences"
private const val KEY_MAX_ACTION_STEPS = "max_action_steps"

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
            "زمینه" -> ContextControls()
            "عامل" -> AgentControls()
            "فضای کار" -> SettingsInfo("فضای کار", "Workspace محیط اجرای واقعی Actionهاست. تنظیمات مسیر یا دسترسی عمومی قابل تغییر نیست تا مرز امنیتی Workspace شکسته نشود.")
            "لاگ و عیب‌یابی" -> SettingsInfo("لاگ و عیب‌یابی", "گزارش Runtime، TTFT، زمان تولید، تنظیمات، Runtime Trace و Action Trace از داده واقعی جمع می‌شوند. گزارش کامل و گزارش خطا از صفحه عیب‌یابی قابل کپی هستند.")
            "عملکرد" -> SettingsInfo("عملکرد", "Performance بدون مقدار ساختگی از Runtime اندازه‌گیری می‌شود. برای تشخیص TTFT و سرعت تولید، صفحه عیب‌یابی آخرین metrics واقعی را نمایش می‌دهد.")
            "امنیت و تأیید" -> SettingsInfo("امنیت و تأیید", "Actionهای حساس قبل از اجرا نیازمند تأیید هستند. این سیاست بخشی از مسیر واقعی Agent است و برای جلوگیری از دور زدن کنترل امنیتی، خاموش‌کردن عمومی آن ارائه نشده است.")
        } }
    }
}

@Composable
private fun InferenceControls() {
    val context = LocalContext.current
    var settings by remember { mutableStateOf(loadSettings(context)) }
    var dirty by remember { mutableStateOf(false) }
    fun update(value: InferenceSettings) { settings = value; dirty = true }
    fun apply() { saveSettings(context, settings); dirty = false }

    Column(verticalArrangement = Arrangement.spacedBy(UiTokens.itemGap)) {
        Text("استنتاج", style = MaterialTheme.typography.titleLarge)
        Text("این مقادیر مستقیماً برای Generation بعدی به Runtime محلی ارسال می‌شوند و روی دستگاه ذخیره می‌شوند.", color = MaterialTheme.colorScheme.onSurfaceVariant)
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
                Text("Seed: ${settings.seed ?: "تصادفی"}")
            }
        }
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(UiTokens.itemGap)) {
            Button(onClick = ::apply, Modifier.weight(1f).heightIn(min = UiTokens.minimumTouchTarget), enabled = dirty) { Text("اعمال تنظیمات") }
            OutlinedButton(onClick = { settings = InferenceSettings(); apply() }, Modifier.weight(1f).heightIn(min = UiTokens.minimumTouchTarget)) { Text("بازنشانی") }
        }
        Text(if (dirty) "تغییرات ذخیره نشده‌اند." else "تنظیمات ذخیره و برای Generation بعدی آماده‌اند.", color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@Composable
private fun ContextControls() {
    val context = LocalContext.current
    var settings by remember { mutableStateOf(loadSettings(context)) }
    var dirty by remember { mutableStateOf(false) }
    fun update(value: InferenceSettings) { settings = value; dirty = true }
    fun apply() { saveSettings(context, settings); dirty = false }

    Column(verticalArrangement = Arrangement.spacedBy(UiTokens.itemGap)) {
        Text("زمینه", style = MaterialTheme.typography.titleLarge)
        Text("تعداد پیام‌های اخیر که Context Builder برای درخواست بعدی در نظر می‌گیرد قابل تنظیم است. Summary، Persistent Task Context و Workspace Context همچنان مالکیت جداگانه در Core دارند.", color = MaterialTheme.colorScheme.onSurfaceVariant)
        UiSurface {
            Column(Modifier.padding(UiTokens.compactPadding), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Text("Recent Messages: ${settings.recentMessages}")
                Slider(value = settings.recentMessages.toFloat(), onValueChange = { update(settings.copy(recentMessages = it.toInt().coerceIn(0, 50))) }, valueRange = 0f..50f, steps = 49)
                Text("پیش‌فرض: ۱۰ · بازه قابل تنظیم: ۰ تا ۵۰", color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
        Button(onClick = ::apply, Modifier.fillMaxWidth().heightIn(min = UiTokens.minimumTouchTarget), enabled = dirty) { Text("اعمال تنظیمات زمینه") }
        Text(if (dirty) "تغییرات ذخیره نشده‌اند." else "تنظیمات زمینه ذخیره و برای Context بعدی آماده‌اند.", color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@Composable
private fun AgentControls() {
    val context = LocalContext.current
    var settings by remember { mutableStateOf(loadSettings(context)) }
    var dirty by remember { mutableStateOf(false) }
    fun update(value: InferenceSettings) { settings = value; dirty = true }
    fun apply() { saveSettings(context, settings); dirty = false }

    Column(verticalArrangement = Arrangement.spacedBy(UiTokens.itemGap)) {
        Text("عامل", style = MaterialTheme.typography.titleLarge)
        Text("حداکثر تعداد Action واقعی که Agent برای هر درخواست می‌تواند اجرا کند. مقدار پیش‌فرض ۴ است و فقط برای جلوگیری از چرخه‌های بی‌نهایت استفاده می‌شود.", color = MaterialTheme.colorScheme.onSurfaceVariant)
        UiSurface {
            Column(Modifier.padding(UiTokens.compactPadding), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Text("Max Action Steps: ${settings.maxActionSteps}")
                Slider(value = settings.maxActionSteps.toFloat(), onValueChange = { update(settings.copy(maxActionSteps = it.toInt().coerceIn(0, 16))) }, valueRange = 0f..16f, steps = 15)
                Text("پیش‌فرض: ۴ · بازه قابل تنظیم: ۰ تا ۱۶", color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
        Button(onClick = ::apply, Modifier.fillMaxWidth().heightIn(min = UiTokens.minimumTouchTarget), enabled = dirty) { Text("اعمال تنظیمات عامل") }
        Text(if (dirty) "تغییرات ذخیره نشده‌اند." else "تنظیمات عامل ذخیره و برای اجرای بعدی آماده‌اند.", color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

private fun loadSettings(context: Context): InferenceSettings {
    val p = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
    migrateSettings(p)
    val d = InferenceSettings()
    val loaded = d.copy(
        temperature = p.getFloat(KEY_TEMPERATURE, d.temperature.toFloat()).toDouble(),
        topP = if (p.contains(KEY_TOP_P)) p.getFloat(KEY_TOP_P, (d.topP ?: 0.9).toFloat()).toDouble() else d.topP,
        topK = if (p.contains(KEY_TOP_K)) p.getInt(KEY_TOP_K, d.topK ?: 40) else d.topK,
        minP = if (p.contains(KEY_MIN_P)) p.getFloat(KEY_MIN_P, (d.minP ?: 0.0).toFloat()).toDouble() else d.minP,
        repeatPenalty = if (p.contains(KEY_REPEAT)) p.getFloat(KEY_REPEAT, (d.repeatPenalty ?: 1.1).toFloat()).toDouble() else d.repeatPenalty,
        maxNewTokens = p.getInt(KEY_MAX_TOKENS, d.maxNewTokens),
        contextLength = if (p.contains(KEY_CONTEXT)) p.getInt(KEY_CONTEXT, d.contextLength ?: 4096) else d.contextLength,
        recentMessages = p.getInt(KEY_RECENT_MESSAGES, d.recentMessages).coerceAtLeast(0),
        seed = if (p.contains(KEY_SEED)) p.getLong(KEY_SEED, d.seed ?: 0L) else d.seed,
        stopSequences = p.getString(KEY_STOPS, null)?.split("\u001f")?.filter { it.isNotEmpty() } ?: d.stopSequences,
        maxActionSteps = p.getInt(KEY_MAX_ACTION_STEPS, d.maxActionSteps).coerceAtLeast(0),
    )
    InferenceSettingsStore.current = loaded
    return loaded
}

private fun migrateSettings(p: android.content.SharedPreferences) {
    val version = p.getInt(KEY_SCHEMA_VERSION, 0)
    if (version < 2) {
        if (!p.contains(KEY_RECENT_MESSAGES)) p.edit().putInt(KEY_RECENT_MESSAGES, 10).apply()
        p.edit().putInt(KEY_SCHEMA_VERSION, CURRENT_SCHEMA_VERSION).apply()
    }
}

private fun saveSettings(context: Context, settings: InferenceSettings) {
    context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
        .putInt(KEY_SCHEMA_VERSION, CURRENT_SCHEMA_VERSION)
        .putFloat(KEY_TEMPERATURE, settings.temperature.toFloat())
        .apply { settings.topP?.let { putFloat(KEY_TOP_P, it.toFloat()) } ?: remove(KEY_TOP_P) }
        .apply { settings.topK?.let { putInt(KEY_TOP_K, it) } ?: remove(KEY_TOP_K) }
        .apply { settings.minP?.let { putFloat(KEY_MIN_P, it.toFloat()) } ?: remove(KEY_MIN_P) }
        .apply { settings.repeatPenalty?.let { putFloat(KEY_REPEAT, it.toFloat()) } ?: remove(KEY_REPEAT) }
        .putInt(KEY_MAX_TOKENS, settings.maxNewTokens)
        .apply { settings.contextLength?.let { putInt(KEY_CONTEXT, it) } ?: remove(KEY_CONTEXT) }
        .putInt(KEY_RECENT_MESSAGES, settings.recentMessages.coerceAtLeast(0))
        .apply { settings.seed?.let { putLong(KEY_SEED, it) } ?: remove(KEY_SEED) }
        .putString(KEY_STOPS, settings.stopSequences.joinToString("\u001f"))
        .putInt(KEY_MAX_ACTION_STEPS, settings.maxActionSteps.coerceAtLeast(0))
        .apply()
    InferenceSettingsStore.current = settings
}

@Composable
private fun SettingsInfo(title: String, message: String) {
    UiSurface { Column(Modifier.padding(UiTokens.compactPadding), verticalArrangement = Arrangement.spacedBy(7.dp)) { Text(title, style = MaterialTheme.typography.titleLarge); Text(message, color = MaterialTheme.colorScheme.onSurfaceVariant) } }
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
