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
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import com.samanramezani.aichattest.ApiConfigStore
import com.woogit.aicore.domain.ApiProtocol
import com.woogit.aicore.domain.ApiProviderConfig
import com.woogit.aicore.domain.ApiProviderConfigStore
import com.woogit.aicore.domain.ApiProviderPresets
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

private val aiSections = listOf("اتصال مدل", "استنتاج", "زمینه", "عامل")
private val appSections = listOf("فضای کار", "لاگ و عیب‌یابی", "عملکرد", "امنیت و تأیید")

@Composable
fun SettingsScreen(
    models: List<ModelDescriptor>, active: ModelDescriptor?, error: String?, onImport: () -> Unit,
    onRefresh: () -> Unit, onActivate: (String) -> Unit, onDeactivate: (() -> Unit)? = null, onDelete: (String) -> Unit,
) {
    var section by remember { mutableStateOf("اتصال مدل") }
    LazyColumn(
        Modifier.fillMaxSize().padding(horizontal = UiTokens.pagePadding),
        contentPadding = PaddingValues(top = UiTokens.sectionGap, bottom = 36.dp),
        verticalArrangement = Arrangement.spacedBy(UiTokens.sectionGap),
    ) {
        item {
            Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text("تنظیمات", style = MaterialTheme.typography.headlineMedium)
                Text("اتصال API، استنتاج، زمینه و عامل", color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
        item { SettingsGroup("هوش مصنوعی", aiSections, section) { section = it } }
        item { SettingsGroup("برنامه", appSections, section) { section = it } }
        item { HorizontalDivider() }
        item {
            when (section) {
                "اتصال مدل" -> ApiProviderSettings()
                "استنتاج" -> InferenceControls()
                "زمینه" -> ContextControls()
                "عامل" -> AgentControls()
                "فضای کار" -> SettingsInfo("فضای کار", "Workspace محیط اجرای واقعی Actionهاست. تنظیمات مسیر یا دسترسی عمومی قابل تغییر نیست تا مرز امنیتی Workspace شکسته نشود.")
                "لاگ و عیب‌یابی" -> SettingsInfo("لاگ و عیب‌یابی", "گزارش Runtime، TTFT، زمان تولید، تنظیمات، Runtime Trace و Action Trace از داده واقعی جمع می‌شوند. گزارش کامل و گزارش خطا از صفحه عیب‌یابی قابل کپی هستند.")
                "عملکرد" -> SettingsInfo("عملکرد", "Performance بدون مقدار ساختگی از Runtime اندازه‌گیری می‌شود. برای تشخیص TTFT و سرعت تولید، صفحه عیب‌یابی آخرین metrics واقعی را نمایش می‌دهد.")
                "امنیت و تأیید" -> SettingsInfo("امنیت و تأیید", "Actionهای حساس قبل از اجرا نیازمند تأیید هستند. این سیاست بخشی از مسیر واقعی Agent است و برای جلوگیری از دور زدن کنترل امنیتی، خاموش‌کردن عمومی آن ارائه نشده است.")
            }
        }
    }
}

@Composable
private fun ApiProviderSettings() {
    val context = LocalContext.current
    var config by remember { mutableStateOf(ApiConfigStore(context).load()) }
    var saved by remember { mutableStateOf(false) }
    var showKey by remember { mutableStateOf(false) }

    fun applyPreset(preset: ApiProviderConfig) {
        config = preset.copy(apiKey = config.apiKey)
        saved = false
    }

    Column(verticalArrangement = Arrangement.spacedBy(UiTokens.itemGap)) {
        Text("اتصال مدل", style = MaterialTheme.typography.titleLarge)
        Text(
            "مدل محلی و Import فایل GGUF از مسیر Generation حذف شده‌اند. از اینجا provider و API خود را به Runtime متصل کنید.",
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        UiSurface {
            Column(Modifier.padding(UiTokens.compactPadding), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("Provider سریع", style = MaterialTheme.typography.titleMedium)
                Text("یک preset را انتخاب کنید؛ سپس مدل و کلید را بررسی یا اصلاح کنید.", color = MaterialTheme.colorScheme.onSurfaceVariant)
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedButton(onClick = { applyPreset(ApiProviderPresets.deepSeek) }, Modifier.weight(1f)) { Text("DeepSeek") }
                    OutlinedButton(onClick = { applyPreset(ApiProviderPresets.openAi) }, Modifier.weight(1f)) { Text("OpenAI") }
                }
                Text("برای providerهای سازگار با OpenAI یا Anthropic، Base URL و Protocol را دستی تنظیم کنید.", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }

        UiSurface {
            Column(Modifier.padding(UiTokens.compactPadding), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedTextField(
                    value = config.providerId,
                    onValueChange = { config = config.copy(providerId = it); saved = false },
                    modifier = Modifier.fillMaxWidth(),
                    label = { Text("Provider ID") },
                    singleLine = true,
                )
                OutlinedTextField(
                    value = config.baseUrl,
                    onValueChange = { config = config.copy(baseUrl = it); saved = false },
                    modifier = Modifier.fillMaxWidth(),
                    label = { Text("Base URL") },
                    supportingText = { Text("مثال: https://api.deepseek.com") },
                    singleLine = true,
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri),
                )
                OutlinedTextField(
                    value = config.model,
                    onValueChange = { config = config.copy(model = it); saved = false },
                    modifier = Modifier.fillMaxWidth(),
                    label = { Text("Model") },
                    singleLine = true,
                )
                OutlinedTextField(
                    value = config.apiKey,
                    onValueChange = { config = config.copy(apiKey = it); saved = false },
                    modifier = Modifier.fillMaxWidth(),
                    label = { Text("API Key") },
                    singleLine = true,
                    visualTransformation = if (showKey) androidx.compose.ui.text.input.VisualTransformation.None else PasswordVisualTransformation(),
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password),
                    trailingIcon = { TextButton(onClick = { showKey = !showKey }) { Text(if (showKey) "پنهان" else "نمایش") } },
                )
            }
        }

        UiSurface {
            Column(Modifier.padding(UiTokens.compactPadding), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("Protocol", style = MaterialTheme.typography.titleMedium)
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    FilterChip(
                        selected = config.protocol == ApiProtocol.OPENAI_CHAT,
                        onClick = { config = config.copy(protocol = ApiProtocol.OPENAI_CHAT); saved = false },
                        label = { Text("OpenAI Chat") },
                    )
                    FilterChip(
                        selected = config.protocol == ApiProtocol.ANTHROPIC_MESSAGES,
                        onClick = { config = config.copy(protocol = ApiProtocol.ANTHROPIC_MESSAGES); saved = false },
                        label = { Text("Anthropic") },
                    )
                }
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    FilterChip(
                        selected = config.thinkingEnabled,
                        onClick = { config = config.copy(thinkingEnabled = !config.thinkingEnabled); saved = false },
                        label = { Text("Thinking") },
                    )
                    Text("Reasoning: ${config.reasoningEffort ?: "provider default"}", Modifier.padding(top = 8.dp), color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
        }

        Button(
            onClick = {
                val normalized = config.copy(providerId = config.providerId.trim(), baseUrl = config.normalizedBaseUrl(), model = config.model.trim(), apiKey = config.apiKey.trim())
                ApiConfigStore(context).save(normalized)
                ApiProviderConfigStore.current = normalized
                config = normalized
                saved = true
            },
            modifier = Modifier.fillMaxWidth().heightIn(min = UiTokens.minimumTouchTarget),
            enabled = config.baseUrl.isNotBlank() && config.model.isNotBlank() && config.apiKey.isNotBlank(),
        ) { Text("ذخیره و اتصال") }
        Text(
            if (saved) "تنظیمات API ذخیره شد و Runtime از اتصال جدید استفاده می‌کند." else "کلید API در حافظه برنامه و فضای ذخیره‌سازی رمزنگاری‌شده نگهداری می‌شود و در سورس پروژه ذخیره نمی‌شود.",
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
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
        Text("تنظیمات Generation برای درخواست‌های بعدی.", color = MaterialTheme.colorScheme.onSurfaceVariant)
        UiSurface {
            Column(Modifier.padding(UiTokens.compactPadding), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Text("Temperature: %.2f".format(settings.temperature))
                Slider(value = settings.temperature.toFloat(), onValueChange = { update(settings.copy(temperature = it.toDouble())) }, valueRange = 0f..2f)
                Text("Top-P: %.2f".format(settings.topP ?: 0.9))
                Slider(value = (settings.topP ?: 0.9).toFloat(), onValueChange = { update(settings.copy(topP = it.toDouble())) }, valueRange = 0f..1f)
                Text("Max New Tokens: ${settings.maxNewTokens}")
                Slider(value = settings.maxNewTokens.toFloat(), onValueChange = { update(settings.copy(maxNewTokens = it.toInt().coerceAtLeast(1))) }, valueRange = 64f..2048f, steps = 31)
            }
        }
        Button(onClick = ::apply, Modifier.fillMaxWidth().heightIn(min = UiTokens.minimumTouchTarget), enabled = dirty) { Text("اعمال تنظیمات") }
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
        Text("تعداد پیام‌های اخیر برای Context درخواست بعدی.", color = MaterialTheme.colorScheme.onSurfaceVariant)
        UiSurface {
            Column(Modifier.padding(UiTokens.compactPadding), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Text("Recent Messages: ${settings.recentMessages}")
                Slider(value = settings.recentMessages.toFloat(), onValueChange = { update(settings.copy(recentMessages = it.toInt().coerceIn(0, 50))) }, valueRange = 0f..50f, steps = 49)
            }
        }
        Button(onClick = ::apply, Modifier.fillMaxWidth().heightIn(min = UiTokens.minimumTouchTarget), enabled = dirty) { Text("اعمال تنظیمات زمینه") }
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
        Text("حداکثر تعداد Action واقعی برای هر درخواست.", color = MaterialTheme.colorScheme.onSurfaceVariant)
        UiSurface {
            Column(Modifier.padding(UiTokens.compactPadding), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Text("Max Action Steps: ${settings.maxActionSteps}")
                Slider(value = settings.maxActionSteps.toFloat(), onValueChange = { update(settings.copy(maxActionSteps = it.toInt().coerceIn(0, 16))) }, valueRange = 0f..16f, steps = 15)
            }
        }
        Button(onClick = ::apply, Modifier.fillMaxWidth().heightIn(min = UiTokens.minimumTouchTarget), enabled = dirty) { Text("اعمال تنظیمات عامل") }
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

@Composable
private fun SettingsGroup(title: String, sections: List<String>, selected: String, onSelect: (String) -> Unit) {
    UiSurface {
        Column(Modifier.padding(vertical = 8.dp)) {
            Text(title, style = MaterialTheme.typography.titleMedium, modifier = Modifier.padding(horizontal = UiTokens.compactPadding, vertical = 8.dp))
            sections.forEach { name ->
                TextButton(
                    onClick = { onSelect(name) },
                    modifier = Modifier.fillMaxWidth().heightIn(min = UiTokens.minimumTouchTarget).semantics { contentDescription = if (selected == name) "$name، انتخاب شده" else name },
                    contentPadding = PaddingValues(horizontal = UiTokens.compactPadding, vertical = 8.dp),
                ) { Text(if (selected == name) "● $name" else name, Modifier.fillMaxWidth()) }
            }
        }
    }
}
