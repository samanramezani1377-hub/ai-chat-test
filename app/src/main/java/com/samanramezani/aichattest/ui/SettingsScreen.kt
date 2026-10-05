package com.samanramezani.aichattest.ui

import android.content.Context
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
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

private val sections = listOf("مدل", "پاسخ", "زمینه", "عامل")

private enum class InferencePreset(val title: String, val description: String, val settings: InferenceSettings) {
    FAST("سریع", "برای گفت‌وگوی روزمره", InferenceSettings(temperature=.7, maxNewTokens=256, topK=32, topP=.9, minP=0.0, repeatPenalty=1.1, contextLength=2048, recentMessages=6, maxActionSteps=2)),
    BALANCED("استاندارد", "پیشنهاد من برای استفاده معمول", InferenceSettings(temperature=.7, maxNewTokens=512, topK=40, topP=.9, minP=0.0, repeatPenalty=1.1, contextLength=4096, recentMessages=10, maxActionSteps=4)),
    DEEP("عمیق", "پاسخ طولانی‌تر، مصرف بیشتر", InferenceSettings(temperature=.65, maxNewTokens=1024, topK=50, topP=.92, minP=0.0, repeatPenalty=1.1, contextLength=8192, recentMessages=20, maxActionSteps=6)),
}

@Composable
fun SettingsScreen(
    models: List<ModelDescriptor>, active: ModelDescriptor?, error: String?, onImport: () -> Unit,
    onRefresh: () -> Unit, onActivate: (String) -> Unit, onDeactivate: (() -> Unit)? = null, onDelete: (String) -> Unit,
) {
    var section by remember { mutableStateOf("مدل") }
    LazyColumn(
        Modifier.fillMaxSize().padding(horizontal = 16.dp),
        contentPadding = PaddingValues(top = 8.dp, bottom = 40.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp),
    ) {
        item {
            Column(verticalArrangement = Arrangement.spacedBy(5.dp)) {
                Text("تنظیمات", style = MaterialTheme.typography.headlineMedium)
                Text("اینجا فقط چیزهایی را می‌بینی که روی تجربه مدل اثر می‌گذارند.", color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
        item {
            Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                sections.forEach { name ->
                    FilterChip(
                        selected = section == name,
                        onClick = { section = name },
                        label = { Text(name) },
                    )
                }
            }
        }
        item {
            when (section) {
                "مدل" -> ModelManagement(active, models, error, onImport, onRefresh, onActivate, onDeactivate, onDelete)
                "پاسخ" -> InferenceControls()
                "زمینه" -> ContextControls()
                "عامل" -> AgentControls()
            }
        }
        item {
            Surface(Modifier.fillMaxWidth(), shape = RoundedCornerShape(18.dp), color = MaterialTheme.colorScheme.surfaceVariant) {
                Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text("معماری محلی", style = MaterialTheme.typography.titleSmall)
                    Text("مدل و inference روی دستگاه اجرا می‌شوند؛ این برنامه برای پاسخ‌گویی به API ابری متکی نیست.", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
        }
    }
}

@Composable
private fun ModelManagement(active: ModelDescriptor?, models: List<ModelDescriptor>, error: String?, onImport: () -> Unit, onRefresh: () -> Unit, onActivate: (String) -> Unit, onDeactivate: (() -> Unit)?, onDelete: (String) -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Surface(Modifier.fillMaxWidth(), shape = RoundedCornerShape(22.dp), color = MaterialTheme.colorScheme.primaryContainer) {
            Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(if (active != null) "مدل آماده است" else "اول یک مدل اضافه کن", style = MaterialTheme.typography.headlineSmall)
                Text(active?.displayName ?: "یک فایل GGUF از حافظه گوشی انتخاب کن.", color = MaterialTheme.colorScheme.onPrimaryContainer)
                if (active != null) {
                    Text("GGUF · " + active.quantization, style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onPrimaryContainer)
                    if (onDeactivate != null) OutlinedButton(onClick = onDeactivate) { Text("خارج‌کردن از حافظه") }
                }
            }
        }
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = onImport, modifier = Modifier.weight(1f).height(50.dp)) { Text("افزودن مدل") }
            OutlinedButton(onClick = onRefresh, modifier = Modifier.height(50.dp)) { Text("تازه‌سازی") }
        }
        if (error != null) ErrorCard(error)
        Text("مدل‌های روی دستگاه", style = MaterialTheme.typography.titleMedium)
        if (models.isEmpty()) {
            Surface(Modifier.fillMaxWidth(), shape = RoundedCornerShape(18.dp), color = MaterialTheme.colorScheme.surfaceVariant) {
                Text("هنوز فایل GGUF وارد نشده است.", Modifier.padding(16.dp), color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        } else {
            models.forEach { model -> ModelCard(model, model.id == active?.id, onActivate, onDelete) }
        }
    }
}

@Composable private fun ModelCard(model: ModelDescriptor, active: Boolean, onActivate: (String) -> Unit, onDelete: (String) -> Unit) {
    var details by remember { mutableStateOf(false) }
    Surface(Modifier.fillMaxWidth(), shape = RoundedCornerShape(18.dp), color = MaterialTheme.colorScheme.surface) {
        Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Row(Modifier.fillMaxWidth()) {
                Column(Modifier.weight(1f)) {
                    Text(model.displayName, style = MaterialTheme.typography.titleMedium)
                    Text("GGUF · " + model.quantization, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
                Text(if (active) "فعال" else "آماده", color = if (active) MaterialTheme.colorScheme.secondary else MaterialTheme.colorScheme.onSurfaceVariant, style = MaterialTheme.typography.labelMedium)
            }
            Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                if (!active) Button(onClick = { onActivate(model.id) }) { Text("فعال‌سازی") }
                TextButton(onClick = { details = !details }) { Text(if (details) "بستن" else "جزئیات") }
                TextButton(onClick = { onDelete(model.id) }, colors = ButtonDefaults.textButtonColors(contentColor = MaterialTheme.colorScheme.error)) { Text("حذف") }
            }
            if (details) {
                HorizontalDivider()
                Text("شناسه: " + model.id, style = MaterialTheme.typography.bodySmall)
                Text("وضعیت: " + model.state, style = MaterialTheme.typography.bodySmall)
            }
        }
    }
}

@Composable private fun ErrorCard(message: String) {
    Surface(Modifier.fillMaxWidth(), shape = RoundedCornerShape(16.dp), color = MaterialTheme.colorScheme.errorContainer) {
        Text(message, Modifier.padding(14.dp), color = MaterialTheme.colorScheme.onErrorContainer)
    }
}

@Composable private fun InferenceControls() {
    val context = androidx.compose.ui.platform.LocalContext.current
    var settings by remember { mutableStateOf(loadSettings(context)) }
    var dirty by remember { mutableStateOf(false) }
    fun update(v: InferenceSettings) { settings = v; dirty = true }
    fun apply() { saveSettings(context, settings); dirty = false }
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text("پاسخ", style = MaterialTheme.typography.titleLarge)
        Text("پروفایل آماده را انتخاب کن؛ اگر لازم شد بعداً تنظیمات دقیق را باز کن.", color = MaterialTheme.colorScheme.onSurfaceVariant)
        Surface(Modifier.fillMaxWidth(), shape = RoundedCornerShape(20.dp), color = MaterialTheme.colorScheme.surface) {
            Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                InferencePreset.values().forEach { preset ->
                    val selected = settings == preset.settings
                    Surface(
                        Modifier.fillMaxWidth().clickable { update(preset.settings) },
                        shape = RoundedCornerShape(14.dp),
                        color = if (selected) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.surfaceVariant,
                    ) {
                        Column(Modifier.padding(13.dp)) {
                            Text((if (selected) "✓ " else "") + preset.title, style = MaterialTheme.typography.titleSmall)
                            Text(preset.description, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                    }
                }
            }
        }
        var advanced by remember { mutableStateOf(false) }
        TextButton(onClick = { advanced = !advanced }) { Text(if (advanced) "بستن تنظیمات پیشرفته" else "تنظیمات پیشرفته") }
        if (advanced) AdvancedInference(settings, ::update)
        SaveRow(dirty, ::apply) { settings = InferenceSettings(); apply() }
    }
}

@Composable private fun AdvancedInference(settings: InferenceSettings, update: (InferenceSettings) -> Unit) {
    Surface(Modifier.fillMaxWidth(), shape = RoundedCornerShape(18.dp), color = MaterialTheme.colorScheme.surfaceVariant) {
        Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(5.dp)) {
            Text("Temperature: %.2f".format(settings.temperature))
            Slider(value = settings.temperature.toFloat(), onValueChange = { update(settings.copy(temperature = it.toDouble())) }, valueRange = 0f..2f)
            Text("Top-P: %.2f".format(settings.topP ?: .9))
            Slider(value = (settings.topP ?: .9).toFloat(), onValueChange = { update(settings.copy(topP = it.toDouble())) }, valueRange = 0f..1f)
            Text("Top-K: " + (settings.topK ?: 40))
            Slider(value = (settings.topK ?: 40).toFloat(), onValueChange = { update(settings.copy(topK = it.toInt())) }, valueRange = 0f..128f)
            Text("Repeat Penalty: %.2f".format(settings.repeatPenalty ?: 1.1))
            Slider(value = (settings.repeatPenalty ?: 1.1).toFloat(), onValueChange = { update(settings.copy(repeatPenalty = it.toDouble())) }, valueRange = .8f..2f)
            Text("Max New Tokens: " + settings.maxNewTokens)
            Slider(value = settings.maxNewTokens.toFloat(), onValueChange = { update(settings.copy(maxNewTokens = it.toInt().coerceAtLeast(1))) }, valueRange = 64f..2048f, steps = 31)
        }
    }
}

@Composable private fun ContextControls() {
    val context = androidx.compose.ui.platform.LocalContext.current
    var settings by remember { mutableStateOf(loadSettings(context)) }
    var dirty by remember { mutableStateOf(false) }
    fun apply() { saveSettings(context, settings); dirty = false }
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text("زمینه", style = MaterialTheme.typography.titleLarge)
        Surface(Modifier.fillMaxWidth(), shape = RoundedCornerShape(20.dp), color = MaterialTheme.colorScheme.surface) {
            Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(7.dp)) {
                Text("پیام‌های اخیر: " + settings.recentMessages)
                Slider(value = settings.recentMessages.toFloat(), onValueChange = { settings = settings.copy(recentMessages = it.toInt().coerceIn(0, 50)); dirty = true }, valueRange = 0f..50f, steps = 49)
                Text("بیشتر = حافظه مکالمه بیشتر و مصرف حافظه بالاتر.", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
        SaveRow(dirty, ::apply) { settings = InferenceSettings(); apply() }
    }
}

@Composable private fun AgentControls() {
    val context = androidx.compose.ui.platform.LocalContext.current
    var settings by remember { mutableStateOf(loadSettings(context)) }
    var dirty by remember { mutableStateOf(false) }
    fun apply() { saveSettings(context, settings); dirty = false }
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text("عامل", style = MaterialTheme.typography.titleLarge)
        Text("تعداد اقدام‌هایی که Agent می‌تواند پشت‌سرهم انجام دهد.", color = MaterialTheme.colorScheme.onSurfaceVariant)
        Surface(Modifier.fillMaxWidth(), shape = RoundedCornerShape(20.dp), color = MaterialTheme.colorScheme.surface) {
            Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(7.dp)) {
                Text("حداکثر اقدام: " + settings.maxActionSteps)
                Slider(value = settings.maxActionSteps.toFloat(), onValueChange = { settings = settings.copy(maxActionSteps = it.toInt().coerceIn(0, 16)); dirty = true }, valueRange = 0f..16f, steps = 15)
            }
        }
        SaveRow(dirty, ::apply) { settings = InferenceSettings(); apply() }
    }
}

@Composable private fun SaveRow(dirty: Boolean, apply: () -> Unit, reset: () -> Unit) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Button(onClick = apply, enabled = dirty, modifier = Modifier.weight(1f).height(48.dp)) { Text("ذخیره") }
        OutlinedButton(onClick = reset, modifier = Modifier.weight(1f).height(48.dp)) { Text("پیش‌فرض") }
    }
}

private fun loadSettings(context: Context): InferenceSettings {
    val p = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
    migrateSettings(p)
    val d = InferenceSettings()
    val loaded = d.copy(
        temperature = p.getFloat(KEY_TEMPERATURE, d.temperature.toFloat()).toDouble(),
        topP = if (p.contains(KEY_TOP_P)) p.getFloat(KEY_TOP_P, (d.topP ?: .9).toFloat()).toDouble() else d.topP,
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
    if (p.getInt(KEY_SCHEMA_VERSION, 0) < 2) {
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
