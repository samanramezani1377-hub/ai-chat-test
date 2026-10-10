package com.samanramezani.aichattest.ui.voice

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.samanramezani.aichattest.voice.VoiceModelEntry

data class VoiceLine(val id: Long, val speaker: String, val text: String)

@Composable
internal fun VoiceConversationPage(
    lines: List<VoiceLine>,
    status: String,
    diagnosticLogs: List<String>,
    listening: Boolean,
    busy: Boolean,
    sttModels: List<VoiceModelEntry>,
    ttsModels: List<VoiceModelEntry>,
    missingSttComponents: List<String>,
    missingKoochikComponents: List<String>,
    missingTtsComponents: List<String>,
    activeSttId: String?,
    activeTtsId: String?,
    onBack: () -> Unit,
    onImportStt: () -> Unit,
    onImportSmallSttModel: () -> Unit,
    onImportSmallSttTokens: () -> Unit,
    onImportKoochikModel: () -> Unit,
    onImportKoochikTokens: () -> Unit,
    onImportTts: () -> Unit,
    onImportTtsModel: () -> Unit,
    onImportTtsConfig: () -> Unit,
    onImportEspeakData: () -> Unit,
    onSelectStt: (String) -> Unit,
    onSelectTts: (String) -> Unit,
    onStart: () -> Unit,
    onStop: () -> Unit,
    onInterrupt: () -> Unit,
) {
    val context = LocalContext.current
    val uriHandler = LocalUriHandler.current
    val clipboard = LocalClipboardManager.current
    var showSettings by remember { mutableStateOf(false) }
    var permissionGranted by remember {
        mutableStateOf(ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED)
    }
    val permission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        permissionGranted = granted
        if (granted) onStart()
    }

    Column(Modifier.fillMaxSize().padding(horizontal = 16.dp, vertical = 8.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
            IconButton(onClick = onBack) { Icon(Icons.Default.ArrowBack, contentDescription = "بازگشت") }
            Column(Modifier.weight(1f)) {
                Text("مکالمه صوتی زنده", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
                Text("گفت‌وگوی محلی با امکان صحبت هم‌زمان و قطع پاسخ", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.secondary)
            }
            IconButton(onClick = { showSettings = !showSettings }) {
                Icon(Icons.Default.Settings, contentDescription = "تنظیمات مکالمه صوتی")
            }
        }

        if (showSettings) {
            Column(
                Modifier.weight(1f, fill = false).fillMaxWidth().verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
                    Text("تنظیمات مکالمه صوتی", style = MaterialTheme.typography.headlineSmall, modifier = Modifier.weight(1f))
                    TextButton(onClick = { showSettings = false }) { Text("بازگشت به مکالمه") }
                }
                Card(Modifier.fillMaxWidth()) {
                    Column(Modifier.fillMaxWidth().padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text("۱. تشخیص گفتار · STT", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                        Text("مدل STT صدای شما را به متن تبدیل می‌کند. برای Shenava هر دو فایل model.onnx و tokens.txt لازم‌اند.", style = MaterialTheme.typography.bodySmall)
                        OutlinedButton(
                            onClick = { uriHandler.openUri("https://huggingface.co/Reza2kn/Shenava-Rizeh-Pizeh-v1.0-sherpa-onnx") },
                            modifier = Modifier.fillMaxWidth(),
                        ) { Text("صفحه دریافت مدل سبک فارسی Shenava") }
                        OutlinedButton(onClick = onImportSmallSttModel, enabled = !busy, modifier = Modifier.fillMaxWidth()) { Text("وارد کردن مدل فارسی · model.onnx") }
                        OutlinedButton(onClick = onImportSmallSttTokens, enabled = !busy, modifier = Modifier.fillMaxWidth()) { Text("وارد کردن واژگان · tokens.txt") }
                        OutlinedButton(onClick = onImportStt, enabled = !busy, modifier = Modifier.fillMaxWidth()) { Text("یا وارد کردن بسته کامل STT · ZIP / TAR.BZ2") }
                        Text("فایل‌ها را یکی‌یکی وارد کن؛ برنامه آن‌ها را نگه می‌دارد و پس از کامل‌شدن، یک مدل واحد می‌سازد.", style = MaterialTheme.typography.bodySmall)
                        val sttSteps = listOf("model.onnx", "tokens.txt")
                        val sttDone = sttSteps.size - missingSttComponents.size
                        LinearProgressIndicator(progress = { sttDone.toFloat() / sttSteps.size }, modifier = Modifier.fillMaxWidth())
                        sttSteps.forEach { step ->
                            Text("${if (step !in missingSttComponents) "✓" else "○"}  $step", style = MaterialTheme.typography.bodySmall)
                        }
                        Text(if (sttDone == sttSteps.size) "هر دو فایل Shenava Rizeh-Pizeh وارد شده‌اند." else "پیشرفت ورود Rizeh-Pizeh: $sttDone از ${sttSteps.size} فایل", style = MaterialTheme.typography.labelMedium)
                        Spacer(Modifier.height(8.dp))
                        Text("Shenava Koochik · دقت بالاتر · حدود ۴۵۹ مگابایت", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
                        Text("این مدل بزرگ‌تر از Rizeh-Pizeh است و برای تشخیص دقیق‌تر فارسی در نظر گرفته شده. دو فایل زیر را جداگانه دریافت و وارد کن.", style = MaterialTheme.typography.bodySmall)
                        OutlinedButton(
                            onClick = { uriHandler.openUri("https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0-sherpa-onnx/resolve/main/model.onnx") },
                            modifier = Modifier.fillMaxWidth(),
                        ) { Text("دانلود مدل Koochik · model.onnx") }
                        OutlinedButton(
                            onClick = { uriHandler.openUri("https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0-sherpa-onnx/resolve/main/tokens.txt") },
                            modifier = Modifier.fillMaxWidth(),
                        ) { Text("دانلود واژگان Koochik · tokens.txt") }
                        OutlinedButton(onClick = onImportKoochikModel, enabled = !busy, modifier = Modifier.fillMaxWidth()) { Text("وارد کردن مدل Koochik · model.onnx") }
                        OutlinedButton(onClick = onImportKoochikTokens, enabled = !busy, modifier = Modifier.fillMaxWidth()) { Text("وارد کردن واژگان Koochik · tokens.txt") }
                        val koochikSteps = listOf("model.onnx", "tokens.txt")
                        val koochikDone = koochikSteps.size - missingKoochikComponents.size
                        LinearProgressIndicator(progress = { koochikDone.toFloat() / koochikSteps.size }, modifier = Modifier.fillMaxWidth())
                        koochikSteps.forEach { step ->
                            Text("${if (step !in missingKoochikComponents) "✓" else "○"}  $step", style = MaterialTheme.typography.bodySmall)
                        }
                        Text(if (koochikDone == koochikSteps.size) "هر دو فایل Koochik وارد شده‌اند؛ مدل آماده است." else "پیشرفت ورود Koochik: $koochikDone از ${koochikSteps.size} فایل", style = MaterialTheme.typography.labelMedium)
                        if (sttModels.isEmpty()) Text("مدل تشخیص گفتار واردشده و آماده‌ای وجود ندارد.", color = MaterialTheme.colorScheme.error)
                        sttModels.forEach { model ->
                            Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
                                RadioButton(selected = model.id == activeSttId, onClick = { onSelectStt(model.id) })
                                Column(Modifier.weight(1f)) {
                                    Text(model.title, fontWeight = FontWeight.Medium)
                                    Text(when (model.asrMode) {
                                        "qwen3-asr" -> "Qwen3-ASR · محلی"
                                        "nemo-ctc" -> if (model.title.contains("Koochik", true)) "Shenava Koochik CTC · فارسی · 114M" else "Shenava Rizeh-Pizeh CTC · فارسی · 6.9M"
                                        else -> "Streaming Transducer · محلی"
                                    }, style = MaterialTheme.typography.bodySmall)
                                }
                            }
                        }
                    }
                }
                Card(Modifier.fillMaxWidth()) {
                    Column(Modifier.fillMaxWidth().padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text("۲. تولید گفتار · TTS", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                        Text("مدل TTS پاسخ دستیار را به صدا تبدیل می‌کند. مدل Piper آماده‌شده باید فایل مدل، تنظیمات و داده‌های آواشناسی را داشته باشد.", style = MaterialTheme.typography.bodySmall)
                        OutlinedButton(onClick = onImportTtsModel, enabled = !busy, modifier = Modifier.fillMaxWidth()) { Text("وارد کردن مدل صوتی · ONNX") }
                        OutlinedButton(onClick = onImportTtsConfig, enabled = !busy, modifier = Modifier.fillMaxWidth()) { Text("وارد کردن تنظیمات Piper · JSON") }
                        OutlinedButton(onClick = onImportEspeakData, enabled = !busy, modifier = Modifier.fillMaxWidth()) { Text("وارد کردن espeak-ng-data · TAR.BZ2") }
                        OutlinedButton(onClick = onImportTts, enabled = !busy, modifier = Modifier.fillMaxWidth()) { Text("یا وارد کردن بسته کامل TTS · ZIP / TAR.BZ2") }
                        Text("سه جزء را جداگانه و به هر ترتیبی وارد کن؛ پس از دریافت هر سه، برنامه آن‌ها را خودکار ترکیب و اعتبارسنجی می‌کند.", style = MaterialTheme.typography.bodySmall)
                        val ttsSteps = listOf("فایل ONNX", "فایل JSON کنار مدل (معمولاً model.onnx.json)", "بسته espeak-ng-data")
                        val ttsDone = ttsSteps.size - missingTtsComponents.size
                        LinearProgressIndicator(progress = { ttsDone.toFloat() / ttsSteps.size }, modifier = Modifier.fillMaxWidth())
                        ttsSteps.forEach { step ->
                            Text("${if (step !in missingTtsComponents) "✓" else "○"}  $step", style = MaterialTheme.typography.bodySmall)
                        }
                        Text(if (ttsDone == ttsSteps.size) "هر سه جزء وارد شده‌اند؛ مدل آماده است." else "پیشرفت ورود مدل: $ttsDone از ${ttsSteps.size} جزء", style = MaterialTheme.typography.labelMedium)
                        if (ttsModels.isEmpty()) Text("مدل تولید گفتار واردشده و آماده‌ای وجود ندارد.", color = MaterialTheme.colorScheme.error)
                        ttsModels.forEach { model ->
                            Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
                                RadioButton(selected = model.id == activeTtsId, onClick = { onSelectTts(model.id) })
                                Column(Modifier.weight(1f)) {
                                    Text(model.title, fontWeight = FontWeight.Medium)
                                    Text("آفلاین · ${model.modelFile.name}", style = MaterialTheme.typography.bodySmall)
                                }
                            }
                        }
                    }
                }
                Card(Modifier.fillMaxWidth()) {
                    Column(Modifier.fillMaxWidth().padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Column(Modifier.weight(1f)) {
                                Text("۳. عیب‌یابی مکالمه صوتی", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                                Text("وضعیت فعلی، کد خطا و رخدادهای اخیر برای گزارش مشکل.", style = MaterialTheme.typography.bodySmall)
                            }
                            IconButton(onClick = {
                                clipboard.setText(AnnotatedString(buildString {
                                    appendLine("AI Chat Test — Live Voice Diagnostics")
                                    appendLine("Status: $status")
                                    appendLine("Listening: $listening")
                                    appendLine("STT: ${sttModels.firstOrNull { it.id == activeSttId }?.title ?: "not selected"}")
                                    appendLine("TTS: ${ttsModels.firstOrNull { it.id == activeTtsId }?.title ?: "not selected"}")
                                    appendLine()
                                    diagnosticLogs.forEach { appendLine(it) }
                                }))
                            }) { Icon(Icons.Default.ContentCopy, contentDescription = "کپی گزارش عیب‌یابی") }
                        }
                        Text(status, style = MaterialTheme.typography.bodyMedium)
                        Text("گزارش قابل کپی", style = MaterialTheme.typography.labelLarge)
                        Surface(
                            modifier = Modifier.fillMaxWidth(),
                            color = MaterialTheme.colorScheme.surfaceVariant,
                            shape = MaterialTheme.shapes.medium,
                        ) {
                            Text(
                                diagnosticLogs.takeLast(80).joinToString("\n").ifBlank { "هنوز رخدادی ثبت نشده است." },
                                modifier = Modifier.padding(10.dp),
                                style = MaterialTheme.typography.bodySmall,
                            )
                        }
                        OutlinedButton(
                            onClick = {
                                clipboard.setText(AnnotatedString(buildString {
                                    appendLine("AI Chat Test — Live Voice Diagnostics")
                                    appendLine("Status: $status")
                                    appendLine("Listening: $listening")
                                    appendLine("STT: ${sttModels.firstOrNull { it.id == activeSttId }?.title ?: "not selected"}")
                                    appendLine("TTS: ${ttsModels.firstOrNull { it.id == activeTtsId }?.title ?: "not selected"}")
                                    appendLine()
                                    diagnosticLogs.forEach { appendLine(it) }
                                }))
                            },
                            modifier = Modifier.fillMaxWidth(),
                        ) { Icon(Icons.Default.ContentCopy, null); Spacer(Modifier.width(8.dp)); Text("کپی گزارش کامل و لاگ‌ها") }
                    }
                }
            }
        } else {
            Card(Modifier.fillMaxWidth()) {
                Row(Modifier.fillMaxWidth().padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
                    Surface(
                        shape = MaterialTheme.shapes.extraLarge,
                        color = if (listening) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.surfaceVariant,
                    ) {
                        Icon(Icons.Default.Mic, contentDescription = null, modifier = Modifier.padding(10.dp))
                    }
                    Spacer(Modifier.width(12.dp))
                    Column(Modifier.weight(1f)) {
                        Text(if (listening) "میکروفون فعال است" else "آماده شروع", fontWeight = FontWeight.SemiBold)
                        Text(status, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                    if (busy) CircularProgressIndicator(modifier = Modifier.size(22.dp), strokeWidth = 2.dp)
                }
            }
            Spacer(Modifier.height(8.dp))
            if (lines.isEmpty()) {
                Column(
                    Modifier.weight(1f).fillMaxWidth().padding(20.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center,
                ) {
                    Surface(
                        shape = MaterialTheme.shapes.extraLarge,
                        color = MaterialTheme.colorScheme.primaryContainer,
                    ) {
                        Icon(Icons.Default.Mic, contentDescription = null, modifier = Modifier.padding(28.dp).size(44.dp), tint = MaterialTheme.colorScheme.onPrimaryContainer)
                    }
                    Spacer(Modifier.height(16.dp))
                    Text("برای گفت‌وگو آماده‌ای؟", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
                    Spacer(Modifier.height(8.dp))
                    Text(
                        "ابتدا در تنظیمات، مدل تشخیص گفتار و مدل تولید گفتار را انتخاب کن. بعد شروع را بزن و صحبت کن؛ پاسخ دستیار به‌صورت صوتی پخش می‌شود.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    Spacer(Modifier.height(12.dp))
                    TextButton(onClick = { showSettings = true }) { Icon(Icons.Default.Settings, null); Spacer(Modifier.width(6.dp)); Text("تنظیم مدل‌ها") }
                }
            } else {
                LazyColumn(
                    modifier = Modifier.weight(1f).fillMaxWidth(),
                    contentPadding = PaddingValues(vertical = 12.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp),
                ) {
                    items(lines, key = { it.id }) { line ->
                        val isUser = line.speaker == "شما"
                        Row(
                            Modifier.fillMaxWidth(),
                            horizontalArrangement = if (isUser) Arrangement.Start else Arrangement.End,
                        ) {
                            Card(
                                modifier = Modifier.fillMaxWidth(0.9f),
                                colors = CardDefaults.cardColors(
                                    containerColor = if (isUser) MaterialTheme.colorScheme.secondaryContainer else MaterialTheme.colorScheme.primaryContainer,
                                ),
                            ) {
                                Column(Modifier.padding(12.dp)) {
                                    Text(line.speaker, style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.Bold)
                                    Spacer(Modifier.height(4.dp))
                                    Text(line.text, style = MaterialTheme.typography.bodyLarge)
                                }
                            }
                        }
                    }
                }
            }
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                Button(
                    onClick = { if (permissionGranted) onStart() else permission.launch(Manifest.permission.RECORD_AUDIO) },
                    enabled = !busy && !listening && activeSttId != null && activeTtsId != null,
                    modifier = Modifier.weight(1f).heightIn(min = 54.dp),
                ) {
                    Icon(Icons.Default.Mic, null)
                    Spacer(Modifier.width(8.dp))
                    Text(if (listening) "در حال گوش‌دادن" else "شروع مکالمه")
                }
                OutlinedButton(onClick = onInterrupt, enabled = listening, modifier = Modifier.heightIn(min = 54.dp)) {
                    Text("قطع پاسخ")
                }
                OutlinedButton(onClick = onStop, enabled = listening || busy, modifier = Modifier.heightIn(min = 54.dp)) {
                    Icon(Icons.Default.Stop, null)
                    Text("توقف")
                }
            }
        }
    }
}
