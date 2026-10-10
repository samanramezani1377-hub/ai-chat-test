package com.samanramezani.aichattest.ui.voice

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.samanramezani.aichattest.voice.VoiceModelEntry

data class VoiceLine(val id: Long, val speaker: String, val text: String)

@Composable
internal fun VoiceConversationPage(
    lines: List<VoiceLine>,
    status: String,
    listening: Boolean,
    busy: Boolean,
    sttModels: List<VoiceModelEntry>,
    ttsModels: List<VoiceModelEntry>,
    activeSttId: String?,
    activeTtsId: String?,
    onBack: () -> Unit,
    onImportStt: () -> Unit,
    onImportTts: () -> Unit,
    onSelectStt: (String) -> Unit,
    onSelectTts: (String) -> Unit,
    onStart: () -> Unit,
    onStop: () -> Unit,
    onInterrupt: () -> Unit,
) {
    val context = LocalContext.current
    var permissionGranted by remember {
        mutableStateOf(ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED)
    }
    val permission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        permissionGranted = granted
        if (granted) onStart()
    }

    Column(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            IconButton(onClick = onBack) { Icon(Icons.Default.ArrowBack, contentDescription = "بازگشت") }
            Column(Modifier.weight(1f)) {
                Text("مکالمه صوتی زنده", style = MaterialTheme.typography.headlineSmall)
                Text("کاملاً محلی · امکان قطع صحبت مدل", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.secondary)
            }
        }
        Text(status, style = MaterialTheme.typography.bodyMedium)
        Card {
            Column(Modifier.fillMaxWidth().padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("مدل تشخیص گفتار (STT)", style = MaterialTheme.typography.titleMedium)
                Text("برای گفت‌وگوی زنده، بستهٔ سازگار با sherpa-onnx و مدل Transducer لازم است.", style = MaterialTheme.typography.bodySmall)
                if (sttModels.isEmpty()) Text("هنوز مدل STT وارد نشده است.", color = MaterialTheme.colorScheme.onSurfaceVariant)
                sttModels.forEach { model ->
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        RadioButton(selected = model.id == activeSttId, onClick = { onSelectStt(model.id) })
                        Column(Modifier.weight(1f)) {
                            Text(model.title)
                            Text("محلی · Transducer", style = MaterialTheme.typography.labelSmall)
                        }
                    }
                }
                OutlinedButton(onClick = onImportStt, modifier = Modifier.fillMaxWidth()) { Text("وارد کردن بستهٔ STT (.zip)") }
            }
        }
        Card {
            Column(Modifier.fillMaxWidth().padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("مدل گفتار (TTS)", style = MaterialTheme.typography.titleMedium)
                if (ttsModels.isEmpty()) Text("بستهٔ تبدیل‌شدهٔ Piper/VITS را وارد کنید.", color = MaterialTheme.colorScheme.onSurfaceVariant)
                ttsModels.forEach { model ->
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        RadioButton(selected = model.id == activeTtsId, onClick = { onSelectTts(model.id) })
                        Column(Modifier.weight(1f)) {
                            Text(model.title)
                            Text("آفلاین · ${model.modelFile.name}", style = MaterialTheme.typography.labelSmall)
                        }
                    }
                }
                OutlinedButton(onClick = onImportTts, modifier = Modifier.fillMaxWidth()) { Text("وارد کردن بستهٔ TTS (.zip)") }
            }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
            Button(
                onClick = { if (permissionGranted) onStart() else permission.launch(Manifest.permission.RECORD_AUDIO) },
                enabled = !busy && activeSttId != null && activeTtsId != null,
                modifier = Modifier.weight(1f).heightIn(min = 52.dp),
            ) { Icon(Icons.Default.Mic, null); Spacer(Modifier.width(6.dp)); Text(if (listening) "گوش‌دادن فعال است" else "شروع مکالمه") }
            OutlinedButton(onClick = onStop, modifier = Modifier.heightIn(min = 52.dp)) { Icon(Icons.Default.Stop, null); Text("توقف") }
        }
        OutlinedButton(onClick = onInterrupt, modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp), enabled = listening) {
            Text("قطع فوری صدای مدل")
        }
        LazyColumn(Modifier.weight(1f).fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            items(lines, key = { it.id }) { line ->
                Card(Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(12.dp)) {
                        Text(line.speaker, style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.secondary)
                        Text(line.text)
                    }
                }
            }
        }
    }
}
