package com.samanramezani.aichattest

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import com.woogit.aicore.domain.ApiProtocol
import com.woogit.aicore.domain.ApiProviderConfig

@Composable
fun ApiSetupScreen(initial: ApiProviderConfig, onSave: (ApiProviderConfig) -> Unit) {
    var provider by remember { mutableStateOf(initial.providerId) }
    var baseUrl by remember { mutableStateOf(initial.baseUrl) }
    var model by remember { mutableStateOf(initial.model) }
    var apiKey by remember { mutableStateOf(initial.apiKey) }
    var protocol by remember { mutableStateOf(initial.protocol) }

    Surface(Modifier.fillMaxSize()) {
        Column(
            Modifier.fillMaxSize().padding(24.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Text("اتصال به مدل هوش مصنوعی", style = MaterialTheme.typography.headlineMedium)
            Text("مدل‌های محلی و llama.cpp دیگر در مسیر Generation استفاده نمی‌شوند. یک API provider انتخاب یا یک endpoint سازگار با OpenAI وارد کنید.", color = MaterialTheme.colorScheme.onSurfaceVariant)
            OutlinedTextField(provider, { provider = it }, Modifier.fillMaxWidth(), label = { Text("Provider ID") }, singleLine = true)
            OutlinedTextField(baseUrl, { baseUrl = it }, Modifier.fillMaxWidth(), label = { Text("Base URL") }, singleLine = true)
            OutlinedTextField(model, { model = it }, Modifier.fillMaxWidth(), label = { Text("Model") }, singleLine = true)
            OutlinedTextField(apiKey, { apiKey = it }, Modifier.fillMaxWidth(), label = { Text("API Key") }, singleLine = true, visualTransformation = PasswordVisualTransformation(), keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password))
            OutlinedTextField(
                value = protocol.name,
                onValueChange = { if (it.equals("ANTHROPIC_MESSAGES", true)) protocol = ApiProtocol.ANTHROPIC_MESSAGES else protocol = ApiProtocol.OPENAI_CHAT },
                Modifier.fillMaxWidth(),
                label = { Text("Protocol: OPENAI_CHAT یا ANTHROPIC_MESSAGES") },
                singleLine = true,
            )
            Button(
                onClick = { onSave(ApiProviderConfig(provider.trim(), baseUrl.trim(), apiKey.trim(), model.trim(), protocol)) },
                Modifier.fillMaxWidth(),
                enabled = baseUrl.isNotBlank() && model.isNotBlank() && apiKey.isNotBlank(),
            ) { Text("ذخیره و ورود") }
        }
    }
}
