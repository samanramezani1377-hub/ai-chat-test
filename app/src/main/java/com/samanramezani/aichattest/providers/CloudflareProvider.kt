package com.samanramezani.aichattest.providers

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Base64
import com.woogit.aicore.domain.ChatMessage
import com.woogit.aicore.domain.GenerationRequest
import com.woogit.aicore.domain.GenerationResult
import com.woogit.aicore.domain.ModelDescriptor
import com.woogit.aicore.domain.ModelRuntime
import com.woogit.aicore.domain.RuntimeInfo
import com.woogit.aicore.runtime.RuntimeAdapter
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL
import java.security.KeyStore
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

enum class InferenceProviderMode { LOCAL, CLOUDFLARE }

data class CloudflareConfig(
    val accountId: String = "",
    val apiToken: String = "",
    val model: String = CloudflareProvider.DEFAULT_MODEL,
    val mode: InferenceProviderMode = InferenceProviderMode.LOCAL,
)

data class ProviderModel(val id: String, val label: String)

/** Small provider contract: new remote providers can be registered without changing AgentSession. */
interface ChatCompletionProvider {
    val id: String
    val displayName: String
    suspend fun generate(config: CloudflareConfig, request: GenerationRequest, onToken: suspend (String) -> Unit): GenerationResult
    suspend fun stopGeneration() = Unit
}

class CloudflareProvider : ChatCompletionProvider {
    private val activeConnection = java.util.concurrent.atomic.AtomicReference<HttpURLConnection?>(null)

    override val id: String = "cloudflare-workers-ai"
    override val displayName: String = "Cloudflare Workers AI"

    override suspend fun generate(
        config: CloudflareConfig,
        request: GenerationRequest,
        onToken: suspend (String) -> Unit,
    ): GenerationResult = withContext(Dispatchers.IO) {
        require(config.accountId.isNotBlank()) { "CLOUDFLARE-001: شناسه حساب Cloudflare وارد نشده است." }
        require(config.apiToken.isNotBlank()) { "CLOUDFLARE-002: توکن API وارد نشده است." }
        require(config.model.isNotBlank()) { "CLOUDFLARE-003: شناسه مدل انتخاب نشده است." }
        require(config.accountId.matches(Regex("[A-Za-z0-9_-]{8,128}"))) {
            "CLOUDFLARE-004: قالب شناسه حساب معتبر نیست."
        }
        val endpoint = URL("https://api.cloudflare.com/client/v4/accounts/${config.accountId}/ai/v1/chat/completions")
        val connection = endpoint.openConnection() as HttpURLConnection
        activeConnection.set(connection)
        val startedAt = System.nanoTime()
        var firstTokenMs: Long? = null
        var outputTokens = 0L
        val output = StringBuilder()
        try {
            connection.requestMethod = "POST"
            connection.connectTimeout = 20_000
            connection.readTimeout = 120_000
            connection.doOutput = true
            connection.setRequestProperty("Authorization", "Bearer ${config.apiToken}")
            connection.setRequestProperty("Content-Type", "application/json; charset=utf-8")
            connection.setRequestProperty("Accept", "text/event-stream")
            val messages = JSONArray()
            request.messages.forEach { message ->
                val role = when (message.role) {
                    ChatMessage.Role.SYSTEM -> "system"
                    ChatMessage.Role.USER -> "user"
                    ChatMessage.Role.ASSISTANT -> "assistant"
                    ChatMessage.Role.TOOL -> "user"
                }
                messages.put(JSONObject().put("role", role).put("content", message.content))
            }
            val body = JSONObject()
                .put("model", config.model.trim())
                .put("messages", messages)
                .put("temperature", request.settings.temperature)
                .put("max_tokens", request.settings.maxNewTokens.coerceIn(1, 8192))
                .put("stream", true)
            request.settings.topP?.let { body.put("top_p", it) }
            request.settings.stopSequences.takeIf { it.isNotEmpty() }?.let { body.put("stop", JSONArray(it)) }
            connection.outputStream.use { it.write(body.toString().toByteArray(Charsets.UTF_8)) }

            val status = connection.responseCode
            val stream = if (status in 200..299) connection.inputStream else connection.errorStream
            val responseText = stream?.bufferedReader(Charsets.UTF_8)?.use { reader ->
                val fallback = StringBuilder()
                var sawSse = false
                while (true) {
                    val line = reader.readLine() ?: break
                    if (line.startsWith("data:")) {
                        sawSse = true
                        val payload = line.removePrefix("data:").trim()
                        if (payload.isNotEmpty() && payload != "[DONE]") {
                            val json = runCatching { JSONObject(payload) }.getOrNull()
                            val delta = json?.optJSONArray("choices")?.optJSONObject(0)
                                ?.optJSONObject("delta")?.optString("content").orEmpty()
                            if (delta.isNotEmpty()) {
                                if (firstTokenMs == null) firstTokenMs = (System.nanoTime() - startedAt) / 1_000_000
                                output.append(delta)
                                outputTokens++
                                onToken(delta)
                            }
                            val apiError = json?.optJSONObject("error")
                            if (apiError != null) throw IllegalStateException("Cloudflare API: ${apiError.optString("message", payload)}")
                        }
                    } else if (!sawSse && line.isNotBlank()) {
                        fallback.append(line)
                    }
                }
                if (!sawSse && fallback.isNotEmpty()) fallback.toString() else null
            }.orEmpty()

            if (status !in 200..299) {
                val detail = runCatching { JSONObject(responseText).optJSONObject("error")?.optString("message") }.getOrNull()
                    ?.takeIf { it.isNotBlank() } ?: responseText.take(500).ifBlank { "HTTP $status" }
                throw IllegalStateException("CLOUDFLARE-HTTP-$status: درخواست Cloudflare ناموفق بود: $detail")
            }
            if (output.isEmpty() && responseText.isNotBlank()) {
                val json = runCatching { JSONObject(responseText) }.getOrNull()
                val choices = json?.optJSONArray("choices")
                val content = choices?.optJSONObject(0)?.optJSONObject("message")?.optString("content").orEmpty()
                if (content.isNotBlank()) {
                    output.append(content)
                    outputTokens = content.trim().split(Regex("\\s+")).size.toLong()
                    firstTokenMs = (System.nanoTime() - startedAt) / 1_000_000
                    onToken(content)
                }
            }
            if (output.isBlank()) throw IllegalStateException("CLOUDFLARE-EMPTY: پاسخ مدل خالی بود؛ شناسه مدل و دسترسی API را بررسی کن.")
            GenerationResult(
                text = output.toString(),
                outputTokens = outputTokens,
                firstTokenTimeMs = firstTokenMs,
                generationTimeMs = (System.nanoTime() - startedAt) / 1_000_000,
            )
        } finally {
            activeConnection.compareAndSet(connection, null)
            connection.disconnect()
        }
    }

    override suspend fun stopGeneration() {
        activeConnection.getAndSet(null)?.disconnect()
    }

    suspend fun listModels(config: CloudflareConfig): List<ProviderModel> = withContext(Dispatchers.IO) {
        require(config.accountId.isNotBlank() && config.apiToken.isNotBlank()) { "ابتدا شناسه حساب و توکن Cloudflare را وارد کن." }
        val url = URL("https://api.cloudflare.com/client/v4/accounts/${config.accountId}/ai/models/search?task=Text%20Generation&per_page=100&hide_experimental=true")
        val connection = url.openConnection() as HttpURLConnection
        try {
            connection.requestMethod = "GET"
            connection.connectTimeout = 15_000
            connection.readTimeout = 30_000
            connection.setRequestProperty("Authorization", "Bearer ${config.apiToken}")
            connection.setRequestProperty("Accept", "application/json")
            val status = connection.responseCode
            val stream = if (status in 200..299) connection.inputStream else connection.errorStream
            val text = stream?.bufferedReader(Charsets.UTF_8)?.use { it.readText() }.orEmpty()
            if (status !in 200..299) throw IllegalStateException("CLOUDFLARE-MODELS-$status: دریافت فهرست مدل‌ها ناموفق بود: ${text.take(300)}")
            val result = JSONObject(text).optJSONArray("result") ?: JSONArray()
            (0 until result.length()).mapNotNull { index ->
                val item = result.optJSONObject(index) ?: return@mapNotNull null
                val name = item.optString("name").takeIf { it.startsWith("@cf/") }
                    ?: item.optString("id").takeIf { it.startsWith("@cf/") }
                    ?: return@mapNotNull null
                val description = item.optString("description").takeIf { it.isNotBlank() && it != "null" }
                ProviderModel(name, if (description == null) name else "${name} · ${description.take(72)}")
            }.distinctBy { it.id }.sortedBy { it.id }
        } finally {
            connection.disconnect()
        }
    }

    companion object {
        const val DEFAULT_MODEL = "@cf/meta/llama-3.1-8b-instruct"
        val suggestedModels = listOf(
            ProviderModel("@cf/meta/llama-3.1-8b-instruct", "Llama 3.1 · 8B"),
            ProviderModel("@cf/qwen/qwen3-30b-a3b-fp8", "Qwen3 · 30B-A3B"),
            ProviderModel("@cf/zai-org/glm-4.7-flash", "GLM 4.7 Flash"),
            ProviderModel("@cf/google/gemma-3-12b-it", "Gemma 3 · 12B"),
            ProviderModel("@cf/openai/gpt-oss-20b", "GPT-OSS · 20B"),
        )
    }
}

/** Persists provider configuration; API tokens are encrypted with an Android Keystore AES-GCM key. */
class CloudflareProviderStore(context: Context) {
    private val prefs = context.applicationContext.getSharedPreferences("model_provider_settings", Context.MODE_PRIVATE)

    fun load(): CloudflareConfig = CloudflareConfig(
        accountId = prefs.getString(KEY_ACCOUNT, "").orEmpty(),
        apiToken = decrypt(prefs.getString(KEY_TOKEN, null)),
        model = prefs.getString(KEY_MODEL, CloudflareProvider.DEFAULT_MODEL) ?: CloudflareProvider.DEFAULT_MODEL,
        mode = runCatching { InferenceProviderMode.valueOf(prefs.getString(KEY_MODE, "LOCAL") ?: "LOCAL") }
            .getOrDefault(InferenceProviderMode.LOCAL),
    )

    fun save(config: CloudflareConfig) {
        val editor = prefs.edit()
            .putString(KEY_ACCOUNT, config.accountId.trim())
            .putString(KEY_MODEL, config.model.trim())
            .putString(KEY_MODE, config.mode.name)
        if (config.apiToken.isBlank()) editor.remove(KEY_TOKEN) else editor.putString(KEY_TOKEN, encrypt(config.apiToken.trim()))
        check(editor.commit()) { "تنظیمات ارائه‌دهنده ذخیره نشد." }
    }

    fun selectMode(mode: InferenceProviderMode) {
        val current = load()
        if (mode == InferenceProviderMode.CLOUDFLARE) {
            require(current.accountId.isNotBlank() && current.apiToken.isNotBlank() && current.model.isNotBlank()) {
                "ابتدا شناسه حساب، توکن API و مدل Cloudflare را کامل کن."
            }
        }
        save(current.copy(mode = mode))
    }

    private fun encrypt(value: String): String {
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.ENCRYPT_MODE, getOrCreateKey())
        val encrypted = cipher.doFinal(value.toByteArray(Charsets.UTF_8))
        return Base64.encodeToString(cipher.iv + encrypted, Base64.NO_WRAP)
    }

    private fun decrypt(value: String?): String {
        if (value.isNullOrBlank()) return ""
        return runCatching {
            val bytes = Base64.decode(value, Base64.NO_WRAP)
            val iv = bytes.copyOfRange(0, 12)
            val encrypted = bytes.copyOfRange(12, bytes.size)
            val cipher = Cipher.getInstance("AES/GCM/NoPadding")
            cipher.init(Cipher.DECRYPT_MODE, getOrCreateKey(), GCMParameterSpec(128, iv))
            String(cipher.doFinal(encrypted), Charsets.UTF_8)
        }.getOrDefault("")
    }

    private fun getOrCreateKey(): SecretKey {
        val store = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
        (store.getKey(KEY_ALIAS, null) as? SecretKey)?.let { return it }
        val generator = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore")
        generator.init(
            KeyGenParameterSpec.Builder(
                KEY_ALIAS,
                KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT,
            ).setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                .setRandomizedEncryptionRequired(true)
                .build()
        )
        return generator.generateKey()
    }

    companion object {
        private const val KEY_ALIAS = "ai_chat_test_provider_token_v1"
        private const val KEY_ACCOUNT = "cloudflare_account_id"
        private const val KEY_TOKEN = "cloudflare_api_token_encrypted"
        private const val KEY_MODEL = "cloudflare_model_id"
        private const val KEY_MODE = "inference_provider_mode"
    }
}

/** Keeps local inference as the default and routes the same AgentSession to a selected API provider. */
class ProviderRoutingModelRuntime(
    private val local: ModelRuntime,
    private val providerStore: CloudflareProviderStore,
    private val cloudflare: ChatCompletionProvider = CloudflareProvider(),
) : RuntimeAdapter {
    override suspend fun load(model: ModelDescriptor) = local.load(model)
    override suspend fun unload() = local.unload()

    override suspend fun generate(request: GenerationRequest, onToken: suspend (String) -> Unit): GenerationResult {
        val config = providerStore.load()
        return when (config.mode) {
            InferenceProviderMode.LOCAL -> local.generate(request, onToken)
            InferenceProviderMode.CLOUDFLARE -> {
                require(config.accountId.isNotBlank()) { "شناسه حساب Cloudflare وارد نشده است." }
                require(config.apiToken.isNotBlank()) { "توکن API در دسترس نیست؛ دوباره واردش کن و ذخیره کن." }
                require(config.model.isNotBlank()) { "شناسه مدل Cloudflare وارد نشده است." }
                cloudflare.generate(config, request, onToken)
            }
        }
    }

    override suspend fun stopGeneration() {
        if (providerStore.load().mode == InferenceProviderMode.CLOUDFLARE) cloudflare.stopGeneration()
        else local.stopGeneration()
    }

    override fun runtimeInfo(): RuntimeInfo {
        val config = providerStore.load()
        return if (config.mode == InferenceProviderMode.CLOUDFLARE) {
            RuntimeInfo(name = "Cloudflare Workers AI", version = "OpenAI-compatible v1", backend = "Cloud API · ${config.model}")
        } else local.runtimeInfo()
    }
}
