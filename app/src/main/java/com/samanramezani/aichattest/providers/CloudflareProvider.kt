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
        var streamEvents = 0
        var reasoningChars = 0
        var lastPayloadShape = "none"
        var finishReason = "none"
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
            // Voice calls set enableThinking=false. Forward that setting to Workers AI's
            // reasoning-capable models; otherwise GLM-4.7-Flash can spend the entire
            // max_tokens budget in reasoning_content and finish with no user-visible text.
            // Workers AI explicitly supports reasoning_effort=null and chat_template_kwargs
            // for GLM-4.7-Flash, Gemma 4, Kimi K2.x, GPT-OSS, and QwQ.
            val modelId = config.model.trim().lowercase()
            val supportsReasoningControls =
                modelId in setOf(
                    "@cf/zai-org/glm-4.7-flash",
                    "@cf/google/gemma-4-26b-a4b-it",
                    "@cf/moonshotai/kimi-k2.5",
                    "@cf/moonshotai/kimi-k2.6",
                ) || modelId.contains("gpt-oss") || modelId.contains("/qwq")
            if (!request.settings.enableThinking && supportsReasoningControls) {
                body.put("reasoning_effort", JSONObject.NULL)
                body.put(
                    "chat_template_kwargs",
                    JSONObject()
                        .put("enable_thinking", false)
                        .put("clear_thinking", true),
                )
            }
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
                            streamEvents++
                            val json = runCatching { JSONObject(payload) }.getOrNull()
                            if (json == null) {
                                lastPayloadShape = "invalid-json"
                                continue
                            }
                            val apiError = json.optJSONObject("error")
                            if (apiError != null) {
                                throw IllegalStateException("Cloudflare API: ${apiError.optString("message", "stream error")}")
                            }
                            val choices = json.optJSONArray("choices")
                            val choice = choices?.optJSONObject(0)
                            val deltaObject = choice?.optJSONObject("delta")
                            val messageObject = choice?.optJSONObject("message")
                            finishReason = choice?.optString("finish_reason")?.takeIf { it.isNotBlank() && it != "null" } ?: finishReason
                            // Never fall back to reasoning/reasoning_content: only user-facing content is eligible for speech.
                            val reasoning = contentText(deltaObject?.opt("reasoning_content"))
                                .ifBlank { contentText(deltaObject?.opt("reasoning")) }
                                .ifBlank { contentText(messageObject?.opt("reasoning_content")) }
                                .ifBlank { contentText(messageObject?.opt("reasoning")) }
                            reasoningChars += reasoning.length
                            val delta = contentText(deltaObject?.opt("content"))
                                .ifBlank { contentText(messageObject?.opt("content")) }
                            // Some Workers AI model families emit a native {response: ...}
                            // terminal event, and may also send empty choices/usage heartbeats.
                            val nativeResponse = contentText(json.opt("response"))
                            lastPayloadShape = buildString {
                                append("keys=").append(json.keys().asSequence().take(12).joinToString(","))
                                append(";choices=").append(choices?.length() ?: 0)
                                append(";deltaKeys=").append(deltaObject?.keys()?.asSequence()?.take(8)?.joinToString(",") ?: "-")
                                append(";messageKeys=").append(messageObject?.keys()?.asSequence()?.take(8)?.joinToString(",") ?: "-")
                                append(";responseChars=").append(nativeResponse.length)
                                append(";reasoningChars=").append(reasoning.length)
                                append(";finish=").append(finishReason)
                            }
                            val chunk = when {
                                delta.isNotEmpty() -> delta
                                output.isEmpty() && nativeResponse.isNotEmpty() -> nativeResponse
                                else -> ""
                            }
                            if (chunk.isNotEmpty()) {
                                if (firstTokenMs == null) firstTokenMs = (System.nanoTime() - startedAt) / 1_000_000
                                output.append(chunk)
                                outputTokens++
                                onToken(chunk)
                            }
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
                val choice = json?.optJSONArray("choices")?.optJSONObject(0)
                val content = contentText(choice?.optJSONObject("message")?.opt("content"))
                    .ifBlank { contentText(json?.opt("response")) }
                if (content.isNotBlank()) {
                    output.append(content)
                    outputTokens = content.trim().split(Regex("\\s+")).size.toLong()
                    firstTokenMs = (System.nanoTime() - startedAt) / 1_000_000
                    onToken(content)
                } else if (json != null) {
                    lastPayloadShape = "jsonKeys=" + json.keys().asSequence().take(12).joinToString(",") +
                        ";choices=" + (json.optJSONArray("choices")?.length() ?: 0) +
                        ";responseChars=" + contentText(json.opt("response")).length
                }
            }
            if (output.isBlank()) {
                throw IllegalStateException(
                    "CLOUDFLARE-EMPTY: پاسخ متنی دریافت نشد؛ model=${config.model}; HTTP=${status}; " +
                        "streamEvents=${streamEvents}; finish=${finishReason}; reasoningChars=${reasoningChars}; payload=${lastPayloadShape}. " +
                        "شناسه مدل/دسترسی API یا خروجی غیرمتنی مدل را بررسی کن."
                )
            }
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

    private fun contentText(value: Any?): String = when (value) {
        null, JSONObject.NULL -> ""
        is String -> value
        is Number, is Boolean -> value.toString()
        is JSONArray -> buildString {
            for (i in 0 until value.length()) {
                val item = value.opt(i)
                when (item) {
                    is JSONObject -> append(item.optString("text").takeIf { it != "null" }.orEmpty())
                    is String, is Number, is Boolean -> append(item.toString())
                }
            }
        }
        is JSONObject -> value.optString("text").takeIf { it != "null" }.orEmpty()
        else -> value.toString()
    }

    override suspend fun stopGeneration() {
        activeConnection.getAndSet(null)?.disconnect()
    }

    suspend fun listModels(config: CloudflareConfig): List<ProviderModel> = withContext(Dispatchers.IO) {
        require(config.accountId.isNotBlank() && config.apiToken.isNotBlank()) { "ابتدا شناسه حساب و توکن Cloudflare را وارد کن." }
        val url = URL("https://api.cloudflare.com/client/v4/accounts/${config.accountId}/ai/models/search?task=Text%20Generation&per_page=100&hide_experimental=true&include_deprecated=false")
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
                // The API query excludes deprecated models; also defensively reject deprecated
                // entries in case Cloudflare returns inconsistent metadata for an account.
                if (isDeprecatedModel(item)) return@mapNotNull null
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

    private fun isDeprecatedModel(item: JSONObject): Boolean {
        if (item.optBoolean("deprecated", false)) return true
        val status = item.optString("status").lowercase()
        if (status == "deprecated" || status == "removed" || status == "retired") return true
        val retiredAt = sequenceOf("deprecated_at", "deprecation_date", "removed_at")
            .map { item.optString(it) }
            .firstOrNull { it.isNotBlank() && it != "null" }
            ?: return false
        // Future planned retirement dates do not count as deprecated. Only reject a model
        // when Cloudflare reports an actual retirement date that has already passed.
        val timestamp = runCatching {
            java.time.Instant.parse(retiredAt).toEpochMilli()
        }.getOrElse {
            runCatching {
                java.time.LocalDate.parse(retiredAt.take(10))
                    .atStartOfDay(java.time.ZoneOffset.UTC).toInstant().toEpochMilli()
            }.getOrNull()
        } ?: return false
        return timestamp <= System.currentTimeMillis()
    }

    companion object {
        // Current active text-generation model; the previous Llama 3.1 ID was deprecated.
        const val DEFAULT_MODEL = "@cf/zai-org/glm-4.7-flash"
        val suggestedModels = listOf(
            ProviderModel("@cf/zai-org/glm-4.7-flash", "GLM 4.7 Flash"),
            ProviderModel("@cf/google/gemma-4-26b-a4b-it", "Gemma 4 · 26B A4B"),
            ProviderModel("@cf/moonshotai/kimi-k2.6", "Kimi K2.6"),
        )
        private val deprecatedModelIds = setOf(
            "@cf/meta/infire-llama-3.1-8b-instruct",
            "@cf/meta/llama-3.1-8b-instruct",
            "@cf/meta/llama-3.1-8b-instruct-awq",
            "@cf/meta/llama-3-8b-instruct",
            "@cf/google/gemma-3-12b-it",
        )
        fun isKnownDeprecatedModelId(id: String): Boolean =
            id.trim().lowercase() in deprecatedModelIds
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
