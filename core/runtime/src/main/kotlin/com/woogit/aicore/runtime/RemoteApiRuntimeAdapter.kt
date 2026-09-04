package com.woogit.aicore.runtime

import com.woogit.aicore.domain.ApiProtocol
import com.woogit.aicore.domain.ApiProviderConfig
import com.woogit.aicore.domain.GenerationRequest
import com.woogit.aicore.domain.GenerationResult
import com.woogit.aicore.domain.ModelDescriptor
import com.woogit.aicore.domain.RuntimeInfo
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.net.HttpURLConnection
import java.net.URL
import java.nio.charset.StandardCharsets
import java.util.concurrent.atomic.AtomicBoolean

/** Network model runtime. It speaks configurable OpenAI-compatible or Anthropic-compatible HTTP APIs. */
class RemoteApiRuntimeAdapter(
    private val configProvider: () -> ApiProviderConfig,
) : RuntimeAdapter, RuntimeMetrics {
    private val stopped = AtomicBoolean(false)
    @Volatile private var last: GenerationResult? = null
    @Volatile private var lastLatency: Long? = null
    @Volatile private var activeConfig: ApiProviderConfig = configProvider()

    override suspend fun load(model: ModelDescriptor) {
        activeConfig = configProvider()
        require(activeConfig.apiKey.isNotBlank()) { "API key is not configured" }
        require(activeConfig.model.isNotBlank()) { "API model is not configured" }
    }

    override suspend fun unload() = Unit

    override suspend fun generate(request: GenerationRequest, onToken: suspend (String) -> Unit): GenerationResult {
        val config = configProvider().also { activeConfig = it }
        require(config.apiKey.isNotBlank()) { "API key is not configured" }
        stopped.set(false)
        val started = System.nanoTime()
        val body = when (config.protocol) {
            ApiProtocol.OPENAI_CHAT -> openAiBody(config, request)
            ApiProtocol.ANTHROPIC_MESSAGES -> anthropicBody(config, request)
        }
        val connection = (URL(endpoint(config)).openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            connectTimeout = config.timeoutMs.toInt().coerceAtLeast(5_000)
            readTimeout = config.timeoutMs.toInt().coerceAtLeast(5_000)
            doOutput = true
            setRequestProperty("Content-Type", "application/json")
            setRequestProperty("Accept", "text/event-stream, application/json")
            when (config.protocol) {
                ApiProtocol.OPENAI_CHAT -> setRequestProperty("Authorization", "Bearer ${config.apiKey}")
                ApiProtocol.ANTHROPIC_MESSAGES -> setRequestProperty("x-api-key", config.apiKey)
            }
        }
        try {
            connection.outputStream.use { it.write(body.toString().toByteArray(StandardCharsets.UTF_8)) }
            val code = connection.responseCode
            val input = if (code in 200..299) connection.inputStream else connection.errorStream
            val response = input?.let { stream -> BufferedReader(InputStreamReader(stream, StandardCharsets.UTF_8)).use { reader -> parseResponse(reader, config.protocol, onToken) } } ?: ""
            if (code !in 200..299) throw IllegalStateException("API HTTP $code: ${response.ifBlank { connection.responseMessage ?: "request failed" }}")
            val elapsed = (System.nanoTime() - started) / 1_000_000
            val result = GenerationResult(text = response, generationTimeMs = elapsed, stopped = stopped.get())
            last = result
            lastLatency = elapsed
            return result
        } finally {
            connection.disconnect()
        }
    }

    private suspend fun parseResponse(reader: BufferedReader, protocol: ApiProtocol, onToken: suspend (String) -> Unit): String {
        val output = StringBuilder()
        while (true) {
            if (stopped.get()) break
            val line = reader.readLine() ?: break
            val trimmed = line.trim()
            if (trimmed.isEmpty() || trimmed.startsWith(":")) continue
            val data = if (trimmed.startsWith("data:")) trimmed.removePrefix("data:").trim() else trimmed
            if (data == "[DONE]") continue
            if (!data.startsWith("{")) continue
            val chunk = runCatching { parseChunk(JSONObject(data), protocol) }.getOrNull().orEmpty()
            if (chunk.isNotEmpty()) {
                output.append(chunk)
                onToken(chunk)
            }
        }
        return output.toString()
    }

    private fun parseChunk(json: JSONObject, protocol: ApiProtocol): String = when (protocol) {
        ApiProtocol.OPENAI_CHAT -> {
            val choice = json.optJSONArray("choices")?.optJSONObject(0) ?: return ""
            choice.optJSONObject("delta")?.optString("content", "")
                ?.ifBlank { choice.optJSONObject("message")?.optString("content", "") ?: "" } ?: ""
        }
        ApiProtocol.ANTHROPIC_MESSAGES -> when (json.optString("type")) {
            "content_block_delta" -> json.optJSONObject("delta")?.optString("text", "") ?: ""
            "content_block_start" -> json.optJSONObject("content_block")?.optString("text", "") ?: ""
            else -> if (json.has("content")) json.optJSONArray("content")?.let { array ->
                (0 until array.length()).joinToString("") { array.optJSONObject(it)?.optString("text", "") ?: "" }
            } ?: "" else ""
        }
    }

    private fun openAiBody(config: ApiProviderConfig, request: GenerationRequest): JSONObject {
        val messages = JSONArray()
        request.messages.forEach { message ->
            messages.put(JSONObject().put("role", message.role.name.lowercase()).put("content", message.content))
        }
        return JSONObject().put("model", config.model).put("messages", messages).put("stream", true)
            .put("temperature", request.settings.temperature.coerceIn(0.0, 2.0))
            .put("max_tokens", request.settings.maxNewTokens.coerceAtLeast(1))
            .apply {
                request.settings.topP?.let { put("top_p", it.coerceIn(0.0, 1.0)) }
                if (request.settings.stopSequences.isNotEmpty()) put("stop", JSONArray(request.settings.stopSequences.take(16)))
                config.reasoningEffort?.let { put("reasoning_effort", it) }
                if (config.providerId == "deepseek") put("thinking", JSONObject().put("type", if (config.thinkingEnabled) "enabled" else "disabled"))
            }
    }

    private fun anthropicBody(config: ApiProviderConfig, request: GenerationRequest): JSONObject {
        val messages = JSONArray()
        val system = StringBuilder()
        request.messages.forEach { message ->
            if (message.role.name == "SYSTEM") system.append(message.content).append('\n')
            else messages.put(JSONObject().put("role", if (message.role.name == "ASSISTANT") "assistant" else "user").put("content", message.content))
        }
        return JSONObject().put("model", config.model).put("max_tokens", request.settings.maxNewTokens.coerceAtLeast(1)).put("messages", messages).put("stream", true)
            .put("temperature", request.settings.temperature.coerceIn(0.0, 2.0))
            .apply {
                if (system.isNotEmpty()) put("system", system.toString().trim())
                request.settings.topP?.let { put("top_p", it.coerceIn(0.0, 1.0)) }
                if (request.settings.stopSequences.isNotEmpty()) put("stop_sequences", JSONArray(request.settings.stopSequences.take(16)))
            }
    }

    private fun endpoint(config: ApiProviderConfig): String = when (config.protocol) {
        ApiProtocol.OPENAI_CHAT -> {
            val base = config.normalizedBaseUrl()
            if (base.endsWith("/chat/completions")) base else "$base/chat/completions"
        }
        ApiProtocol.ANTHROPIC_MESSAGES -> {
            val base = config.normalizedBaseUrl()
            if (base.endsWith("/messages")) base else "$base/v1/messages"
        }
    }

    override suspend fun stopGeneration() { stopped.set(true) }
    override fun runtimeInfo(): RuntimeInfo = RuntimeInfo(name = "remote-api", version = "1", backend = activeConfig.providerId, contextLength = null)
    override fun lastGeneration(): GenerationResult? = last
    override fun lastLoadTimeMs(): Long? = lastLatency
}
