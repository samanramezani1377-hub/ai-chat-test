package com.samanramezani.aichattest

import android.content.Context
import android.util.Base64
import com.woogit.aicore.domain.ApiProtocol
import com.woogit.aicore.domain.ApiProviderConfig
import java.nio.charset.StandardCharsets
import java.security.KeyStore
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

/** Encrypted-at-rest storage for user-provided API configuration. */
class ApiConfigStore(context: Context) {
    private val prefs = context.applicationContext.getSharedPreferences("remote_api", Context.MODE_PRIVATE)
    private val keyAlias = "ai-chat-remote-api-v1"

    fun load(): ApiProviderConfig {
        val defaults = ApiProviderConfig()
        val encrypted = prefs.getString(KEY_API_KEY, null)
        val apiKey = encrypted?.let { runCatching { decrypt(it) }.getOrNull() }.orEmpty()
        return defaults.copy(
            providerId = prefs.getString(KEY_PROVIDER, defaults.providerId) ?: defaults.providerId,
            baseUrl = prefs.getString(KEY_BASE_URL, defaults.baseUrl) ?: defaults.baseUrl,
            model = prefs.getString(KEY_MODEL, defaults.model) ?: defaults.model,
            protocol = runCatching { ApiProtocol.valueOf(prefs.getString(KEY_PROTOCOL, defaults.protocol.name) ?: defaults.protocol.name) }.getOrDefault(defaults.protocol),
            reasoningEffort = prefs.getString(KEY_REASONING, defaults.reasoningEffort),
            thinkingEnabled = prefs.getBoolean(KEY_THINKING, defaults.thinkingEnabled),
            apiKey = apiKey,
        )
    }

    fun save(config: ApiProviderConfig) {
        prefs.edit()
            .putString(KEY_PROVIDER, config.providerId)
            .putString(KEY_BASE_URL, config.baseUrl.trim())
            .putString(KEY_MODEL, config.model.trim())
            .putString(KEY_PROTOCOL, config.protocol.name)
            .putString(KEY_REASONING, config.reasoningEffort)
            .putBoolean(KEY_THINKING, config.thinkingEnabled)
            .putString(KEY_API_KEY, encrypt(config.apiKey))
            .apply()
    }

    private fun secretKey(): SecretKey {
        val store = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
        (store.getKey(keyAlias, null) as? SecretKey)?.let { return it }
        val generator = KeyGenerator.getInstance("AES", "AndroidKeyStore")
        generator.init(android.security.keystore.KeyGenParameterSpec.Builder(
            keyAlias,
            android.security.keystore.KeyProperties.PURPOSE_ENCRYPT or android.security.keystore.KeyProperties.PURPOSE_DECRYPT,
        ).setBlockModes(android.security.keystore.KeyProperties.BLOCK_MODE_GCM)
            .setEncryptionPaddings(android.security.keystore.KeyProperties.ENCRYPTION_PADDING_NONE)
            .setKeySize(256)
            .build())
        return generator.generateKey()
    }

    private fun encrypt(value: String): String {
        if (value.isBlank()) return ""
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.ENCRYPT_MODE, secretKey())
        val iv = cipher.iv
        val ciphertext = cipher.doFinal(value.toByteArray(StandardCharsets.UTF_8))
        return Base64.encodeToString(iv + ciphertext, Base64.NO_WRAP)
    }

    private fun decrypt(encoded: String): String {
        if (encoded.isBlank()) return ""
        val data = Base64.decode(encoded, Base64.NO_WRAP)
        require(data.size > 12)
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.DECRYPT_MODE, secretKey(), GCMParameterSpec(128, data.copyOfRange(0, 12)))
        return String(cipher.doFinal(data.copyOfRange(12, data.size)), StandardCharsets.UTF_8)
    }

    companion object {
        private const val KEY_PROVIDER = "provider"
        private const val KEY_BASE_URL = "base_url"
        private const val KEY_MODEL = "model"
        private const val KEY_PROTOCOL = "protocol"
        private const val KEY_REASONING = "reasoning_effort"
        private const val KEY_THINKING = "thinking_enabled"
        private const val KEY_API_KEY = "api_key"
    }
}
