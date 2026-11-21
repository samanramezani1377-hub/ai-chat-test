package com.samanramezani.aichattest.voice

import org.json.JSONObject
import java.io.File

/**
 * Adapter over the official Hugging Face Tokenizers Rust pipeline.
 * tokenizer.json is authoritative; no custom Persian normalization or algorithm fallback.
 */
internal class LayaTokenizer(tokenizerFile: File, tokenizerConfigFile: File) : AutoCloseable {
    private val native = LayaNativeTokenizer(tokenizerFile.readBytes())
    private val config = JSONObject(tokenizerConfigFile.readText(Charsets.UTF_8))

    private fun configuredToken(name: String): String {
        val raw = config.opt(name)
        val value = when (raw) {
            is String -> raw
            is JSONObject -> raw.optString("content").ifBlank { raw.optString("token") }
            else -> ""
        }
        require(value.isNotBlank()) {
            "LAYA-TOKENIZER-012: توکن ویژه '$name' در tokenizer_config.json وجود ندارد."
        }
        return value
    }

    val clsToken: String = configuredToken("cls_token")
    val sepToken: String = configuredToken("sep_token")
    val maskToken: String = configuredToken("mask_token")
    val clsId: Long = native.tokenId(clsToken)
        ?: error("LAYA-TOKENIZER-013: شناسهٔ cls_token در tokenizer.json پیدا نشد: $clsToken")
    val sepId: Long = native.tokenId(sepToken)
        ?: error("LAYA-TOKENIZER-014: شناسهٔ sep_token در tokenizer.json پیدا نشد: $sepToken")
    val maskId: Long = native.tokenId(maskToken)
        ?: error("LAYA-TOKENIZER-015: شناسهٔ mask_token در tokenizer.json پیدا نشد: $maskToken")

    fun encode(raw: String, maxTokens: Int): List<Long> {
        if (maxTokens <= 0 || raw.isEmpty()) return emptyList()
        return native.encode(raw, addSpecialTokens = false).asList().take(maxTokens)
    }

    override fun close() = native.close()
}
