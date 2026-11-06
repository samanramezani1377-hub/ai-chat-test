package com.samanramezani.aichattest.voice

/**
 * On-device wrapper around the official Hugging Face Tokenizers Rust implementation.
 * The full tokenizer.json pipeline (added tokens, normalization, pre-tokenization,
 * model, post-processing and decoder configuration) is loaded once per model session.
 */
internal class LayaNativeTokenizer(tokenizerJson: ByteArray) : AutoCloseable {
    private var pointer: Long = createTokenizer(tokenizerJson)

    init {
        check(pointer != 0L) { "LAYA-TOKENIZER-NATIVE-010: ساخت توکنایزر بومی ناموفق بود." }
    }

    private external fun createTokenizer(tokenizerJson: ByteArray): Long
    private external fun encode(pointer: Long, text: String, addSpecialTokens: Boolean): IntArray
    private external fun tokenId(pointer: Long, token: String): Int
    private external fun deleteTokenizer(pointer: Long)

    @Synchronized
    fun encode(text: String, addSpecialTokens: Boolean = false): LongArray {
        check(pointer != 0L) { "LAYA-TOKENIZER-NATIVE-011: توکنایزر بسته شده است." }
        return encode(pointer, text, addSpecialTokens).map(Int::toLong).toLongArray()
    }

    @Synchronized
    fun tokenId(token: String): Long? {
        check(pointer != 0L) { "LAYA-TOKENIZER-NATIVE-011: توکنایزر بسته شده است." }
        return tokenId(pointer, token).takeIf { it >= 0 }?.toLong()
    }

    @Synchronized
    override fun close() {
        if (pointer != 0L) {
            deleteTokenizer(pointer)
            pointer = 0L
        }
    }

    companion object {
        init {
            System.loadLibrary("laya_tokenizers")
        }
    }
}
