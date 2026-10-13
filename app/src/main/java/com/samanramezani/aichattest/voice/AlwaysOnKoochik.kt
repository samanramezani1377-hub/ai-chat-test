package com.samanramezani.aichattest.voice

import java.text.Normalizer

/**
 * Finds the wake word in one finalized, bounded ASR utterance and returns only the speech
 * after it. A null result means no wake word; an empty string means the wake word was alone.
 * The caller must discard the source transcript immediately after this call.
 */
internal fun extractDinaCommand(transcript: String): String? {
    val normalized = Normalizer.normalize(transcript, Normalizer.Form.NFC)
        .replace('ي', 'ی')
        .replace('ى', 'ی')
        .replace('ك', 'ک')
        .replace('ـ'.toString(), "")
        .lowercase()
    val match = Regex("(?<![\\p{L}\\p{N}])(?:دینا|دينا|دينه|dina)(?![\\p{L}\\p{N}])")
        .find(normalized) ?: return null
    return normalized.substring(match.range.last + 1)
        .trim()
        .trimStart('،', ',', ':', ';', '؛', '؟', '?', '!', '.', '-', '—')
        .trim()
}
