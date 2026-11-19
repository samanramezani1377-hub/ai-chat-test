package com.samanramezani.aichattest

import android.content.Context
import com.woogit.aicore.domain.InferenceSettings
import com.woogit.aicore.domain.InferenceSettingsStore

/**
 * Single read path for persisted inference controls.
 * Keep keys/schema aligned with SettingsScreen's existing settings editor.
 */
object InferenceSettingsAccess {
    private const val PREFS = "inference_settings"
    private const val KEY_SCHEMA_VERSION = "schema_version"
    private const val CURRENT_SCHEMA_VERSION = 2
    private const val KEY_TEMPERATURE = "temperature"
    private const val KEY_TOP_P = "top_p"
    private const val KEY_TOP_K = "top_k"
    private const val KEY_MIN_P = "min_p"
    private const val KEY_REPEAT = "repeat_penalty"
    private const val KEY_MAX_TOKENS = "max_new_tokens"
    private const val KEY_CONTEXT = "context_length"
    private const val KEY_RECENT_MESSAGES = "recent_messages"
    private const val KEY_SEED = "seed"
    private const val KEY_STOPS = "stop_sequences"
    private const val KEY_MAX_ACTION_STEPS = "max_action_steps"

    fun load(context: Context): InferenceSettings {
        val prefs = context.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        if (prefs.getInt(KEY_SCHEMA_VERSION, 0) < CURRENT_SCHEMA_VERSION) {
            if (!prefs.contains(KEY_RECENT_MESSAGES)) {
                prefs.edit().putInt(KEY_RECENT_MESSAGES, 10).apply()
            }
            prefs.edit().putInt(KEY_SCHEMA_VERSION, CURRENT_SCHEMA_VERSION).apply()
        }
        val defaults = InferenceSettings()
        return defaults.copy(
            temperature = prefs.getFloat(KEY_TEMPERATURE, defaults.temperature.toFloat()).toDouble(),
            topP = if (prefs.contains(KEY_TOP_P)) prefs.getFloat(KEY_TOP_P, (defaults.topP ?: .9).toFloat()).toDouble() else defaults.topP,
            topK = if (prefs.contains(KEY_TOP_K)) prefs.getInt(KEY_TOP_K, defaults.topK ?: 40) else defaults.topK,
            minP = if (prefs.contains(KEY_MIN_P)) prefs.getFloat(KEY_MIN_P, (defaults.minP ?: 0.0).toFloat()).toDouble() else defaults.minP,
            repeatPenalty = if (prefs.contains(KEY_REPEAT)) prefs.getFloat(KEY_REPEAT, (defaults.repeatPenalty ?: 1.1).toFloat()).toDouble() else defaults.repeatPenalty,
            maxNewTokens = prefs.getInt(KEY_MAX_TOKENS, defaults.maxNewTokens).coerceAtLeast(1),
            contextLength = if (prefs.contains(KEY_CONTEXT)) prefs.getInt(KEY_CONTEXT, defaults.contextLength ?: 8192).coerceAtLeast(1) else defaults.contextLength,
            recentMessages = prefs.getInt(KEY_RECENT_MESSAGES, defaults.recentMessages).coerceAtLeast(0),
            seed = if (prefs.contains(KEY_SEED)) prefs.getLong(KEY_SEED, defaults.seed ?: 0L) else defaults.seed,
            stopSequences = prefs.getString(KEY_STOPS, null)?.split("\u001f")?.filter { it.isNotEmpty() } ?: defaults.stopSequences,
            maxActionSteps = prefs.getInt(KEY_MAX_ACTION_STEPS, defaults.maxActionSteps).coerceAtLeast(0),
            enableThinking = false,
        ).also { InferenceSettingsStore.current = it }
    }
}
