package com.samanramezani.aichattest.voice

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File

/**
 * Performs a real ONNX Runtime load check before a Pocket TTS bundle is installed.
 *
 * Creating each session makes ONNX Runtime resolve the graph and its external tensor
 * data. This is deliberately a load/contract probe, not a claim that synthesis has
 * been exercised; actual text-to-audio execution belongs in the synthesis tests.
 */
internal class PocketTtsFarsiRuntimeProbe(private val directory: File) {
    internal data class Result(val ok: Boolean, val detail: String)

    fun inspect(): Result {
        val graphNames = listOf(
            "flow_lm_step.onnx",
            "g2p_encoder.onnx",
            "g2p_decoder.onnx",
            "mimi_encoder.onnx",
            "mimi_decoder_step_kv.onnx",
        )
        val missing = graphNames.filter { !File(directory, it).isFile }
        if (missing.isNotEmpty()) {
            return Result(false, "گراف‌های ONNX پیدا نشدند: ${missing.joinToString("، ")}")
        }

        val options = OrtSession.SessionOptions()
        val sessions = mutableListOf<OrtSession>()
        return try {
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
            options.setIntraOpNumThreads(1)
            options.setInterOpNumThreads(1)
            val env = OrtEnvironment.getEnvironment()
            for (name in graphNames) {
                val file = File(directory, name)
                val session = env.createSession(file.absolutePath, options)
                sessions += session
                if (session.inputNames.isEmpty() || session.outputNames.isEmpty()) {
                    error("$name ورودی یا خروجی ندارد.")
                }
            }
            Result(
                true,
                "هر پنج گراف ONNX و داده‌های خارجی آن‌ها با ONNX Runtime بارگذاری شدند؛ این بررسی هنوز تولید واقعی صوت را آزمایش نمی‌کند.",
            )
        } catch (t: Throwable) {
            Result(
                false,
                "بارگذاری ONNX ناموفق بود: ${t.message ?: t.javaClass.simpleName}",
            )
        } finally {
            sessions.asReversed().forEach { runCatching { it.close() } }
            runCatching { options.close() }
        }
    }
}
