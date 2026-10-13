package com.samanramezani.aichattest.voice

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.json.JSONObject
import java.io.File
import java.util.zip.ZipFile

/**
 * Validates the actual ONNX graphs and external-data packaging before a bundle can be
 * treated as executable. This is deliberately a preflight only: it does not claim to
 * synthesize audio or replace LocalPiperTts.
 */
internal class PocketTtsFarsiRuntimeProbe(private val modelDir: File) {
    data class Report(
        val ok: Boolean,
        val detail: String,
        val graphInputs: Map<String, List<String>> = emptyMap(),
    )

    fun inspect(): Report {
        val required = listOf(
            "manifest.json",
            "weights.npz",
            "decode_state_init.npz",
            "tokenizer_ph.model",
            "male_hello.wav",
            "flow_lm_step.onnx",
            "flow_lm_step.onnx.data",
            "g2p_encoder.onnx",
            "g2p_decoder.onnx",
            "g2p_decoder.onnx.data",
            "mimi_encoder.onnx",
            "mimi_encoder.onnx.data",
            "mimi_decoder_step_kv.onnx",
            "mimi_decoder_step_kv.onnx.data",
            "normalize_fa.py",
        )
        val missing = required.filter { name ->
            val f = File(modelDir, name)
            !f.isFile || f.length() == 0L
        }
        if (missing.isNotEmpty()) {
            return Report(false, "POCKET-TTS-RUNTIME-001: فایل‌های لازم موجود نیستند: ${missing.joinToString()}")
        }

        return try {
            val manifest = JSONObject(File(modelDir, "manifest.json").readText(Charsets.UTF_8))
            val constants = manifest.optJSONObject("constants")
                ?: error("manifest.json فاقد constants است")
            val expectedSampleRate = constants.optInt("sample_rate", 0)
            require(expectedSampleRate == 24_000) {
                "نرخ نمونه‌برداری مدل نامعتبر است: $expectedSampleRate"
            }
            require(constants.optInt("ldim", 0) > 0 && constants.optInt("dim", 0) > 0) {
                "ابعاد latent/embedding در manifest نامعتبر است"
            }

            // NPZ files are ZIP containers. Verify the required arrays by name rather
            // than trusting only their file sizes.
            ZipFile(File(modelDir, "weights.npz")).use { zip ->
                val entries = zip.entries().asSequence().map { it.name }.toSet()
                val arrays = setOf("lut_weight.npy", "speaker_proj.npy", "bos_before_voice.npy", "emb_std.npy", "emb_mean.npy")
                val absent = arrays - entries
                require(absent.isEmpty()) { "weights.npz فاقد آرایه‌هاست: ${absent.joinToString()}" }
            }
            ZipFile(File(modelDir, "decode_state_init.npz")).use { zip ->
                require(zip.entries().hasMoreElements()) { "decode_state_init.npz خالی است" }
            }

            val env = OrtEnvironment.getEnvironment()
            val options = OrtSession.SessionOptions().apply {
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                setIntraOpNumThreads(2)
                setInterOpNumThreads(1)
            }
            val graphNames = listOf(
                "flow_lm_step.onnx",
                "g2p_encoder.onnx",
                "g2p_decoder.onnx",
                "mimi_encoder.onnx",
                "mimi_decoder_step_kv.onnx",
            )
            val inputs = linkedMapOf<String, List<String>>()
            try {
                graphNames.forEach { name ->
                    val path = File(modelDir, name).absolutePath
                    env.createSession(path, options).use { session ->
                        inputs[name] = session.inputNames.toList()
                        require(session.inputNames.isNotEmpty()) { "$name فاقد ورودی ONNX است" }
                    }
                }
            } finally {
                options.close()
            }

            val wav = File(modelDir, "male_hello.wav")
            val header = wav.inputStream().use { input -> ByteArray(44).also { bytes ->
                var offset = 0
                while (offset < bytes.size) {
                    val count = input.read(bytes, offset, bytes.size - offset)
                    if (count < 0) break
                    offset += count
                }
                require(offset == 44) { "صدای مرجع WAV ناقص است" }
            } }
            require(String(header, 0, 4, Charsets.US_ASCII) == "RIFF" &&
                String(header, 8, 4, Charsets.US_ASCII) == "WAVE") {
                "صدای مرجع male_hello.wav قالب WAV معتبر ندارد"
            }
            Report(
                true,
                "POCKET-TTS-PREFLIGHT-OK: هر پنج گراف ONNX بارگذاری شدند؛ آرایه‌های NPZ، manifest و WAV بررسی شدند. این آزمون فقط سلامت بسته را تأیید می‌کند و تولید صوت هنوز باید توسط موتور استنتاج پیاده‌سازی شود.",
                inputs,
            )
        } catch (t: Throwable) {
            Report(
                false,
                "POCKET-TTS-RUNTIME-002: بارگذاری/اعتبارسنجی مدل ناموفق بود: ${t.message ?: t.javaClass.simpleName}",
            )
        }
    }
}
