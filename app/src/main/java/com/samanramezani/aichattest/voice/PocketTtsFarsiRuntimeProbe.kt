package com.samanramezani.aichattest.voice

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import org.json.JSONObject
import java.io.File
import java.util.zip.ZipFile

/**
 * Contract/preflight test for the published Pocket TTS Farsi v2 ONNX package.
 * Checks actual graph signatures against the upstream manifest, not merely that
 * ONNX files can be opened. Dynamic sequence/audio dimensions are allowed; fixed
 * model dimensions, names, dtypes, and state wiring must match the reference.
 */
internal class PocketTtsFarsiRuntimeProbe(private val modelDir: File) {
    data class Report(
        val ok: Boolean,
        val detail: String,
        val graphInputs: Map<String, List<String>> = emptyMap(),
    )

    fun inspect(): Report {
        val required = listOf(
            "manifest.json", "weights.npz", "decode_state_init.npz", "tokenizer_ph.model",
            "male_hello.wav", "flow_lm_step.onnx", "flow_lm_step.onnx.data",
            "g2p_encoder.onnx", "g2p_decoder.onnx", "g2p_decoder.onnx.data",
            "mimi_encoder.onnx", "mimi_encoder.onnx.data",
            "mimi_decoder_step_kv.onnx", "mimi_decoder_step_kv.onnx.data", "normalize_fa.py",
        )
        val missing = required.filter { !File(modelDir, it).isFile || File(modelDir, it).length() == 0L }
        if (missing.isNotEmpty()) {
            return Report(false, "POCKET-TTS-RUNTIME-001: فایل‌های لازم موجود نیستند: ${missing.joinToString()}")
        }

        return try {
            val manifest = JSONObject(File(modelDir, "manifest.json").readText(Charsets.UTF_8))
            val constants = manifest.optJSONObject("constants") ?: error("manifest.json فاقد constants است")
            val ldim = constants.optInt("ldim", 0)
            val dim = constants.optInt("dim", 0)
            val layers = constants.optInt("layers", 0)
            val heads = constants.optInt("heads", 0)
            val headDim = constants.optInt("dim_per_head", 0)
            val capacity = constants.optInt("cache_capacity", 0)
            require(constants.optInt("sample_rate", 0) == 24_000) { "sample_rate باید 24000 باشد." }
            require(ldim == 32 && dim == 1024 && layers == 6 && heads == 16 && headDim == 64 && capacity >= 256) {
                "ثابت‌های مدل با قرارداد Pocket TTS v2 سازگار نیستند: ldim=$ldim dim=$dim layers=$layers heads=$heads headDim=$headDim capacity=$capacity"
            }

            val arrays = ZipFile(File(modelDir, "weights.npz")).use { zip ->
                zip.entries().asSequence().map { it.name }.toSet()
            }
            val requiredArrays = setOf("lut_weight.npy", "speaker_proj.npy", "bos_before_voice.npy", "emb_std.npy", "emb_mean.npy")
            require((requiredArrays - arrays).isEmpty()) { "weights.npz فاقد آرایه‌هاست: ${(requiredArrays - arrays).joinToString()}" }
            val stateNames = ZipFile(File(modelDir, "decode_state_init.npz")).use { zip ->
                zip.entries().asSequence().map { it.name }.filter { it.endsWith(".npy") }
                    .map { it.removeSuffix(".npy") }.toSet()
            }
            require(stateNames.size == 27) { "decode_state_init باید 27 حالت decoder داشته باشد؛ تعداد فعلی=${stateNames.size}" }

            val env = OrtEnvironment.getEnvironment()
            val options = OrtSession.SessionOptions().apply {
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                setIntraOpNumThreads(2)
                setInterOpNumThreads(1)
            }
            val inputs = linkedMapOf<String, List<String>>()
            try {
                fun session(file: String, check: (OrtSession) -> Unit) {
                    env.createSession(File(modelDir, file).absolutePath, options).use { s ->
                        inputs[file] = s.inputNames.toList()
                        check(s)
                    }
                }
                fun signature(s: OrtSession, ins: Set<String>, outs: Set<String>, graph: String) {
                    val actualIn = s.inputNames.toSet()
                    val actualOut = s.outputNames.toSet()
                    require(actualIn == ins) {
                        "$graph ورودی‌ها با قرارداد مرجع فرق دارند؛ expected=${ins.sorted()} actual=${actualIn.sorted()}"
                    }
                    require(actualOut == outs) {
                        "$graph خروجی‌ها با قرارداد مرجع فرق دارند؛ expected=${outs.sorted()} actual=${actualOut.sorted()}"
                    }
                }
                fun tensor(s: OrtSession, input: Boolean, name: String, dtype: OnnxJavaType, dims: List<Long?>) {
                    val node = (if (input) s.inputInfo else s.outputInfo)[name]
                        ?: error("${s} فاقد گره ${if (input) "input" else "output"} به نام $name است")
                    val info = node.info as? TensorInfo ?: error("$name باید Tensor باشد.")
                    require(info.type == dtype) { "$name dtype نامعتبر است: expected=$dtype actual=${info.type}" }
                    val actual = info.shape
                    require(actual.size == dims.size) { "$name rank نامعتبر است: expected=$dims actual=${actual.contentToString()}" }
                    dims.forEachIndexed { i, expected ->
                        if (expected != null) require(actual[i] == expected || actual[i] < 0) {
                            "$name dimension[$i] نامعتبر است: expected=$expected actual=${actual.contentToString()}"
                        }
                    }
                }
                session("flow_lm_step.onnx") { s ->
                    signature(s, setOf("sequence", "text_emb", "offset", "noise", "cache"), setOf("latent", "eos", "new_kv"), "FlowLM")
                    tensor(s, true, "sequence", OnnxJavaType.FLOAT, listOf(1, null, ldim.toLong()))
                    tensor(s, true, "text_emb", OnnxJavaType.FLOAT, listOf(1, null, dim.toLong()))
                    tensor(s, true, "offset", OnnxJavaType.INT64, emptyList())
                    tensor(s, true, "noise", OnnxJavaType.FLOAT, listOf(1, ldim.toLong()))
                    tensor(s, true, "cache", OnnxJavaType.FLOAT, listOf(layers.toLong(), 2, capacity.toLong(), heads.toLong(), headDim.toLong()))
                    tensor(s, false, "latent", OnnxJavaType.FLOAT, listOf(1, ldim.toLong()))
                    tensor(s, false, "eos", OnnxJavaType.FLOAT, listOf(1, 1))
                    tensor(s, false, "new_kv", OnnxJavaType.FLOAT, listOf(layers.toLong(), 2, null, heads.toLong(), headDim.toLong()))
                }
                session("g2p_encoder.onnx") { s ->
                    signature(s, setOf("input_ids"), setOf("hidden"), "G2P encoder")
                    tensor(s, true, "input_ids", OnnxJavaType.INT64, listOf(1, null))
                    tensor(s, false, "hidden", OnnxJavaType.FLOAT, listOf(1, null, 512))
                }
                session("g2p_decoder.onnx") { s ->
                    signature(s, setOf("decoder_input_ids", "encoder_hidden"), setOf("logits"), "G2P decoder")
                    tensor(s, true, "decoder_input_ids", OnnxJavaType.INT64, listOf(1, null))
                    tensor(s, true, "encoder_hidden", OnnxJavaType.FLOAT, listOf(1, null, 512))
                    tensor(s, false, "logits", OnnxJavaType.FLOAT, listOf(1, null, 384))
                }
                session("mimi_encoder.onnx") { s ->
                    signature(s, setOf("audio"), setOf("latent"), "Mimi encoder")
                    tensor(s, true, "audio", OnnxJavaType.FLOAT, listOf(1, 1, null))
                    tensor(s, false, "latent", OnnxJavaType.FLOAT, listOf(1, null, ldim.toLong()))
                }
                session("mimi_decoder_step_kv.onnx") { s ->
                    val expectedInputs = stateNames + "latent"
                    val actualInputs = s.inputNames.toSet()
                    require(actualInputs == expectedInputs) {
                        "Mimi decoder ورودی‌های state را درست ندارد؛ missing=${(expectedInputs - actualInputs).sorted()} unexpected=${(actualInputs - expectedInputs).sorted()}"
                    }
                    val outputs = s.outputNames.toSet()
                    require("audio" in outputs && "kv0" in outputs && "kv1" in outputs) {
                        "Mimi decoder باید audio و kv0/kv1 را برگرداند؛ outputs=${outputs.sorted()}"
                    }
                    require(outputs.count { it.endsWith("o") && it != "audio" } == stateNames.size - 2) {
                        "تعداد خروجی‌های state کوچک با 27 state ورودی تطبیق ندارد: ${outputs.sorted()}"
                    }
                    tensor(s, true, "latent", OnnxJavaType.FLOAT, listOf(1, 1, ldim.toLong()))
                    tensor(s, false, "audio", OnnxJavaType.FLOAT, listOf(1, 1, 1920))
                    tensor(s, false, "kv0", OnnxJavaType.FLOAT, listOf(2, 1, null, 8, 64))
                    tensor(s, false, "kv1", OnnxJavaType.FLOAT, listOf(2, 1, null, 8, 64))
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
            require(String(header, 0, 4, Charsets.US_ASCII) == "RIFF" && String(header, 8, 4, Charsets.US_ASCII) == "WAVE") {
                "صدای مرجع male_hello.wav قالب WAV معتبر ندارد"
            }
            Report(
                true,
                "POCKET-TTS-CONTRACT-PASS: امضای ورودی/خروجی، dtype، rank و ابعاد ثابت هر 5 گراف با قرارداد مدل مرجع بررسی شد؛ 27 حالت decoder و فایل‌های وزن/صدا معتبرند. این تست قرارداد گراف‌هاست؛ تست تولید واقعی PCM جداگانه انجام می‌شود.",
                inputs,
            )
        } catch (t: Throwable) {
            Report(false, "POCKET-TTS-CONTRACT-FAIL: اعتبارسنجی قرارداد مدل ناموفق بود: ${t.message ?: t.javaClass.simpleName}")
        }
    }
}
