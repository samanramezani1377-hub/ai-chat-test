package com.samanramezani.aichattest.voice

import com.k2fsa.sherpa.onnx.FeatureConfig
import com.k2fsa.sherpa.onnx.OnlineModelConfig
import com.k2fsa.sherpa.onnx.OnlineRecognizer
import com.k2fsa.sherpa.onnx.OnlineRecognizerConfig
import com.k2fsa.sherpa.onnx.OnlineTransducerModelConfig
import com.k2fsa.sherpa.onnx.OfflineModelConfig
import com.k2fsa.sherpa.onnx.OfflineNemoEncDecCtcModelConfig
import com.k2fsa.sherpa.onnx.OfflineQwen3AsrModelConfig
import com.k2fsa.sherpa.onnx.OfflineRecognizer
import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig
import com.k2fsa.sherpa.onnx.OnlineStream
import java.io.File

/** Streaming local ASR for sherpa-onnx online transducer packages. */
data class AsrUpdate(val text: String, val endpoint: Boolean)

interface LocalAsr {
    fun accept(samples: FloatArray, sampleRate: Int = 16000, speech: Boolean = true): AsrUpdate
    fun close()
}

class LocalStreamingAsr(private val entry: VoiceModelEntry) : LocalAsr {
    private val recognizer: OnlineRecognizer
    private val stream: OnlineStream

    init {
        val model = OnlineModelConfig().apply {
            transducer = OnlineTransducerModelConfig().apply {
                encoder = requireNotNull(entry.encoder).absolutePath
                decoder = requireNotNull(entry.decoder).absolutePath
                joiner = requireNotNull(entry.joiner).absolutePath
            }
            tokens = requireNotNull(entry.tokensFile).absolutePath
            numThreads = 2
            provider = "cpu"
            modelType = entry.modelType
        }
        recognizer = OnlineRecognizer(config = OnlineRecognizerConfig().apply {
            featConfig = FeatureConfig().apply { sampleRate = 16000; featureDim = 80 }
            modelConfig = model
            enableEndpoint = true
        })
        stream = recognizer.createStream()
    }

    override fun accept(samples: FloatArray, sampleRate: Int, speech: Boolean): AsrUpdate {
        stream.acceptWaveform(samples, sampleRate)
        while (recognizer.isReady(stream)) recognizer.decode(stream)
        val text = recognizer.getResult(stream).text
        val endpoint = recognizer.isEndpoint(stream)
        if (endpoint) recognizer.reset(stream)
        return AsrUpdate(text, endpoint)
    }

    override fun close() {
        runCatching { stream.release() }
        recognizer.release()
    }
}

/**
 * Offline Qwen3-ASR adapter for multilingual Persian speech. It accumulates the current
 * utterance while microphone capture stays live, then decodes after 600 ms of silence.
 * VAD-based barge-in remains immediate even though transcript finalization is offline.
 */
/**
 * Tiny Persian Shenava Rizeh-Pizeh FastConformer CTC model (6.9M parameters).
 * This export is non-streaming: keep the microphone live, buffer the utterance,
 * and decode locally after endpoint silence.
 */
class LocalNemoCtcAsr(private val entry: VoiceModelEntry) : LocalAsr {
    private val recognizer: OfflineRecognizer
    private val utterance = ArrayList<Float>(16000 * 4)
    private var hasSpeech = false
    private var silenceSamples = 0
    private val maxUtteranceSamples = 16000 * 20
    private val endSilenceSamples = 16000 * 3 / 5

    init {
        val modelPath = entry.modelFile.absolutePath
        val tokensPath = requireNotNull(entry.tokensFile).absolutePath
        require(entry.modelFile.isFile && entry.modelFile.length() > 1024) {
            "VOICE-STT-CTC-001: فایل model.onnx وجود ندارد یا خالی است."
        }
        require(File(tokensPath).isFile) { "VOICE-STT-CTC-002: فایل tokens.txt پیدا نشد." }
        recognizer = OfflineRecognizer(config = OfflineRecognizerConfig().apply {
            featConfig = FeatureConfig().apply { sampleRate = 16000; featureDim = 80 }
            modelConfig = OfflineModelConfig().apply {
                nemo = OfflineNemoEncDecCtcModelConfig().apply { model = modelPath }
                tokens = tokensPath
                numThreads = 2
                debug = false
                provider = "cpu"
            }
            decodingMethod = "greedy_search"
        })
    }

    /**
     * Bounded rolling-window probe used only to spot "دینا" before the current utterance ends.
     * It creates a temporary stream and never appends probe samples to the persistent utterance buffer.
     */
    fun recognizeWindow(samples: FloatArray, sampleRate: Int = 16000): String {
        require(sampleRate == 16000) { "VOICE-STT-CTC-003: مدل Shenava به صدای ۱۶ کیلوهرتز نیاز دارد." }
        if (samples.size < 8000) return ""
        val stream = recognizer.createStream()
        return try {
            stream.acceptWaveform(samples, sampleRate)
            recognizer.decode(stream)
            recognizer.getResult(stream).text.trim()
        } finally {
            stream.release()
        }
    }

    override fun accept(samples: FloatArray, sampleRate: Int, speech: Boolean): AsrUpdate {
        require(sampleRate == 16000) { "VOICE-STT-CTC-003: مدل Shenava به صدای ۱۶ کیلوهرتز نیاز دارد." }
        if (speech) {
            hasSpeech = true
            silenceSamples = 0
        } else if (hasSpeech) {
            silenceSamples += samples.size
        }
        if (hasSpeech) {
            val remaining = maxUtteranceSamples - utterance.size
            val count = minOf(remaining, samples.size)
            for (i in 0 until count) utterance.add(samples[i])
        }
        if (!hasSpeech || (silenceSamples < endSilenceSamples && utterance.size < maxUtteranceSamples)) {
            return AsrUpdate("", false)
        }
        val audio = FloatArray(utterance.size) { utterance[it] }
        utterance.clear()
        hasSpeech = false
        silenceSamples = 0
        val stream = recognizer.createStream()
        return try {
            stream.acceptWaveform(audio, sampleRate)
            recognizer.decode(stream)
            AsrUpdate(recognizer.getResult(stream).text.trim(), true)
        } finally {
            stream.release()
        }
    }

    override fun close() {
        utterance.clear()
        recognizer.release()
    }
}

class LocalQwen3Asr(private val entry: VoiceModelEntry) : LocalAsr {
    private val recognizer: OfflineRecognizer
    private val utterance = ArrayList<Float>(16000 * 8)
    private var hasSpeech = false
    private var silenceSamples = 0
    private val maxUtteranceSamples = 16000 * 30
    private val endSilenceSamples = 16000 * 3 / 5

    init {
        val qwen = OfflineQwen3AsrModelConfig().apply {
            convFrontend = requireNotNull(entry.convFrontend).absolutePath
            encoder = requireNotNull(entry.encoder).absolutePath
            decoder = requireNotNull(entry.decoder).absolutePath
            tokenizer = requireNotNull(entry.tokenizerDirectory).absolutePath
            maxTotalLen = 512
            maxNewTokens = 256
            temperature = 1e-6f
            topP = 0.8f
        }
        recognizer = OfflineRecognizer(config = OfflineRecognizerConfig().apply {
            featConfig = FeatureConfig().apply { sampleRate = 16000; featureDim = 80 }
            modelConfig = OfflineModelConfig().apply {
                qwen3Asr = qwen
                numThreads = 2
                debug = false
                provider = "cpu"
            }
        })
    }

    override fun accept(samples: FloatArray, sampleRate: Int, speech: Boolean): AsrUpdate {
        require(sampleRate == 16000) { "VOICE-STT-QWEN-001: مدل Qwen3-ASR به صدای ۱۶ کیلوهرتز نیاز دارد." }
        if (speech) {
            hasSpeech = true
            silenceSamples = 0
        } else if (hasSpeech) {
            silenceSamples += samples.size
        }
        if (hasSpeech) {
            // Bound memory even if an upstream caller sends unusually large frames.
            val remaining = maxUtteranceSamples - utterance.size
            val count = minOf(remaining, samples.size)
            for (i in 0 until count) utterance.add(samples[i])
        }
        if (!hasSpeech || (silenceSamples < endSilenceSamples && utterance.size < maxUtteranceSamples)) {
            return AsrUpdate("", false)
        }
        val audio = FloatArray(utterance.size) { utterance[it] }
        utterance.clear()
        hasSpeech = false
        silenceSamples = 0
        val stream = recognizer.createStream()
        return try {
            stream.acceptWaveform(audio, sampleRate)
            recognizer.decode(stream)
            AsrUpdate(recognizer.getResult(stream).text.trim(), true)
        } finally {
            stream.release()
        }
    }

    override fun close() {
        utterance.clear()
        recognizer.release()
    }
}
