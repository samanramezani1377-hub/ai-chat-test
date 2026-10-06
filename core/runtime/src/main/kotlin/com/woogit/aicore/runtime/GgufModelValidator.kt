package com.woogit.aicore.runtime

import com.woogit.aicore.domain.ModelError
import com.woogit.aicore.domain.ModelFormat
import com.woogit.aicore.domain.ModelInspection
import com.woogit.aicore.domain.ModelResult
import com.woogit.aicore.domain.Quantization
import com.woogit.aicore.domain.ValidationStatus

/** Validates GGUF profiles supported by the local llama.cpp Android runtime. */
class GgufModelValidator(
    private val supportedArchitectures: Set<String> = DEFAULT_ARCHITECTURES,
    private val supportedQuantizations: Set<Quantization> = DEFAULT_QUANTIZATIONS
) : ModelValidator {
    override suspend fun validate(inspection: ModelInspection): ModelResult<ValidationStatus> {
        inspection.error?.let { return ModelResult.Failure(it) }

        if (inspection.format != ModelFormat.GGUF) {
            return ModelResult.Failure(ModelError.UnsupportedFormat("Only GGUF models are supported"))
        }
        if (inspection.version !in SUPPORTED_GGUF_VERSIONS) {
            return ModelResult.Failure(ModelError.InvalidModel("Unsupported GGUF version: ${inspection.version}"))
        }
        if (inspection.sizeBytes <= 0L) {
            return ModelResult.Failure(ModelError.InvalidModel("GGUF model has an invalid file size"))
        }
        if (inspection.tensorCount <= 0L) {
            return ModelResult.Failure(ModelError.InvalidModel("GGUF model contains no tensors"))
        }

        val architecture = inspection.metadata.architecture?.trim()?.lowercase()
            ?: return ModelResult.Failure(ModelError.InvalidMetadata("GGUF model does not declare an architecture"))
        if (!isSupportedArchitecture(architecture, inspection.metadata.name)) {
            return ModelResult.Failure(ModelError.UnsupportedArchitecture("Unsupported model architecture: ${inspection.metadata.architecture}"))
        }

        if (inspection.quantization == Quantization.UNKNOWN) {
            return ModelResult.Failure(ModelError.UnsupportedQuantization("Unable to determine model quantization"))
        }
        if (inspection.quantization !in supportedQuantizations) {
            return ModelResult.Failure(ModelError.UnsupportedQuantization("Unsupported model quantization: ${inspection.quantization}"))
        }

        inspection.metadata.contextLength?.takeIf { it <= 0L }?.let {
            return ModelResult.Failure(ModelError.InvalidMetadata("GGUF context length is invalid"))
        }
        inspection.metadata.embeddingLength?.takeIf { it <= 0L }?.let {
            return ModelResult.Failure(ModelError.InvalidMetadata("GGUF embedding length is invalid"))
        }
        inspection.metadata.blockCount?.takeIf { it <= 0L }?.let {
            return ModelResult.Failure(ModelError.InvalidMetadata("GGUF block count is invalid"))
        }

        return ModelResult.Success(ValidationStatus.VALID)
    }

    private fun isSupportedArchitecture(architecture: String, modelName: String?): Boolean {
        if (architecture in supportedArchitectures) return true

        // MiniCPM5 intentionally uses the standard LlamaForCausalLM architecture.
        // Only accept the generic "llama" architecture when the GGUF identifies itself
        // as MiniCPM5, so arbitrary unsupported Llama-family imports are not opened up.
        return architecture == "llama" && modelName?.contains("minicpm5", ignoreCase = true) == true
    }

    companion object {
        private val SUPPORTED_GGUF_VERSIONS = setOf(1, 2, 3)

        // Qwen3/Qwen3.5 and LFM2 are implemented by the pinned llama.cpp runtime.
        // MiniCPM5 uses standard general.architecture = "llama" and is allow-listed
        // separately by model name above.
        private val DEFAULT_ARCHITECTURES = setOf("qwen3", "qwen35", "lfm2")

        // MiniCPM5 official GGUF releases are available as Q4_K_M, Q8_0 and F16.
        // Keep Q4_K_M/Q5/Q6 available for future llama.cpp-compatible models too.
        private val DEFAULT_QUANTIZATIONS = setOf(
            Quantization.Q4_K_S,
            Quantization.Q4_K_M,
            Quantization.Q5_K_S,
            Quantization.Q5_K_M,
            Quantization.Q6_K,
            Quantization.Q8_0
        )
    }
}
