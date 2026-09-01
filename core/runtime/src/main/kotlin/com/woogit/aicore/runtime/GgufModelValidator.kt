package com.woogit.aicore.runtime

import com.woogit.aicore.domain.ModelError
import com.woogit.aicore.domain.ModelFormat
import com.woogit.aicore.domain.ModelInspection
import com.woogit.aicore.domain.ModelResult
import com.woogit.aicore.domain.ModelValidator
import com.woogit.aicore.domain.Quantization
import com.woogit.aicore.domain.ValidationStatus

/**
 * Validates the structural information extracted from a GGUF model.
 *
 * This validator deliberately does not claim runtime compatibility. Runtime support is
 * established by the concrete runtime adapter that will load the model.
 */
class GgufModelValidator(
    private val supportedArchitectures: Set<String> = DEFAULT_ARCHITECTURES,
    private val supportedQuantizations: Set<Quantization> = DEFAULT_QUANTIZATIONS
) : ModelValidator {
    override suspend fun validate(inspection: ModelInspection): ModelResult<ValidationStatus> {
        if (inspection.error != null) {
            return ModelResult.Failure(inspection.error)
        }

        if (inspection.format != ModelFormat.GGUF) {
            return ModelResult.Failure(
                ModelError.UnsupportedFormat("Only GGUF models are supported")
            )
        }

        if (inspection.version !in SUPPORTED_GGUF_VERSIONS) {
            return ModelResult.Failure(
                ModelError.InvalidModel("Unsupported GGUF version: ${inspection.version}")
            )
        }

        if (inspection.sizeBytes <= 0L) {
            return ModelResult.Failure(
                ModelError.InvalidModel("GGUF model has an invalid file size")
            )
        }

        if (inspection.tensorCount <= 0L) {
            return ModelResult.Failure(
                ModelError.InvalidModel("GGUF model contains no tensors")
            )
        }

        val architecture = inspection.metadata.architecture?.trim()?.lowercase()
            ?: return ModelResult.Failure(
                ModelError.InvalidMetadata("GGUF model does not declare an architecture")
            )

        if (architecture !in supportedArchitectures) {
            return ModelResult.Failure(
                ModelError.UnsupportedArchitecture(
                    "Unsupported model architecture: ${inspection.metadata.architecture}"
                )
            )
        }

        if (inspection.quantization == Quantization.UNKNOWN) {
            return ModelResult.Failure(
                ModelError.UnsupportedQuantization("Unable to determine model quantization")
            )
        }

        if (inspection.quantization !in supportedQuantizations) {
            return ModelResult.Failure(
                ModelError.UnsupportedQuantization(
                    "Unsupported model quantization: ${inspection.quantization}"
                )
            )
        }

        val contextLength = inspection.metadata.contextLength
        if (contextLength != null && contextLength <= 0L) {
            return ModelResult.Failure(
                ModelError.InvalidMetadata("GGUF context length is invalid")
            )
        }

        val embeddingLength = inspection.metadata.embeddingLength
        if (embeddingLength != null && embeddingLength <= 0L) {
            return ModelResult.Failure(
                ModelError.InvalidMetadata("GGUF embedding length is invalid")
            )
        }

        val blockCount = inspection.metadata.blockCount
        if (blockCount != null && blockCount <= 0L) {
            return ModelResult.Failure(
                ModelError.InvalidMetadata("GGUF block count is invalid")
            )
        }

        return ModelResult.Success(ValidationStatus.VALID)
    }

    companion object {
        private val SUPPORTED_GGUF_VERSIONS = setOf(1, 2, 3)

        // The initial application target is Qwen3. Other architectures can be enabled by
        // dependency injection when a concrete runtime supports them.
        private val DEFAULT_ARCHITECTURES = setOf("qwen3")

        // Q6_K is the selected initial model target. F16/F32 are kept for validation so a
        // compatible runtime can explicitly support them without changing the validator.
        private val DEFAULT_QUANTIZATIONS = setOf(
            Quantization.Q6_K,
            Quantization.F16,
            Quantization.F32
        )
    }
}
