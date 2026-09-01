package com.woogit.aicore.domain

import java.nio.file.Path

/** Supported on-disk model container formats. */
enum class ModelFormat { GGUF, UNKNOWN }

enum class ModelState {
    NOT_IMPORTED, IMPORTING, VALIDATING, IMPORTED, READY, LOADING, ACTIVE, UNLOADING, INVALID, FAILED
}

enum class Quantization {
    F32, F16, Q2_K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_0, Q4_1, Q4_K_S, Q4_K_M,
    Q5_0, Q5_1, Q5_K_S, Q5_K_M, Q6_K, Q8_0, UNKNOWN
}

data class ModelMetadata(
    val architecture: String? = null,
    val name: String? = null,
    val contextLength: Long? = null,
    val embeddingLength: Long? = null,
    val blockCount: Long? = null,
    val tokenizerModel: String? = null,
    val raw: Map<String, Any?> = emptyMap()
)

data class RuntimeCompatibility(
    val supported: Boolean,
    val reason: String? = null
)

data class ModelDescriptor(
    val id: String,
    val displayName: String,
    val path: Path,
    val format: ModelFormat,
    val quantization: Quantization,
    val sizeBytes: Long,
    val metadata: ModelMetadata,
    val validation: ValidationStatus,
    val runtimeCompatibility: RuntimeCompatibility,
    val state: ModelState
)

enum class ValidationStatus { UNKNOWN, VALID, INVALID }

data class ModelInspection(
    val format: ModelFormat,
    val version: Int? = null,
    val metadata: ModelMetadata = ModelMetadata(),
    val quantization: Quantization = Quantization.UNKNOWN,
    val sizeBytes: Long,
    val tensorCount: Long = 0,
    val error: ModelError? = null
)

sealed class ModelError(
    open val code: String,
    open val message: String,
    open val cause: Throwable? = null
) {
    data class FileAccess(override val message: String, override val cause: Throwable? = null) : ModelError("FILE_ACCESS", message, cause)
    data class UnsupportedFormat(override val message: String) : ModelError("UNSUPPORTED_FORMAT", message)
    data class InvalidModel(override val message: String) : ModelError("INVALID_MODEL", message)
    data class InvalidMetadata(override val message: String) : ModelError("INVALID_METADATA", message)
    data class UnsupportedArchitecture(override val message: String) : ModelError("UNSUPPORTED_ARCHITECTURE", message)
    data class UnsupportedQuantization(override val message: String) : ModelError("UNSUPPORTED_QUANTIZATION", message)
    data class RuntimeUnavailable(override val message: String) : ModelError("RUNTIME_UNAVAILABLE", message)
    data class LoadFailed(override val message: String, override val cause: Throwable? = null) : ModelError("LOAD_FAILED", message, cause)
    data class OutOfMemory(override val message: String, override val cause: Throwable? = null) : ModelError("OUT_OF_MEMORY", message, cause)
    data class ImportCancelled(override val message: String = "Model import was cancelled") : ModelError("IMPORT_CANCELLED", message)
    data class Storage(override val message: String, override val cause: Throwable? = null) : ModelError("STORAGE_ERROR", message, cause)
    data class Inference(override val message: String, override val cause: Throwable? = null) : ModelError("INFERENCE_ERROR", message, cause)
}

sealed class ModelResult<out T> {
    data class Success<T>(val value: T) : ModelResult<T>()
    data class Failure(val error: ModelError) : ModelResult<Nothing>()
}
