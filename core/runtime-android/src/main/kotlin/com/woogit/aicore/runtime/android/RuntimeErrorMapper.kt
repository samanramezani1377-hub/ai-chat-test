package com.woogit.aicore.runtime.android

import com.woogit.aicore.domain.ModelError

/** Converts concrete runtime failures into the domain error taxonomy. */
internal object RuntimeErrorMapper {
    fun loadFailure(throwable: Throwable, path: String): ModelError =
        if (throwable is OutOfMemoryError) {
            ModelError.OutOfMemory("Not enough memory to load model: $path", throwable)
        } else {
            ModelError.LoadFailed("Unable to load local model: $path", throwable)
        }

    /**
     * Maps the native GPU-only activation result without misreporting unsupported
     * OpenCL devices as generic model or inference failures.
     *
     * Native codes are intentionally stable at this boundary:
     * 1 = model load failed, 4 = invalid GPU-layer request,
     * 5 = no usable OpenCL GPU backend, 6 = GPU-only residency validation failed.
     */
    fun nativeLoadFailure(code: Int): ModelError = when (code) {
        4 -> ModelError.RuntimeUnavailable(
            "Invalid GPU-only configuration. Activate the model with full OpenCL GPU offload."
        )
        5 -> ModelError.RuntimeUnavailable(
            "This device's OpenCL GPU is not supported by the current GPU-only backend. " +
                "CPU fallback is disabled; no model was activated."
        )
        6 -> ModelError.RuntimeUnavailable(
            "The model could not remain fully resident on the OpenCL GPU. " +
                "CPU weight fallback is disabled; no model was activated."
        )
        else -> ModelError.LoadFailed("llama.cpp failed to load the model (code=$code)")
    }

    fun inferenceFailure(throwable: Throwable): ModelError =
        if (throwable is OutOfMemoryError) {
            ModelError.OutOfMemory("Not enough memory for local inference", throwable)
        } else {
            ModelError.Inference("Local model inference failed", throwable)
        }
}
