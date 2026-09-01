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

    fun inferenceFailure(throwable: Throwable): ModelError =
        if (throwable is OutOfMemoryError) {
            ModelError.OutOfMemory("Not enough memory for local inference", throwable)
        } else {
            ModelError.Inference("Local model inference failed", throwable)
        }
}
