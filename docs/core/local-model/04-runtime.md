# 04 — Runtime

## Runtime contract

`:core:runtime` remains the platform-neutral model lifecycle layer. The Android implementation in `:core:runtime-android` is `OnnxRuntimeGenAiAndroidRuntimeAdapter`, and the application composition root injects it as the primary local runtime.

The old llama.cpp adapter, JNI bridge, CMake build, GGML/OpenCL patches and native inference sources have been removed from this branch. Local inference uses the pinned ONNX Runtime GenAI Android Java/JNI SDK (v0.11.4) for `arm64-v8a`. The SDK AAR is downloaded during the Gradle build and verified against a fixed SHA-256 before its Java and native artifacts are used.

## Activation

A model can activate only when its format is `ONNX_GENAI`, its persisted validation state is valid, and its directory contains `genai_config.json`. Activation creates the GenAI `Model` and tokenizer objects before the model is persisted as active. If loading fails, partially created native objects are closed and the active-model pointer is cleared.

Legacy GGUF registry entries are not loaded by this runtime. A stale persisted GGUF active pointer is cleared during restoration, and the settings UI disables activation for those entries while allowing deletion.

## Generation

The adapter formats the conversation, tokenizes it with the model's tokenizer, configures bounded generation options, then emits decoded token chunks to the existing chat flow. Generation is bounded by the model context and requested output budget; stop strings and cancellation are handled by the adapter. Runtime metrics are exposed through the existing `RuntimeMetrics` contract.

The chat UI, history, provider routing and conversation orchestration continue to use the shared `ModelRuntime` contract. The model package must be exported for ONNX Runtime GenAI; GGUF is not a compatible artifact and cannot be converted by merely renaming the file.

## Model package requirements

A ZIP package must include:
- `genai_config.json`
- one or more `.onnx` graph files (including any external tensor data referenced by the graph)
- tokenizer assets supported by the model package, such as `tokenizer.json`, `tokenizer.model`, `tokenizer_config.json` or `vocab.json`

The importer rejects path traversal, limits the number of ZIP entries and caps expanded data at 32 GiB. Runtime/provider availability still depends on the execution providers encoded in the package and supported by the packaged GenAI runtime.

## Validation status

CI covers JVM lifecycle/importer tests, Gradle compilation, APK assembly and native library packaging checks. A successful CI build does not prove that every external ONNX model will load or generate correctly on a physical device; a smoke test with the exact model package on the target phone remains necessary.
