# Android GPU Inference Engine — Architecture

## 1. Runtime flow

```text
test-app (Android Activity)
  -> GpuEngineTestApi (Kotlin API)
      -> NativeGpuEngine (JNI)
          -> GpuInferenceSession
              -> GgufModelLoader / GgufReader
              -> tokenizer + architecture + quantization registries
              -> model execution plan and lowerings
              -> weight streamer + memory manager + KV cache
              -> GpuEngine
                  -> DawnBackend
                  -> UniversalGlesBackend
```

The test application is deliberately outside the native engine. It only imports a model, requests activation, reads diagnostics and copies the report. It contains no model execution policy of its own.

## 2. Native engine core

The backend-neutral core is in `engine/src/main/cpp/gpu_engine`.

- `gpu_engine.*`: backend registration, device selection, lifecycle and diagnostics.
- `execution_engine.*`: buffers, tensors, memory leases, operator/kernel registry, graph and scheduler.
- `gguf_reader.*`: independent GGUF parsing and bounded tensor reads.
- `model_registry.*`: architecture/quantization registration and fail-closed model preflight.
- `tokenizer.*` and `tokenizer_adapters.cpp`: GGUF tokenizer metadata and profile adapters.
- `quantization.*`: quantization descriptors, decoders, reference implementations and GPU candidates.
- `model_graph.*`, `model_lowerings.*`, `operators.*`: model graph and operator lowering.
- `weight_streamer.*`, `memory_planner.*`, `kv_cache.*`: bounded model storage and execution memory.
- `sampler.*`: sampling and stop-sequence primitives.
- `inference_session.*`: model activation, preflight, graph preparation and generation orchestration.

The core must not include or link llama.cpp/GGML execution APIs. GGUF is parsed directly by the engine.

## 3. Android backends

### Dawn/WebGPU

`dawn_backend.cpp` implements the generic GPU backend API using Dawn. The build obtains the Dawn source revision from the metadata in the pinned AndroidX WebGPU AAR, then builds the native Dawn implementation. The adapter request uses the OpenGL ES compatibility route; Vulkan remains disabled.

### Universal OpenGL ES

`universal_gles_backend.cpp` implements the same backend contract using GLES compute. `universal_gpu_adapter.cpp` owns the EGL context thread, and `universal_adapter_bridge.cpp` routes backend work onto that context thread.

Dawn and GLES capability reports are distinct. GLES extension support must not be inferred from Dawn/WebGPU features, or vice versa.

## 4. Model activation policy

Model activation is fail-closed:

1. Initialize a backend and pass its baseline compute validation.
2. Inspect the GGUF header, metadata and tensor directory.
3. Validate tokenizer profile, architecture metadata and tensor storage.
4. Build and validate a model execution plan.
5. Validate quantization and operator/kernel coverage for the selected backend.
6. Estimate memory using the explicit application-provided budget.
7. Stream required weights and prepare model state.

Any unsupported or unverified requirement rejects activation with a diagnostic blocker. A successful GGUF parse or backend self-test alone is not sufficient.

## 5. Android boundary

The Android library exposes a small Kotlin API over JNI. Native implementation is built as `libai_chat_runtime.so` for `arm64-v8a`. The test app uses Android's Storage Access Framework to select a GGUF file and copies it into app-private storage before passing a filesystem path to native code.

There is no CPU inference runtime or silent fallback in this branch. Vulkan and OpenCL are not built.

## 6. Validation boundaries

- Host C++ contract tests validate deterministic engine logic and reference math.
- Android CI validates CMake configuration, Dawn integration, NDK compilation, JNI linkage and APK packaging.
- Device tests validate actual driver behavior, backend initialization, shader compilation, dispatch/readback, memory behavior and model activation.
- Model-level tests must additionally validate tokenizer parity, numerical output, generation quality and stability.

Do not promote an unverified profile or kernel based only on a successful compile.
