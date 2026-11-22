# Android GPU Inference Engine — Current State

Last reviewed: 2026-11-22  
Working branch: `engine`

## Project boundary

This branch is now an engine-only Android project. The previous AI Chat application and its unrelated features have been removed from this branch.

The retained project contains:
- The independent C++ GPU engine and Android JNI boundary.
- GGUF parsing, model metadata/preflight, tokenizer adapters, quantization references, graph/lowering, memory planning, streaming, sampling and inference-session code.
- Dawn/WebGPU and Universal OpenGL ES backends.
- Host-side C++ contract tests.
- A minimal Android test application that imports a GGUF file, attempts GPU activation and displays/copies the complete diagnostic report.
- Engine-specific documentation and CI.

The branch no longer contains the previous chat UI, voice/TTS/ASR, widgets, agent/action systems, cloud providers, llama.cpp, GGML, Rust tokenizer, ONNX Runtime, sherpa-onnx or CPU inference runtime.

## Runtime topology

```text
Android GPU Engine Test App
  -> GpuEngineTestApi
      -> NativeGpuEngine JNI
          -> GpuInferenceSession
              -> GGUF reader and model preflight
              -> tokenizer / architecture / quantization registries
              -> graph, lowering, memory and weight streaming
              -> Android backend composition
                  -> Dawn WebGPU (OpenGL ES compatibility adapter)
                  -> Universal OpenGL ES backend
```

There is no CPU inference fallback in this branch. A failed GPU activation is reported as a failure; the test application does not silently route the model to another runtime.

## Build and dependency boundary

- Native language: C++17 engine; Dawn is configured with the C++ standard required by its build.
- Android ABI: `arm64-v8a` only.
- Android minimum SDK: 29.
- GPU backends: Dawn/WebGPU and Universal GLES.
- Vulkan: disabled.
- OpenCL: not part of this engine.
- Native model format: GGUF.
- External native GPU dependency: Dawn, pinned from the source revision recorded in the AndroidX WebGPU AAR metadata.
- No llama.cpp/GGML execution library or source tree is fetched or linked.

## Implemented foundations

- Independent GGUF v1-v3 reader with bounded tensor range reads.
- Metadata-driven tokenizer and architecture registries with fail-closed preflight.
- Host-side quantization decoders and numerical reference paths for the formats represented in the implementation.
- Backend-neutral buffers, tensors, command batches, memory leases, graph scheduling and dispatch authorization.
- Dawn and GLES backend implementations with device discovery, limits, self-tests, buffer operations and compute dispatch.
- GGUF weight streaming, KV-cache primitives, model lowerings and an inference session.
- C++ contract tests that compile without Android or a GPU.

These foundations do not imply that every listed model architecture, tokenizer, quantization or kernel is production-ready.

## Known limitations

1. Built-in tokenizer profiles remain unverified until exact profile parity is demonstrated against authoritative reference implementations and fixtures.
2. Model architecture and graph coverage is incomplete. Unsupported or unverified combinations must continue to fail closed.
3. Kernel candidates become usable only after backend-specific numerical validation. Source presence or compilation is not validation.
4. Android CI proves build/link and host contract behavior; it does not prove physical GPU compatibility or model output correctness.
5. No universal support claim is justified for all Android GPUs, models or quantizations.
6. Full end-to-end model generation on physical Android GPUs remains to be demonstrated and recorded per device/backend/model combination.

## Current test strategy

- `scripts/test_gpu_engine_contracts.sh`: compile and run backend-neutral C++ contract tests on the host.
- `scripts/verify_gpu_engine_independence.sh`: guard the engine against llama.cpp/GGML build dependencies and enforce the Android/Vulkan policy.
- GitHub Actions: run both host checks and build the minimal arm64 Android test APK.
- Physical-device runs: import a known GGUF file in the test app, capture the full diagnostic log, and record device/GPU/driver/model details.

A passing CI run must never be described as proof of real-device inference.
