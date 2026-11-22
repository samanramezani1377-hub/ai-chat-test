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

## Latest physical-device finding (Xiaomi 2303CRA44A, Android 14)

The first captured activation attempt imported a 1.49 GiB Qwen3.8-2B Q6_K GGUF successfully, but no backend completed baseline validation:

- Dawn/WebGPU: baseline validation became unavailable at Embedding because the device was no longer ready. The current callback records only a boolean health state, so the underlying Dawn device-loss/uncaptured-error message is not yet preserved.
- Universal GLES: SSMConv1D shader compilation failed because the GLSL identifier `sample` is reserved by the mobile GLSL ES compiler.
- Vulkan is disabled as required; the app reports no CPU fallback.

The SSMConv1D shader now uses `sample_value` in both embedded WGSL and GLSL sources, and the host contract script guards against reintroducing the reserved identifier. The test app now exposes the existing native generation/cancellation JNI entry points so end-to-end inference can be exercised once a model passes all validation gates.

Model activation now honors the model preflight report and fails closed while architecture, tokenizer, quantization, graph or model-kernel validation remains incomplete. The Qwen3.5 structural recipe reads key-head width, value-head width and rotary width independently from GGUF metadata instead of assuming they all equal embedding length divided by query-head count. Shape mismatch diagnostics include both expected and actual tensor dimensions. In particular, the Qwen3.5 BPE adapter is marked validated only because CI compares native encode/decode results with the pinned upstream Hugging Face tokenizer on a multilingual and Unicode edge-case corpus. Vocabulary, merges, token types and special-token IDs are still read from each GGUF file; model-specific token IDs are not hard-coded.

The app's current memory budget is a conservative estimate derived from Android system-memory information, not a measurement of dedicated GPU VRAM. Diagnostics should be interpreted accordingly.

## Known limitations

1. The `gpt2/qwen35` BPE profile is validated by a pinned Hugging Face Qwen3.5 reference-parity suite in CI. Other built-in tokenizer profiles remain unverified until equivalent independent reference coverage exists.
2. Model architecture and graph coverage is incomplete. Unsupported or unverified combinations must continue to fail closed. Structural tensor checks must use architecture metadata for independent head and rotary dimensions; tensor shape diagnostics report both expected and observed dimensions.
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


## Latest device-log follow-up (2026-11-23)

The next Xiaomi 2303CRA44A run exposed two preflight/plan defects before GPU execution:

- Qwen3.8-2B and Qwen3.5-0.8B both stopped at Embedding_Q6_K. The Q6_K packed kernel candidate and lowering existed, but the inference-session compatibility gate accepted Q6_K only for MatMul and rejected the same quantization on Embedding. The dense adapter also lacked the Q6_K embedding operator selection. Both routing/compatibility gaps are now corrected; backend-specific numerical validation remains mandatory.
- Qwen3-0.6B reported projection and QK-Norm shape mismatches because the dense adapter assumed head_dim = hidden_size / attention_heads. Qwen3-0.6B uses hidden size 1024, 16 query heads, 8 KV heads and head dimension 128; its Q projection width is 2048. The adapter now obtains key/value head widths from GGUF metadata when present and otherwise derives them from the first projection tensor shapes, then validates every layer against those dimensions. The official model config also declares tied word embeddings for this exact 0.6B configuration; a narrowly matched config profile handles GGUF exports that omit the tie flag and output tensor. Source: https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json.
- A regression contract now covers a dense Qwen3 graph with independent head width and Q6_K embedding operator selection.

These code changes are not evidence of successful GPU inference. The next physical-device run must show a selected Dawn or GLES backend, successful backend numerical validation, a structurally valid and compiled model graph, and generated tokens. If Q6_K embedding fails validation, the diagnostic must preserve the specific backend/kernel validation error rather than reporting a generic missing operator.
