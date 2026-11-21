# GPU Inference Engine — Current-State Audit

Status: **AUDITED FROM SOURCE — 2026-11-21**  
Target branch: `engine`  
Audited HEAD: `07890ec925240671c29a1a485b52551101463db2`

## Scope and evidence

This audit is based on the branch tree, native CMake configuration, Android JNI runtime, Kotlin runtime adapter, GGUF inspector, Dawn build integration, the Universal GPU Adapter, and the existing CI workflow. Hardware-specific behavior is marked **UNVERIFIED** unless an actual device run is available.

## Existing execution topology

```text
Android UI / AI Core
  -> LlamaCppAndroidRuntimeAdapter
      -> NativeLlamaCpp JNI
          -> nativeLoad / nativeGenerate
              -> llama.cpp model + context + sampler
                  -> GGML scheduler / graph / tensor / backend
                      -> GGML WebGPU backend -> Dawn -> OpenGL ES
                      -> GGML CPU backend
```

- **CPU path:** llama.cpp owns model loading, context, tokenization, sampling and generation. The generic ARM64 CPU configuration uses an ARMv8-A baseline, OpenMP and KleidiAI runtime dispatch.
- **GPU path:** GPU model loading and inference are currently performed by llama.cpp/GGML. CMake builds GGML WebGPU and OpenCL; Vulkan is explicitly disabled. The WebGPU patch forces Dawn's OpenGL ES compatibility adapter.
- **Fallback:** GPU load/residency/context failures release the partial runtime and retry with llama.cpp CPU parameters. This is an existing whole-model fallback path; its behavior must remain covered by regression tests.
- **Universal GPU Adapter:** `universal_gpu_adapter.cpp/.h` creates a process-lifetime EGL/GLES context on a dedicated owner thread, queries GLES limits/extensions, compiles and dispatches a small integer compute shader, and validates SSBO readback. It is currently a capability probe/self-test, **not** a general command queue, allocator, tensor runtime, or model inference backend.
- **GGUF:** the Kotlin `GgufInspector` reads header/metadata for import validation. Native llama.cpp currently owns the actual GGUF tensor/model loading used for inference.

## Critical coupling and gaps

1. `native_runtime.cpp` directly includes llama and GGML internal headers, including backend implementation, OpenCL and WebGPU headers.
2. Native CMake fetches llama.cpp and links `llama`, `llama-common`, `mtmd`, `ggml-opencl` and `ggml-webgpu` into the current runtime library.
3. The current GPU model load calls `llama_model_load_from_file`; context creation calls `llama_init_from_model`; generation is through llama.cpp. Therefore the current GPU path is not independent.
4. GPU residency is measured by inspecting GGML tensor buffers. The existing 1% minimum GPU weight residency threshold is a placement heuristic, not proof that useful inference work executes on GPU.
5. The WebGPU implementation is an upstream GGML backend modified by CMake source patches. Those patches are coupled to a pinned llama.cpp source layout and exact source anchors.
6. The Universal GPU Adapter's GLES self-test and Dawn's WebGPU feature report are separate facts. GLES extensions or compute success must never be treated as proof of a WebGPU feature or an inference kernel.
7. The existing GGUF inspector is metadata-oriented; it is not yet a complete independent tensor-directory reader, quantization decoder, or model loader.
8. The current CI contains extensive source-contract assertions and Android build/unit-test jobs. CI does not prove real-device GPU inference, cross-vendor portability, numerical parity, or long-run stability.

## Non-negotiable preservation boundary

- Do not modify the CPU execution implementation or its llama.cpp settings as part of GPU engine work.
- Do not remove the existing CPU fallback.
- Do not enable Vulkan.
- Keep the old GPU route available behind a controlled migration switch until the replacement has real-device parity evidence.
- Do not claim a backend, operator, quantization, architecture, or GPU is supported based only on discovery or compilation.

## Audit conclusion

The repository has a useful Android application/runtime boundary, an existing CPU fallback, a Dawn build path, and a reusable GLES capability/self-test component. It does **not** yet contain an independent GPU inference engine. The first safe implementation step is to introduce independent engine contracts and capability reporting without changing the active inference route. Actual Dawn execution, tensor/memory management, operators, quantization, model loading, and generation must then be implemented and validated incrementally.


## Incremental GPU engine implementation after the audit snapshot

The audit above is a historical snapshot at `07890ec925240671c29a1a485b52551101463db2`. Subsequent work on `engine` has added:

- Independent F32 WGSL candidates and CPU references for Add, MatMul, RMSNorm, SiLU and Softmax. These candidates remain **UNVERIFIED** pending real Dawn dispatch/readback parity tests.
- An independent Q8_0 host decoder with validated 32-value / 34-byte block parsing, plus a GPU dequantization WGSL candidate that remains **UNVERIFIED**.
- A Llama dense metadata adapter that validates required dimension metadata but remains **UNVERIFIED** as a complete model architecture implementation.
- A public engine API to query the existing Universal GPU Adapter. It reports the existing EGL/GLES compute self-test and limits as capability diagnostics only; it is not yet an executable engine backend.
- Independent sampler/stop handling and GGUF tokenizer metadata parsing. No complete tokenizer implementation or generation loop is connected yet.

The active Android inference route is still llama.cpp/GGML for both current GPU and CPU modes. No migration or route switch has been performed. The current branch still does not meet the goal of replacing llama.cpp GPU inference.

- An independent per-layer KV cache primitive now estimates and allocates key/value buffers under the GPU memory budget, enforces sequential appends and tracks logical valid-token lengths. It is not yet wired into an Attention operator or model generation.
- A GGUF model inspection loader now orchestrates independent parsing and fail-closed architecture/quantization preflight; it does not yet load an executable graph or claim GPU readiness.
- Operator candidates now also include F32 Embedding and RoPE, plus direct Q8_0 quantized-weight MatMul for F32 activations. These GPU paths remain unverified and are not selected for inference.


## Latest Universal GLES backend increment — 2026-11-21

The Universal GPU Adapter now exposes a serialized task queue on the same dedicated thread that owns its EGL/GLES context. Synchronous tasks return structured callback errors; asynchronous tasks support safe resource cleanup without recursively blocking the context owner thread.

A separate `ai_chat_gpu_backend_gles` target now implements the generic Backend contract over that adapter-owned context:
- device discovery and GLES capability mapping;
- device buffer allocation and deferred destruction;
- staged buffer upload/readback and buffer copies;
- GLSL ES compute shader compile/link/dispatch;
- SSBO binding/range validation, workgroup-limit checks, memory barriers and synchronization;
- a backend-level integer compute/readback self-test.

The backend rejects dispatches unless their kernel identity is marked `Validated` and a GLSL ES source is supplied. The engine core remains independent of EGL/GLES; the adapter bridge and GLES backend are separate targets. This does **not** establish model inference support: no production operator is validated for GLES, the backend is not connected to the Android model router, and hardware compatibility remains **UNVERIFIED** until real-device tests run.


## Additional independent runtime components — 2026-11-21

- Android backend composition now registers Dawn first and the Universal GLES backend second. `initialize_android_gpu_engine()` uses ordered backend initialization and records whether a backend fallback occurred. This is an engine API and composition root; the Android app's model-generation path has not yet switched to it.
- Generic GPU dispatch now validates per-axis workgroup sizes, total local invocations, workgroup counts, SSBO binding indices, ranges and alignments against discovered limits.
- GGUF tokenizer metadata now preserves space-prefix, whitespace normalization and byte-fallback flags.
- Independent SentencePiece-score-merging and byte-level BPE adapters are registered for selected profiles (`llama/default`, `llama/llama`, `spm/default`, and `gpt2` with `gpt2`, `llama3`, `qwen2`, `qwen3` pre-tokenizers). Their tokenization/pre-tokenization parity is **UNVERIFIED**; model preflight reports this explicitly and the public TokenizerEngine refuses to execute unvalidated adapters.
- An initial direct Q8_0 F32-activation MatMul candidate is available for WGSL and Universal GLES. It remains **UNVERIFIED** and is not eligible for production dispatch.


## Model graph lowering and execution orchestration — 2026-11-22

- Added backend-scoped lowering registries for F32 Add, Multiply, MatMul, RMSNorm, SiLU, Softmax and RoPE. Lowerings resolve named activation/weight/output tensors from an explicit execution context and create backend-specific dispatch candidates.
- The Android composition API now registers Dawn and Universal GLES lowering sets into a caller-owned registry.
- Added a graph execution entry point that compiles the full graph, authorizes every kernel against the selected backend's validated operator registry, schedules the graph and marks a run as executed only after backend synchronization succeeds.
- A successful graph run does not mark a model GPU-ready. Architecture/tokenizer/quantization and end-to-end model parity qualification remain separate gates.
- Embedding is not registered as a model lowering until backend-owned token-ID upload/lifetime is integrated. Attention is not registered until KV-cache append/read semantics are connected. The present F32 MatMul builder accepts only its currently implemented matrix layout; unsupported rank/layout combinations fail closed.
- The model lowering layer and full app inference integration remain UNVERIFIED; no model generation route has been switched away from llama.cpp.

- KV cache now supports typed, contiguous device-to-device K/V appends through the backend command-copy path, advances valid-token state only after successful submit/synchronization, and exposes valid-prefix K/V tensor views. The primitive is host-contract-tested; it is not yet sequenced between projection/RoPE and Attention nodes in an end-to-end model graph.

- Added an Embedding model lowering that uploads the current token-ID vector through the selected backend and retains the upload buffer through the dispatch binding. Model graph compilation now recognizes `token_ids` as a typed external host input rather than requiring a fake Tensor binding. The upload/lowering is contract-tested but remains UNVERIFIED for numerical GPU execution.
