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

- The graph scheduler now supports internal copy-only nodes in the same ordered command batch as compute dispatches. Dense model plans insert a KVCacheAppend side-effect node after K/V projection and RoPE, then make Attention depend on that node through explicit control dependencies. Planned K/V cache tensor views are bound for the requested past+current sequence length; valid-token accounting is committed only after the full graph submits and synchronizes successfully. Host contract coverage includes the copy-node path and cache view binding. End-to-end real-GPU Attention parity remains UNVERIFIED.

- Added a model activation allocator that resolves dynamic token dimensions, creates F32 device Tensor bindings through MemoryManager, reuses buffers only when producer/last-consumer intervals do not overlap, and pins declared retained outputs (the logits tensor). Dense residual hidden-state names are now versioned per layer to avoid accidental input/output aliasing. Host contract tests cover safe non-overlapping slot reuse; full model memory sizing and real-device pressure remain UNVERIFIED.

- Added an opt-in Android JNI diagnostic probe (`NativeLlamaCpp.gpuEngineDiagnostics()`) that initializes the new Dawn-first / Universal GLES backend composition, reports the selected device/backend and discovered limits/features, then shuts the engine down. It does not load a model, alter the existing llama.cpp runtime state, or claim inference readiness. The probe has not yet been executed on physical Android hardware.

- Added a plan-driven GGUF weight binder for single-storage-binding F32 and Q8_0 weights. F32 weights become typed device Tensor bindings; Q8_0 weights remain encoded in a typed quantized-weight map and are retained by weight leases. Dense plan construction selects `MatMul_Q8_0` for Q8_0 projection weights, with F32 activation and flattened 3D output support, and registers both Dawn and Universal GLES candidates. Multi-chunk execution, Q8_0 kernel numerical validation, Q8_0 Embedding, and other quant formats remain UNVERIFIED/unsupported by this model binder.

- Added a runtime kernel qualification suite for F32 Multiply, MatMul, RMSNorm, SiLU, Softmax, RoPE, Attention and Embedding, alongside the existing Add validators. It executes backend candidates, reads output back and compares with CPU reference implementations before promoting each kernel. Fake-backend tests ensure candidates cannot be promoted without real numerical execution. The Android diagnostics probe now runs this suite and the separate Q8_0 gate. Actual physical-device validation results are not yet available.

- **2026-11-22 / backend qualification and source authorization hardening:** Android backend selection now tries Dawn and then Universal GLES only after all baseline F32 operator candidates pass actual compile/dispatch/readback parity. Dispatch authorization compares the submitted shader with the registered validated source while allowing only named shape/parameter constant specialization; altered shader bodies are rejected. Q8_0 validation state is tracked per backend, not inferred from the summary state. This is host/source-verified; physical Android GPU results remain **UNVERIFIED**.
- **2026-11-22 / Universal GLES lifecycle and dynamic invocation fixes:** added a bounded 64-entry compute-program cache keyed by kernel identity plus specialized source, byte-granular SSBO allocation/copy support for packed weights, consistent device-loss propagation, and safe activation/KV-view rebinding between prefill and decode token counts. Real-device cache behavior, buffer pressure, and repeated decode correctness remain **UNVERIFIED**.

- **2026-11-22 / Q4_0 direct quantized MatMul path:** added Dawn WGSL and Universal GLES GLSL kernels that read packed Q4_0 blocks directly (FP16 scale plus low/high nibbles), independent CPU-reference MatMul, backend-specific numerical validation, operator registrations, GGUF weight binding, and dense-plan lowering. Q4_0 is promoted only after backend readback parity; no physical-device result is claimed until the Android probe runs on hardware.

- **2026-11-22 / Q4_1 direct quantized MatMul path:** added Dawn and Universal GLES packed Q4_1 kernels (FP16 scale/minimum plus packed nibbles), CPU-reference MatMul, per-backend numerical validation, Android diagnostics reporting, GGUF weight binding, and dense-plan lowering. The implementation remains **UNVERIFIED** on physical GPU hardware until backend readback parity is observed on real devices.

- **2026-11-22 / Quantized embedding path:** Q4_0 and Q4_1 now have direct packed-weight Embedding kernels for Dawn and Universal GLES, CPU decoder references, backend-specific numerical qualification, and architecture-plan/lowering routing. This closes the quantized token-embedding blocker for these two formats at the operator level; it does not resolve multi-chunk weight residency or full-model execution readiness.

- **2026-11-22 / layer-serial weight streaming execution:** added a streamed graph execution path that loads only the current operation's GGUF weight tensors, compiles and submits that operation, then releases its weight bindings before advancing. Quantized weight bindings are now recognized by graph compilation. This avoids requiring every model weight to remain resident at once, provided each individual tensor fits one backend buffer. Multi-chunk execution of a single oversized tensor remains **UNVERIFIED / NOT IMPLEMENTED**.

- **2026-11-22 / Android inference-session integration:** added `GpuInferenceSession` and JNI/Kotlin routing for full-GPU requests. The session initializes the selected Android backend through the shared Android backend composition, uses the independent GGUF reader, architecture/quantization registries, per-operation weight streaming, KV cache, graph/operator lowering and sampler. The app falls back to llama.cpp with GPU layers set to zero if GPU model loading fails; if GPU generation fails, partial GPU text is discarded, the model is loaded on llama.cpp CPU, and the same prompt is retried. Explicit partial-layer requests currently fall back to CPU rather than silently pretending to support partial placement.
- **Current inference support boundary:** the session is fail-closed for weights other than F32, Q4_0, Q4_1 and Q8_0; Q8_0 token embeddings now have direct packed-weight F32-output kernels for Dawn and Universal GLES, with independent decoder-reference validation; support is promoted only after backend readback parity. Activations and KV cache currently use F32. A single weight tensor must fit the backend storage-buffer limit; splitting one tensor across multiple GPU dispatches is not implemented. Dense Transformer adapters are the only intended initial model path; MoE, recurrent, hybrid and multimodal execution are not yet implemented.
- **Memory and output improvements:** weights are loaded per operation and released after dispatch; prefill retains only the final-token logits row rather than allocating logits for every prompt token. The memory preflight estimates KV cache, activations and the peak padded set of weight buffers simultaneously loaded for one operation against a conservative app-provided shared-memory budget.
- **Validation status:** backend kernels are numerically validated at runtime on the selected device where their validators exist. Full model output parity against llama.cpp, long-run stability across physical GPU families, and universal model/quantization support remain **UNVERIFIED**. Diagnostics report these states explicitly.

- The latest reliability pass fixes Q4_1 MatMul candidate registration for both GPU backends, qualifies baseline F32 operators against additional irregular/non-workgroup-aligned shapes, and makes the memory preflight account for the padded peak weight set loaded by one streamed operation. Model-specific quantization or operator-coverage failures now trigger a validated retry on the alternate GPU backend before CPU fallback. Diagnostics expose GPU vendor/device/driver/API, selected backend, limits, feature count/list, peak per-operation weight residency and fallback reason. These changes do not remove the remaining **UNVERIFIED** tokenizer, physical-device, full-model parity, and non-Dense architecture limitations.

- The GGUF reader now retains a mutex-protected file stream for repeated streamed-weight reads and indexes tensor names after parsing, reducing per-operation file-open and linear tensor-lookup overhead. Reader state is released with the model session; repeated range reads are covered by contract tests.

- **Current hard blocker:** all built-in SentencePiece and byte-level BPE tokenizer adapters remain marked **UNVERIFIED**. The session therefore fails closed at tokenizer preflight for ordinary GGUF models; a successful backend/kernel self-test does not yet mean the app can generate text on GPU. CPU fallback remains the expected safe path until exact tokenizer parity suites pass for specific metadata profiles.

- **2026-11-22 / GLES quantized dequantization candidates:** added backend-specific GLSL ES 3.1 candidates for Q4_0, Q4_1, Q5_0, Q5_1 and Q8_0 dequantization, alongside the existing Dawn WGSL candidates. Each candidate validates complete block sizes, padded input-buffer range and F32 output shape before dispatch construction. Added a shared runtime validation path that compares GPU readback against the independent host block decoders and only returns a dispatch marked Validated after numerical parity. These new GLES candidates are still **UNVERIFIED** until that validator passes on physical GLES hardware; they are not yet wired into the model's supported-weight set or the inference-session quantization registry.

- **2026-11-22 / SentencePiece Unigram algorithm correction:** replaced the previous greedy adjacent-pair merge approximation with score-maximizing Viterbi segmentation over normalized UTF-8 offsets, added explicit byte-token fallback and rejected non-finite score tables. A synthetic regression case demonstrates a globally higher-scoring segmentation than greedy pair merging. The adapter remains **UNVERIFIED** because full SentencePiece normalizer/spec parity and comparisons against a trusted tokenizer implementation have not yet passed.
