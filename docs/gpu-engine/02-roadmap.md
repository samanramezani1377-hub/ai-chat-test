# GPU Engine Implementation Roadmap

All unvalidated items are **UNVERIFIED**. This roadmap is cumulative; do not skip correctness gates.

| Stage | Deliverable | Exit evidence |
|---|---|---|
| 0 | Current-state audit | branch/HEAD, dependency map, CPU/GPU topology, CI and risk report |
| 1 | Architecture contracts | backend/tensor/memory/operator/kernel/quant/model/scheduler/runtime contracts; dependency rules |
| 2 | Backend foundation | Dawn device discovery, capabilities, buffers, command submission, synchronization, device-loss recovery |
| 3 | Tensor and reference operators | CPU reference math independent of llama/GGML; GPU kernels; shape/dtype numerical tests |
| 4 | Memory and scheduling | allocator/pool/arena, graph lifetimes, staging, cancellation, KV/recurrent cache primitives |
| 5 | Quantization | independent GGUF type map and per-format decoders/kernels; reference parity |
| 6 | First model | one selected dense architecture; GGUF load, prefill, decode, KV cache, deterministic comparisons |
| 7 | Architecture expansion | GQA/MHA, MoE, hybrid/recurrent, then multimodal adapters with independent tests |
| 8 | General optimization | fusion, tiled GEMM, quantized GEMM, attention, adaptive workgroups, memory reuse |
| 9 | Compatibility and soak | device matrix, long runs, memory pressure, device loss, numerical and performance reports |
| 10 | GPU migration | route validated combinations to new engine; retain llama.cpp CPU fallback; remove legacy GPU only after parity |

## Execution safety gates

Graph nodes now require a kernel identity that is present in the active Operator Registry and marked Validated. Dispatch authorization binds the selected backend; the scheduler rejects backend mismatches, and Dawn independently rejects unvalidated dispatches. Candidate builders remain Unverified until their kernels have passed the required numerical and real-device validation.

## Additional required inference components

The first architecture pass exposed two runtime components that were implicit in the original outline but are mandatory for true independence: a standalone tokenizer engine and a standalone sampler/generation controller. They must not call llama.cpp on the GPU route. GGUF tokenizer arrays, normalization/pre-tokenization rules, special token IDs, BPE/Unigram merge behavior, and exact parity tests must be implemented before a first model can be considered end-to-end supported.

## Universal GPU Adapter integration status

The Universal GPU Adapter now exposes serialized synchronous and asynchronous tasks on its dedicated EGL/GLES owner thread. The separate `ai_chat_gpu_backend_gles` target uses that API for buffers, copies, uploads, readback, GLSL ES compute dispatch and synchronization. Backend initialization runs its own integer compute/readback self-test. Production operator kernels, model execution and real-device qualification remain **UNVERIFIED**.

## Immediate next increments

1. Keep the active inference route unchanged.
2. Add a standalone `ai_chat_gpu_engine` C++ library with no llama.cpp/GGML include or link dependency.
3. Define backend/device/capability/status contracts and a registry.
4. Bridge Universal GPU Adapter's verified GLES probe results into generic diagnostics without treating them as inference readiness.
5. Add source-level architecture guards ensuring the standalone engine target does not include GGML/llama headers.
6. Implement Dawn as a separate backend target linked directly to the pinned Dawn build, after verifying the available Dawn C API and lifecycle constraints.
7. Implement a real command queue/buffer API and test it before adding tensor kernels.

## Implemented increments

- **2026-11-21 / contract + bridge:** backend-neutral device/buffer/command/tensor contracts, registry, and a diagnostics bridge from Universal GPU Adapter. The bridge deliberately reports `inference_ready=0`.
- **2026-11-21 / Dawn backend foundation:** a separate `ai_chat_gpu_backend_dawn` target requests the OpenGL ES Compatibility adapter, reads Dawn limits/features, creates buffers, uploads/downloads, records buffer copies and WGSL compute dispatches, and runs a small integer compute/readback self-test. It is not yet selected by the Android runtime and does not yet implement model operators, quantization, GGUF tensor loading, or inference. **UNVERIFIED:** Android NDK build and real-device Dawn self-test until run on hardware/CI.
- **2026-11-21 / tensor, memory and graph foundations:** checked tensor shape/stride/byte-range calculations, budgeted RAII memory leases, fail-closed operator-kernel selection, dependency-aware graph ordering, cancellation-aware command recording, one-submit/one-sync scheduler behavior, and host-side contract tests. These are foundational contracts, not model-level feature support.
- **2026-11-21 / GGUF model inspection loader:** added an orchestrator that opens the independent GGUF reader and runs architecture/quantization preflight, while exposing blockers without falsely treating a parsed file as GPU-ready. Weight streaming remains a separate bounded operation.\n- **2026-11-21 / independent GGUF reader:** parses GGUF v1-v3 header, scalar metadata and selected tokenizer arrays (tokens, merges, scores and token types), tensor directory, alignment and offsets; calculates storage byte ranges for the listed known tensor types and supports bounded chunk reads. Quantized type layouts are metadata-only here: their GPU decoders and kernels remain **UNVERIFIED**. Unknown types are preserved as unknown and cannot be streamed as sized tensors.
- **2026-11-21 / model registries:** added separate quantization-format and architecture-adapter registries plus a fail-closed model preflight report. A Llama dense metadata adapter exists but remains **UNVERIFIED**. The Q8_0 host decoder is marked validated after block-level tests; no GPU quantized MatMul kernel is validated. Preflight intentionally never marks a model inference-ready before tokenizer, graph/operator/kernel and model-level validation.
- **2026-11-21 / F32 operator candidates:** added generic Add, Multiply, MatMul, RMSNorm, SiLU, Softmax, RoPE, Embedding and causal Attention WGSL candidates, shape/buffer validation and CPU reference implementations. All GPU kernels remain `UNVERIFIED`; registry selection rejects them until real Dawn/WebGPU numerical parity tests promote them.
- **2026-11-21 / engine lifecycle API:** added explicit backend registration, device enumeration, backend initialization, graph execution, shutdown and runtime diagnostics. This API is not yet connected to the app router; it reports `inference_ready=false` by design.
- **2026-11-21 / GGUF weight streaming foundation:** reads validated tensor ranges in bounded, block-aligned chunks and uploads them through the backend-managed memory budget. Current behavior still retains all chunks for the loaded tensor; true layer-wise eviction/offload and KV-cache allocation are pending.
- **2026-11-21 / execution safety gate:** graph construction now requires registry-authorized validated kernels, the scheduler enforces active-backend matching, and Dawn rejects dispatches without a validated identity. Current operator candidates remain unverified and cannot enter a production graph.\n- **2026-11-21 / independent sampler:** added temperature, top-k, top-p, repetition penalty, seeded sampling, EOS and stop-sequence handling with deterministic host-side tests. This is not yet integrated with a model/tokenizer generation loop.
- **2026-11-21 / GPU memory planner:** added overflow-checked preflight accounting for resident or streamed weights, KV cache, scratch memory and an explicit safety reserve. It requires an application-provided memory budget and never guesses physical GPU memory.\n- **2026-11-21 / runtime selection policy:** added a pure fail-closed selector that chooses the dedicated GPU engine only when device, tokenizer, architecture, quantization, operator coverage, model graph and memory checks all pass; otherwise it chooses the existing CPU llama.cpp path when available. This policy is not yet connected to the Android app router.\n- **2026-11-21 / KV cache primitive:** added checked memory estimation, per-layer K/V buffer allocation under the engine memory budget, sequential append validation and logical layer reset. Initial validated storage type is F32; F16/BF16 remain constrained by alignment and backend validation. This is a storage primitive, not yet connected to Attention or model generation.\n- **2026-11-21 / Q8_0 decoder:** added a block-accurate 32-value / 34-byte host decoder and FP16 scale conversion with finite-value and size validation. Added generic WGSL dequantization candidates that read packed Q4_0, Q4_1, Q5_0, Q5_1 and Q8_0 blocks directly from storage buffers, plus an explicit Q8_0 quantized-weight MatMul candidate for F32 activations and a CPU reference. All GPU quantization candidates remain **UNVERIFIED**; no quantized GPU path is enabled for inference. Host decoder tests are present; the GPU candidate and quantized MatMul remain **UNVERIFIED**.

## Validation policy

CI/source checks are useful but not hardware evidence. Do not wait for CI to proceed with independent implementation work, but never describe a CI pass as real-device compatibility. Real-device qualification requires the exact GPU, OS, driver, Dawn version, features/limits, test suite and test result to be recorded.

- **2026-11-21 / Universal GLES backend:** added an owner-thread task queue to the existing Universal GPU Adapter and a separate GLES backend implementing device discovery, buffer allocation/deferred destruction, copy, upload/readback, GLSL ES dispatch, SSBO/workgroup validation and synchronization. Backend initialization includes a second compute/readback self-test. No production GLES operator kernel is validated and no model inference route uses this backend yet.
- **2026-11-21 / backend isolation:** moved Universal adapter probing out of the backend-neutral `GpuEngine` class; the adapter bridge now belongs to the GLES backend target. The host contract test script checks that both Dawn and GLES backends remain free of llama.cpp/GGML execution includes. Hardware behavior remains **UNVERIFIED**.

- **2026-11-21 / tokenizer adapter foundation:** added exact-profile registry, SentencePiece score-merging and byte-level BPE adapter implementations, special-token handling, GGUF normalization/byte-fallback metadata, and host contract tests. All built-in adapters remain **UNVERIFIED** because Unicode normalization and profile-specific pre-tokenizer parity have not yet been proven against reference tokenizers. Model preflight now reports tokenizer adapter status rather than a generic pending flag; `TokenizerEngine` fails closed.
- **2026-11-21 / Android backend composition:** added a Dawn-first / Universal GLES fallback initializer and fallback diagnostics. The composition root is linked into the Android runtime, but model generation has not yet been routed through the new engine.
- **2026-11-21 / dispatch safety:** Dawn and GLES dispatch validation now considers local workgroup dimensions, invocation counts, workgroup counts and storage binding limits. The Q8_0 direct MatMul candidate is available on both shader backends but remains **UNVERIFIED**.


- **2026-11-22 / model lowerings:** introduced backend-scoped model-plan lowering callbacks for F32 Add, Multiply, MatMul, RMSNorm, SiLU, Softmax and RoPE. Registered both Dawn and Universal GLES lowering sets through the Android composition root. Embedding and Attention remain intentionally unsupported at the lowering layer until token upload and KV-cache contracts are integrated.
- **2026-11-22 / graph execution orchestration:** added a fail-closed plan-to-graph-to-scheduler entry point. Every dispatch still requires a backend-specific validated kernel; execution success is recorded separately and does not imply model-family readiness.

- **2026-11-22 / device KV cache path:** added typed tensor validation, device-to-device K/V cache append via command-buffer copies (no host readback), post-synchronization valid-token accounting, and tensor views over valid cache prefixes. Integration into staged model graph execution remains pending.

- **2026-11-22 / Embedding lowering:** added backend-owned token-ID upload and dispatch-lifetime binding for the F32 Embedding candidate; graph compilation now resolves `token_ids` as a declared external input. Tokenizer and GPU numerical parity remain UNVERIFIED.

- **2026-11-22 / graph-ordered KV cache:** introduced internal copy-only graph nodes, explicit control dependencies and side-effect outputs; dense plans now sequence K/V cache append before Attention. Added typed planned-prefix K/V tensor views and post-sync valid-token commits. The host contract validates ordering and cache state; real-device ordering and Attention parity remain UNVERIFIED.

- **2026-11-22 / activation liveness allocation:** added dynamic-shape F32 activation allocation with last-consumer tracking, safe slot reuse for non-overlapping tensor lifetimes and retained-output pinning. Residual hidden states are versioned per layer to prevent output/input aliasing. Parallel graph lifetime analysis remains future work.

- **2026-11-22 / Android backend probe:** added an opt-in JNI diagnostics method to exercise the Dawn-first / Universal GLES backend composition and expose actual Android device limits/features without changing the llama.cpp load/generate path. Physical-device execution remains pending.

- **2026-11-22 / model weight binding:** added plan-driven single-buffer F32 and Q8_0 GGUF weight loading with persistent MemoryLease ownership. Q8_0 weights stay encoded and route to explicit MatMul_Q8_0 lowering on Dawn/GLES, including flattened 3D projections. Multi-chunk inference, quantized Embedding, and numerical backend validation remain pending.

- **2026-11-22 / operator qualification suite:** added real backend readback parity validators for F32 Multiply, MatMul, RMSNorm, SiLU, Softmax, RoPE, Attention and Embedding, with registry promotion only after CPU-reference parity. Android diagnostics runs the suite; Q8_0 remains separately gated. Physical-device evidence is pending.

- **2026-11-22 / backend validation gate:** Android backend selection now validates actual F32 operator execution and numerical readback before selecting Dawn; if Dawn is incomplete, it shuts down and tries Universal GLES. If neither passes, the caller receives an explicit failure and can retain the llama.cpp CPU route. This is baseline operator qualification only, not model-specific inference readiness.
- **2026-11-22 / dispatch integrity:** registry authorization now checks that each dispatch shader is the validated registered source, permitting only the explicit set of shape/parameter constant substitutions used by kernel builders. Shader-body changes outside those specializations fail closed.
- **2026-11-22 / runtime invocation lifecycle:** dynamic activation buffers and KV-cache tensor views can be rebuilt for changing prefill/decode lengths without clearing persistent weight bindings or cache storage. Universal GLES shader programs are cached with a fixed 64-entry bound; packed quantized buffers may use byte-granular sizes. Physical-device validation and performance measurements remain pending.

- **2026-11-22 / Q4_0 quantized execution milestone:** added direct packed-block MatMul for Dawn and Universal GLES, a CPU reference using the independent Q4_0 decoder, a per-backend validation gate, and model-loader/lowering routing for Q4_0 weights. This does not imply Q4_1/Q5/Q6_K support; those formats remain separately gated and must receive their own kernels and tests.

- **2026-11-22 / Q4_1 quantized execution milestone:** added direct Q4_1 MatMul candidates and independent reference validation for Dawn/GLES, including model-plan selection and GGUF loader/lowering coverage. Q4_0 and Q4_1 are now separate explicitly gated formats; Q5/Q6_K and other formats remain unsupported for model execution until their own direct kernels and tests are added.

- **2026-11-22 / Q4 token embedding support:** added direct Q4_0/Q4_1 Embedding kernels, CPU reference parity gates, backend registrations, GGUF architecture-plan routing and model lowerings. Full model support remains gated by all other operations, quantized tensor types, memory budget, streaming constraints, and real-device tests.

- **2026-11-22 / per-operation weight residency:** introduced streamed model execution in validated plan order, lazy GGUF weight loading per operation, dispatch/synchronization, and deterministic release of temporary weight buffers. This is the first path that can avoid holding all model weights on GPU simultaneously. Single-tensor multi-chunk compute and a full application inference session are still pending.

- **2026-11-22 / Android session + runtime routing:** connected the independent GPU engine to Android JNI and the existing runtime adapter. Full-GPU requests try AIChatGpuEngine; unsupported models, quantizations, insufficient memory, backend failures or device errors fall back to llama.cpp CPU. Runtime generation failures trigger a controlled CPU reload/retry, with partial GPU output buffered and discarded to prevent duplicate text.
- **2026-11-22 / streamed inference memory:** model weights are loaded and released one operation at a time; final prefill logits are projected from the last hidden-state row only. Current limits are F32 activations/KV cache, F32/Q4_0/Q4_1/Q8_0 weight paths, no Q8_0 embedding, and no multi-chunk execution for a single oversized tensor. Dense Transformer inference remains experimental until reference-parity and real-device matrix tests are complete.

- **2026-11-22 / registration and preflight correctness:** fixed an early return that prevented Q4_1 MatMul candidates from being registered for Dawn and Universal GLES; added fail-closed fake-backend regression checks for both registrations. GPU memory preflight now accounts for the peak padded set of weight buffers loaded by one streamed operation, not merely the largest individual tensor, and preserves SSBO allocation padding.
- **2026-11-22 / broader baseline kernel qualification:** expanded physical-backend numerical validation to cover irregular and non-workgroup-aligned shapes for binary operators, MatMul, RMSNorm, Softmax/SiLU, RoPE (including NeoX style), GQA Attention with past KV state, and Embedding. These checks qualify the tested generic F32 kernel paths, not all possible shapes, data types, drivers, or devices.

- **2026-11-22 / model-specific backend retry:** if the first backend passes baseline F32 qualification but fails the selected model's quantized-kernel validation, the inference session now shuts it down, initializes the alternate registered backend through the same backend abstraction, revalidates baseline operators and retries the model's quantized kernels. The selected backend and fallback reason are exposed in diagnostics. This is still fail-closed and does not claim both backends support the model unless that backend's own numerical tests pass.

- **2026-11-22 / model-coverage backend retry:** extended the model-specific backend retry gate beyond quantization. If the selected backend cannot satisfy the validated model operator/lowering coverage, the session now tries the alternate backend and repeats baseline-kernel, quantization, and model-coverage validation before deciding whether GPU inference is safe.

- **2026-11-22 / GGUF streamed-read efficiency:** the independent GGUF reader now retains one synchronized file stream for the reader lifetime instead of reopening the model file for every streamed weight chunk. Repeated and concurrent range reads serialize on the retained stream; the parser and weight layout remain independent of GGML.

- **2026-11-22 / GGUF metadata lookup:** the independent reader now builds a tensor-name index at parse time, replacing repeated linear scans during per-operation weight streaming with indexed lookup. The index is rebuilt from the validated tensor table and does not change file parsing or tensor offsets.

- **2026-11-22 / enforced dependency boundary:** added a CI guard that scans the backend-neutral GPU engine source for llama.cpp/GGML API includes or symbols and checks that the engine core, Dawn backend, Universal GLES backend, and Android composition targets do not link llama.cpp/GGML. The legacy runtime target may still link those libraries for the preserved CPU implementation; Vulkan remains explicitly disabled.

- **2026-11-22 / Q8_0 quantized embedding:** added direct packed Q8_0 token-embedding kernels for Dawn and Universal GLES, an independent CPU decoder reference, backend numerical validation, GGUF architecture mapping, model lowering and fail-closed inference-session validation. This removes the previous explicit Q8_0 embedding rejection when the selected backend passes its own numerical test; it does not imply that Q8_0 is validated on every GPU.

- **2026-11-22 / BPE pre-tokenizer correctness increment:** common case-insensitive English contraction suffixes (apostrophe-s, apostrophe-t, apostrophe-re, apostrophe-ve, apostrophe-m, apostrophe-ll, apostrophe-d) are now split before byte-level BPE merge ranking for the registered GPT-2/Qwen-style profiles, with a synthetic merge-boundary regression test. Built-in BPE adapters remain **UNVERIFIED**; this is not full regex-pretokenizer or model-vocabulary parity.

- **2026-11-22 / transient pool bound fix:** corrected an unsigned-underflow edge case when a released buffer was larger than the pool's half-budget cap. Oversized entries are now freed instead of accidentally being retained; a regression test checks that the full budget becomes available again.
