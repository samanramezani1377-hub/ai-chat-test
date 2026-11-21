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

## Additional required inference components

The first architecture pass exposed two runtime components that were implicit in the original outline but are mandatory for true independence: a standalone tokenizer engine and a standalone sampler/generation controller. They must not call llama.cpp on the GPU route. GGUF tokenizer arrays, normalization/pre-tokenization rules, special token IDs, BPE/Unigram merge behavior, and exact parity tests must be implemented before a first model can be considered end-to-end supported.

## Universal GPU Adapter integration status

The engine now exposes `GpuEngine::probe_universal_adapter(DeviceInfo&)` and maps the existing persistent EGL/OpenGL ES probe, limits, renderer/vendor strings and real compute self-test result into engine diagnostics. This is deliberately discovery-only: it does not create an executable OpenGL ES backend, because the current UniversalGpuAdapter does not yet expose a safe owner-thread command/buffer API to the engine. The OpenGL ES compute execution backend remains **UNVERIFIED / NOT IMPLEMENTED**.

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
- **2026-11-21 / independent GGUF reader:** parses GGUF v1-v3 header, scalar metadata and selected tokenizer arrays (tokens, merges, scores and token types), tensor directory, alignment and offsets; calculates storage byte ranges for the listed known tensor types and supports bounded chunk reads. Quantized type layouts are metadata-only here: their GPU decoders and kernels remain **UNVERIFIED**. Unknown types are preserved as unknown and cannot be streamed as sized tensors.
- **2026-11-21 / model registries:** added separate quantization-format and architecture-adapter registries plus a fail-closed model preflight report. A Llama dense metadata adapter exists but remains **UNVERIFIED**. The Q8_0 host decoder is marked validated after block-level tests; no GPU quantized MatMul kernel is validated. Preflight intentionally never marks a model inference-ready before tokenizer, graph/operator/kernel and model-level validation.
- **2026-11-21 / F32 operator candidates:** added generic Add, MatMul, RMSNorm, SiLU and Softmax WGSL candidates, shape/buffer validation and CPU reference implementations. All GPU kernels remain `UNVERIFIED`; registry selection rejects them until real Dawn/WebGPU numerical parity tests promote them.
- **2026-11-21 / engine lifecycle API:** added explicit backend registration, device enumeration, backend initialization, graph execution, shutdown and runtime diagnostics. This API is not yet connected to the app router; it reports `inference_ready=false` by design.
- **2026-11-21 / GGUF weight streaming foundation:** reads validated tensor ranges in bounded, block-aligned chunks and uploads them through the backend-managed memory budget. Current behavior still retains all chunks for the loaded tensor; true layer-wise eviction/offload and KV-cache allocation are pending.
- **2026-11-21 / independent sampler:** added temperature, top-k, top-p, repetition penalty, seeded sampling, EOS and stop-sequence handling with deterministic host-side tests. This is not yet integrated with a model/tokenizer generation loop.
- **2026-11-21 / runtime selection policy:** added a pure fail-closed selector that chooses the dedicated GPU engine only when device, tokenizer, architecture, quantization, operator coverage, model graph and memory checks all pass; otherwise it chooses the existing CPU llama.cpp path when available. This policy is not yet connected to the Android app router.\n- **2026-11-21 / KV cache primitive:** added checked memory estimation, per-layer K/V buffer allocation under the engine memory budget, sequential append validation and logical layer reset. Initial validated storage type is F32; F16/BF16 remain constrained by alignment and backend validation. This is a storage primitive, not yet connected to Attention or model generation.\n- **2026-11-21 / Q8_0 decoder:** added a block-accurate 32-value / 34-byte host decoder and FP16 scale conversion with finite-value and size validation. Added a generic WGSL dequantization candidate that reads packed Q8_0 bytes directly from a storage buffer. Host decoder tests are present; the GPU candidate and quantized MatMul remain **UNVERIFIED**.

## Validation policy

CI/source checks are useful but not hardware evidence. Do not wait for CI to proceed with independent implementation work, but never describe a CI pass as real-device compatibility. Real-device qualification requires the exact GPU, OS, driver, Dawn version, features/limits, test suite and test result to be recorded.
