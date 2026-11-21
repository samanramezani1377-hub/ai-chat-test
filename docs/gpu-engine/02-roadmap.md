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

## Additional required inference components\n\nThe first architecture pass exposed two runtime components that were implicit in the original outline but are mandatory for true independence: a standalone tokenizer engine and a standalone sampler/generation controller. They must not call llama.cpp on the GPU route. GGUF tokenizer arrays, normalization/pre-tokenization rules, special token IDs, BPE/Unigram merge behavior, and exact parity tests must be implemented before a first model can be considered end-to-end supported.\n\n## Immediate next increments

1. Keep the active inference route unchanged.
2. Add a standalone `ai_chat_gpu_engine` C++ library with no llama.cpp/GGML include or link dependency.
3. Define backend/device/capability/status contracts and a registry.
4. Bridge Universal GPU Adapter's verified GLES probe results into generic diagnostics without treating them as inference readiness.
5. Add source-level architecture guards ensuring the standalone engine target does not include GGML/llama headers.
6. Implement Dawn as a separate backend target linked directly to the pinned Dawn build, after verifying the available Dawn C API and lifecycle constraints.
7. Implement a real command queue/buffer API and test it before adding tensor kernels.\n\n## Implemented increments\n\n- **2026-11-21 / contract + bridge:** backend-neutral device/buffer/command/tensor contracts, registry, and a diagnostics bridge from Universal GPU Adapter. The bridge deliberately reports `inference_ready=0`.\n- **2026-11-21 / Dawn backend foundation:** a separate `ai_chat_gpu_backend_dawn` target requests the OpenGL ES Compatibility adapter, reads Dawn limits/features, creates buffers, uploads/downloads, records buffer copies and WGSL compute dispatches, and runs a small integer compute/readback self-test. It is not yet selected by the Android runtime and does not yet implement model operators, quantization, GGUF tensor loading, or inference. **UNVERIFIED:** Android NDK build and real-device Dawn self-test until run on hardware/CI.\n- **2026-11-21 / tensor, memory and graph foundations:** checked tensor shape/stride/byte-range calculations, budgeted RAII memory leases, fail-closed operator-kernel selection, dependency-aware graph ordering, cancellation-aware command recording, one-submit/one-sync scheduler behavior, and host-side contract tests. These are foundational contracts, not model-level feature support.\n- **2026-11-21 / independent GGUF reader:** parses GGUF v1-v3 header, scalar metadata and array descriptors, tensor directory, alignment and offsets; calculates storage byte ranges for the listed known tensor types and supports bounded chunk reads. Quantized type layouts are metadata-only here: their GPU decoders and kernels remain **UNVERIFIED**. Unknown types are preserved as unknown and cannot be streamed as sized tensors.\n- **2026-11-21 / model registries:** added separate quantization-format and architecture-adapter registries plus a fail-closed model preflight report. No architecture adapter or quantization decoder is currently registered as validated, and preflight intentionally never marks a model inference-ready before graph/operator/kernel validation.\n- **2026-11-21 / first operator candidate:** added a generic F32 Add WGSL candidate, shape/buffer validation and CPU reference implementation. The kernel is explicitly `UNVERIFIED`; registry selection rejects it until real Dawn/WebGPU numerical parity tests promote it.\n- **2026-11-21 / engine lifecycle API:** added explicit backend registration, device enumeration, backend initialization, graph execution, shutdown and runtime diagnostics. This API is not yet connected to the app router; it reports `inference_ready=false` by design.\n- **2026-11-21 / GGUF weight streaming foundation:** reads validated tensor ranges in bounded, block-aligned chunks and uploads them through the backend-managed memory budget. Current behavior still retains all chunks for the loaded tensor; true layer-wise eviction/offload and KV-cache allocation are pending.

## Validation policy

CI/source checks are useful but not hardware evidence. Do not wait for CI to proceed with independent implementation work, but never describe a CI pass as real-device compatibility. Real-device qualification requires the exact GPU, OS, driver, Dawn version, features/limits, test suite and test result to be recorded.
