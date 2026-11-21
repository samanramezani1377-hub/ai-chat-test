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

## Immediate next increments

1. Keep the active inference route unchanged.
2. Add a standalone `ai_chat_gpu_engine` C++ library with no llama.cpp/GGML include or link dependency.
3. Define backend/device/capability/status contracts and a registry.
4. Bridge Universal GPU Adapter's verified GLES probe results into generic diagnostics without treating them as inference readiness.
5. Add source-level architecture guards ensuring the standalone engine target does not include GGML/llama headers.
6. Implement Dawn as a separate backend target linked directly to the pinned Dawn build, after verifying the available Dawn C API and lifecycle constraints.
7. Implement a real command queue/buffer API and test it before adding tensor kernels.

## Validation policy

CI/source checks are useful but not hardware evidence. Do not wait for CI to proceed with independent implementation work, but never describe a CI pass as real-device compatibility. Real-device qualification requires the exact GPU, OS, driver, Dawn version, features/limits, test suite and test result to be recorded.
