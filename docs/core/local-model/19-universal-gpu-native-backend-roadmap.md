# Universal GPU Adapter → Native Android GPU Backend

Status: Phase 1 foundation in progress; persistent owner-thread EGL/GLES context and compute self-test implemented, device validation pending.
Target branch: `new-kor`.
Platform policy: Android `arm64-v8a`, OpenGL ES compute, no Vulkan.

## Goal

Evolve `universal_gpu_adapter.cpp/.h` from a persistent EGL/OpenGL ES compute foundation into the owner of a vendor-neutral native GPU execution backend. The backend must eventually execute supported GGML graph operations directly through Android's EGL + OpenGL ES compute API, without Dawn, WebGPU, OpenCL, vendor-specific SDKs, or a third-party GPU inference backend.

This does **not** mean removing the Android GPU driver or platform EGL/GLES libraries. Those are the hardware execution interface and remain required. It also does not initially replace llama.cpp/GGML graph construction, model loading, tokenization, scheduling, or CPU fallback.

## Non-negotiable constraints

- Keep `arm64-v8a` as the supported Android ABI.
- Keep Vulkan disabled.
- Do not assume Adreno, Mali, a particular phone, model, or quantization.
- Detect capabilities at runtime; never spoof extensions or vendor identity.
- F32 is the correctness baseline. FP16 and packed/quantized paths are opt-in only after capability and correctness tests.
- Unsupported operations and data types must remain on CPU; never claim full GPU execution from buffer residency alone.
- Preserve a reliable CPU fallback at every migration phase.
- Do not remove the current WebGPU/Dawn path until the native backend passes correctness, stability, and performance gates on multiple GPU families.
- Keep each phase independently buildable and revertible.

## Target architecture

`UniversalGpuAdapter` will own the complete native GPU path:

1. **Platform/context layer** — EGL display, context and pbuffer/surface lifecycle; a dedicated owner thread; safe context binding and teardown.
2. **Capability model** — GLES version, compute limits, SSBO limits, shader language features, extensions, workgroup limits and memory budget. Capability hints must be separated from features proven by compiling/running a test shader.
3. **Memory manager** — SSBO allocation, aligned suballocation/pooling, bounded allocation, buffer reuse, explicit ownership, transfer/readback and device-loss handling.
4. **Shader system** — GLSL ES 3.10 baseline compiler/linker, generated kernel variants, source hashing, persistent cache with versioned keys, compile diagnostics and cache invalidation.
5. **Compute queue** — ordered dispatch, barriers, fences, synchronization and bounded in-flight work. All GL calls must execute on the context-owning thread.
6. **Kernel registry** — operation + type + layout + shape capability matching. Each kernel declares exact supported combinations and required device features.
7. **GGML bridge** — a custom GGML backend registration, device, buffer type and graph-compute implementation. The bridge must report unsupported nodes accurately so GGML can schedule them on CPU.
8. **Diagnostics/profiling** — per-operation dispatch, fallback reason, bytes transferred, allocation high-water mark, shader cache hit/miss, GPU time and CPU/GPU parity information.

The adapter should expose a narrow C++ interface to the GGML bridge. GGML internals must not leak into the EGL/context or shader-management layer.

## Migration phases and acceptance gates

### Phase 0 — Preserve the current runtime

- Keep the existing WebGPU/Dawn backend and CPU fallback unchanged.
- Make the native CMake patch system modular and fail-fast.
- Keep CI checks aware of the extracted CMake modules.
- No inference routing changes in this phase.

**Status:** CMake modularization and the MatMul verification-variable fix are committed. The most recently inspected completed Actions run passed all jobs, but CI for these newer commits still needs confirmation.

### Phase 1 — Native GLES compute foundation

Implement inside the Universal GPU Adapter:

- persistent context lifecycle on a dedicated worker thread;
- a real compute-shader compile/link test;
- SSBO allocate/write/dispatch/readback;
- explicit memory barriers and fences;
- a deterministic vector-add and reduction self-test against CPU results;
- structured failure reporting and cleanup after partial initialization.

**Current progress:** `PersistentGlesProbe` creates and owns the EGL display/context/pbuffer on a dedicated worker thread. The context remains current on that same thread after startup; the baseline integer compute shader compiles, dispatches, writes an SSBO, applies a memory barrier, reads the result back, and checks the exact CPU reference output. Teardown is also performed on the owner thread. The process-lifetime adapter has an explicit shutdown API, and its static owner performs shutdown during native-library teardown. GPU selection still requires the compute self-test to pass; raw GLES 3.1 capability detection alone is insufficient.

**Remaining:** a real command queue/API for GPU operations, bounded SSBO memory manager, shader/program cache, explicit fences, repeatable init/teardown and concurrent-rendering tests, and physical-device validation. No model operations are routed to this backend yet.

**Acceptance:** tests pass on at least two distinct Android GPU families; repeated init/teardown and concurrent app rendering do not crash or leak EGL state.

### Phase 2 — Custom GGML backend shell

- Register a separate backend name and device through GGML's backend API.
- Implement device discovery, buffer type, tensor transfers, graph support checks and safe graph execution.
- Initially advertise only operations with verified GLES kernels.
- Preserve CPU scheduling for every other operation.
- Add a backend self-test independent of model loading.

**Acceptance:** GGML selects the custom backend only for supported nodes; all unsupported nodes remain correct on CPU.

### Phase 3 — F32 inference kernels

Add and validate the basic operations needed by transformer graphs, prioritizing:
- copy/contiguous and tensor transfers;
- add, multiply, scale and simple elementwise operations;
- RMS normalization;
- SiLU and supported activation combinations;
- RoPE;
- softmax;
- get-rows/gather;
- F32 matrix multiplication.

Each kernel requires CPU-reference tests across edge shapes, non-aligned dimensions, and buffer boundaries.

### Phase 4 — Quantized kernels

Add quantized formats one at a time, with explicit format/layout tests:
- Q8_0;
- Q4_0;
- Q4_K;
- Q5_K;
- Q6_K;
- additional formats only when required and independently verified.

Do not assume that decoding a quantized block correctly proves matrix multiplication correctness. Test dequantization, accumulation, tail handling, and numerical error separately.

### Phase 5 — Transformer execution coverage

Add the remaining attention, cache and graph operations only after the primitive kernels are stable. Validate Prefill and Decode separately, including long context, grouped-query attention, recurrent/hybrid architectures where applicable, and cancellation/reload lifecycle.

### Phase 6 — Differential validation and rollout

- Run identical prompts and fixed inputs through CPU and custom GPU paths.
- Compare tensor outputs within documented tolerances.
- Run deterministic repeated tests, stress tests, memory-pressure tests and thermal-duration tests.
- Benchmark end-to-end TTFT, Prefill, Decode tokens/s, peak memory and energy/thermal behavior.
- Enable custom GPU execution by operation capability, then by default only after multi-device acceptance.

### Phase 7 — Remove legacy GPU backend dependencies

Only after Phase 6 passes:
- stop building/linking `ggml-webgpu` and Dawn;
- remove WebGPU-only CMake patches and runtime selection;
- remove OpenCL backend, loader, headers and diagnostics if no remaining feature requires them;
- keep EGL/GLES platform libraries and the custom GGML backend;
- verify that the app still builds and runs CPU-only when GLES compute is unavailable.

## Risks and explicit non-goals

- OpenGL ES compute support does not imply that every GPU driver implements every optional feature correctly.
- A successful EGL context or shader self-test does not prove transformer-kernel correctness.
- GPU memory availability is not reliably inferred from a single GLES query; allocations must be bounded and recoverable.
- Driver shader compilers may have vendor-specific bugs. The baseline must remain portable GLSL ES and fail closed.
- A custom backend is a substantial engineering project. This plan does not claim that the current GLES context/self-test foundation already performs model inference.
- Replacing llama.cpp/GGML itself is out of scope for this migration; the first goal is to remove third-party GPU backend engines while retaining the mature model/runtime orchestration.

## Current source boundary

At present, `universal_gpu_adapter.cpp` owns a persistent EGL pbuffer/context on a dedicated worker thread and runs one integer compute self-test with SSBO readback. It does not yet expose a general command queue or persistent GPU memory manager, register a GGML backend, or execute model tensors.

The current model path still selects `ggml-webgpu` backed by Dawn/OpenGL ES. The native adapter is only a capability probe and must not be reported as the active inference backend until the GGML bridge and kernels are implemented and verified.
