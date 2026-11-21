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
