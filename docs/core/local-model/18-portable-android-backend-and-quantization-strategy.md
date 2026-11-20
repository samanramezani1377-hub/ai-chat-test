# Portable Android Inference: Backend and Quantization Strategy

Status: implementation guidance for `new-kor`
Pinned engine: llama.cpp `9871df5911a03a813518bcd623ee2eba91bf32aa`
Primary accelerator path: GGML WebGPU through AndroidX Dawn, OpenGL ES Compatibility
Vulkan: intentionally disabled

## Goals and non-goals

The runtime must not encode assumptions about one phone, GPU vendor, Android SoC, model family, or GGUF quantization. Device selection is capability-driven. Quantization support is operation- and shape-specific; a type being listed in a backend helper does not imply that every graph operation can execute that type on that backend.

The CPU implementation remains a complete correctness fallback. WebGPU is an optional acceleration path and must never be required to load or run a valid model. The user interface is not part of this backend work and must not be changed to expose implementation details.

Current build constraints are still explicit: Android API 29+, arm64-v8a. These are project packaging constraints, not claims that every Android device is supported.

## Runtime policy

1. Register the statically linked WebGPU backend and enumerate GGML devices.
2. Select a WebGPU GPU/IGPU only when the backend exposes one.
3. Let GGML's `supports_op` capability checks decide which graph nodes can execute on WebGPU. Unsupported nodes remain eligible for GGML scheduler placement on CPU.
4. Load with WebGPU preferred, then validate the buffers actually assigned to WebGPU.
5. Record GPU-owned, CPU-owned, and other/unassigned tensor bytes and their percentage of total model tensor bytes. Do not label partial residency as full residency.
6. If WebGPU model loading, validation, or context creation fails, release partial native state and load the same GGUF on CPU.
7. Keep diagnostics explicit about the selected execution path and the reason for fallback.

The residency percentage is deliberately based on tensor byte counts, not layer count or a specific quantization's block layout. It is a diagnostic measure; it is not a performance guarantee. Runtime profiling is still required to prove that graph compute is actually faster.

## Quantization portability

The pinned upstream WebGPU backend's `ggml_webgpu_supported_qtype` helper currently lists:

- Q1_0
- Q4_0, Q4_1
- Q5_0, Q5_1
- Q8_0
- Q2_K, Q3_K, Q4_K, Q5_K, Q6_K
- IQ1_S, IQ1_M
- IQ2_XXS, IQ2_XS, IQ2_S
- IQ3_XXS, IQ3_S
- IQ4_NL, IQ4_XS
- MXFP4, NVFP4

In the same pinned source, the `MUL_MAT` and `MUL_MAT_ID` capability branches list many of these weight types for F32 right-hand inputs, while F16 right-hand inputs have a narrower F16/F16 path. `GET_ROWS` uses the supported-qtype helper for its source tensor. Other operations have their own independent type/shape conditions.

Therefore:
- Do not add a global `if quant == Q6_K` execution path.
- Add or improve quantized kernels at the GGML operation/type dispatch boundary.
- Keep support checks adjacent to the kernel implementation and test positive and negative cases.
- A quantization is considered GPU-supported only for the exact operations, tensor shapes, and adapter limits that pass capability checks.
- Unsupported quantizations must continue through scheduler/CPU fallback, not fail model activation.

## Device portability

The AndroidX WebGPU dependency is currently alpha. The selected adapter request is explicitly WebGPU Compatibility + OpenGL ES, with Vulkan disabled. This avoids a Vulkan dependency but does not guarantee that every Android driver exposes a usable compatible adapter.

Do not select adapters by marketing name or vendor string. Use WebGPU adapter/device limits and GGML's operation capability checks. In particular, the backend rejects operations whose relevant tensor buffers exceed the adapter's `maxStorageBufferBindingSize`; this can vary between devices even when the GPU vendor is the same.

Validation targets for future device testing should include at least:
- Qualcomm Adreno
- ARM Mali
- Samsung Xclipse
- PowerVR
- MediaTek devices with different GPU/driver generations

These are test families, not a promise that every product in each family is compatible.

## Performance work order

### P0 — Correctness and truthful telemetry
- Keep Kotlin, AndroidX WebGPU, Dawn headers, Prefab package and native binary versions compatible.
- Ensure the WebGPU adapter is actually selected through the OpenGL ES Compatibility path.
- Distinguish WebGPU state from legacy OpenCL diagnostics.
- Report byte-based model residency and mixed CPU/GPU placement honestly.
- Keep CPU fallback deterministic and testable.

### P1 — Measure before tuning
Collect separate measurements for prompt prefill and token decode:
- tokens/s and elapsed time for prefill and decode
- backend assignment per graph node
- CPU time, WebGPU GPU time (when timestamp queries are supported), command submission count, and synchronization/readback time
- peak resident memory and adapter buffer limits
- thermal/frequency observations where Android exposes them

Never infer acceleration from a requested GPU layer count alone.

### P2 — Optimize the dominant kernels
Prioritize the measured bottlenecks, usually matrix multiplication/GEMV for decode and larger matrix multiplication for prefill. Then examine attention, RoPE, normalization, quantized row gathering, and KV-cache operations. Any fused kernel must preserve numerical behavior and retain a generic fallback.

### P3 — Reduce orchestration and transfers
Measure command encoder/pass/submit overhead, command batch size, buffer allocation/reuse, queue waits, and logits readback. The upstream WebGPU backend currently defines a command-submit batch size of 64; treat this as a tuning candidate, not a universal optimum. Do not change it without cross-device prefill/decode benchmarks.

### P4 — CPU fallback
Keep the generic ARMv8-A baseline portable and use runtime-dispatched optimized kernels (including KleidiAI where available). Tune thread counts and batch sizes from measured workload/device characteristics; avoid compiling the entire app for one phone's ISA. Validate both small and large models to avoid oversubscription and memory pressure.

## Acceptance criteria for new optimizations

A change is ready only when:
- It builds for the supported Android ABI.
- Unit and native CI checks pass.
- CPU-only inference still works.
- WebGPU inference is tested on at least one actual Android device before claiming device-level success.
- Unsupported operations/types fall back without corrupting output.
- Benchmarks report prefill and decode separately, with model, GGUF quantization, context, Android build, adapter, driver, and temperature conditions recorded.
- No performance claim is made from static source inspection alone.

## Known limitations / not yet proven

- No actual-device WebGPU inference or cross-vendor test is established by this document.
- AndroidX WebGPU is an alpha dependency; API/driver compatibility may change.
- The upstream quantization list is not a complete support matrix for every GGML op.
- GPU residency alone does not prove that the critical graph nodes ran on GPU or that end-to-end latency improved.
- Moving sampling fully onto GPU, reducing per-token logits readback, and tuning command submission remain measurement-driven follow-up work.


## No-ShaderF16 quantized decode path (2026-11-20)

The Android CMake integration now provides a baseline-F32 variant of the pinned
GGML WebGPU quantized MatVec path. FP16 values remain stored in their original
compact 16-bit representation, but shaders read them through
\`unpack2x16float()\` and perform scale handling in F32. Q8_1 activation
scale/sum pairs preserve their four-byte layout through
\`pack2x16float()\` / \`unpack2x16float()\`.

On adapters without WebGPU ShaderF16, the backend may now admit quantized
\`MUL_MAT\` only for the exact decode/MatVec shape:

- output type: F32;
- activation/right-hand input: F32;
- weight/left-hand input: a quantized GGML type supported by the pinned
  upstream MatVec shader;
- output batch dimension: at most four.

The existing upstream quantized dispatch covers the listed Q and IQ formats,
including Q6_K, subject to its own type dispatch and adapter limits. The
optional Q8 activation-quantization path additionally requires the
\`packed_4x8_integer_dot_product\` WebGPU feature; without it, the direct
quantized MatVec accumulator is selected.

This does **not** yet accelerate larger prompt-prefill GEMMs without
ShaderF16. Those shapes use the tiled matrix shader family, which still
contains F16 workgroup storage and remains excluded by the conservative
no-ShaderF16 operation gate. F16 tensor operations and other unsupported
shapes continue to use CPU scheduling/fallback.

The change removes a WebGPU ShaderF16 dependency from the quantized decode
MatVec path; it does not prove that a particular Android driver compiles the
WGSL, that its numerical output matches the CPU reference, or that end-to-end
generation is faster. Acceptance still requires a physical-device Q6_K
decode comparison, actual GPU-residency measurement, and separate prefill
and decode benchmarks.


## No-ShaderF16 quantized prefill GEMM (2026-11-20)

The no-ShaderF16 path now extends beyond decode-time MatVec. The pinned
upstream register-tile GEMM shader is compiled with NO_SHADER_F16 for adapters
without ShaderF16. The F32 alias is applied before common_decls.tmpl and
mul_mat_decls.tmpl are included, so common helper declarations and quantized
dequantization temporaries do not retain an F16 type dependency.

Quantized weights remain in their packed GGUF representation. Scale values
stored as FP16 are decoded into F32; temporary workgroup values use F32.
The shader library disables the F16 subgroup-matrix variant when the adapter
does not support ShaderF16. F32-input/F32-output plain GEMM uses the same
portable register-tile path.

This covers only quantized types and shapes supported by the pinned upstream
register-tile shader and its dispatch table. It does not invent kernels for
unimplemented quantization formats. F16 tensor operations remain on CPU.

The change still requires a successful native Android build and physical
device validation. Numerical parity against CPU must be checked across Q4,
Q5, Q6_K, Q8 and IQ formats. Prompt-prefill and decode throughput, GPU memory
residency, and fallback behavior must be measured independently. A successful
source patch or a reported layer count is not sufficient evidence of GPU
execution.


## No-ShaderF16 quantized prompt-prefill GEMM

The no-ShaderF16 path now includes a portable variant of the pinned upstream
register-tile matrix multiplication shader, not just the small-batch decode
MatVec kernel. It accepts F32 activations and F32 outputs with quantized
weights supported by the upstream WebGPU dequantization templates.

Quantized model weights remain packed in their original representation.
Dequantization into F32 occurs in the shader's temporary workgroup tiles;
the implementation does not expand the model file or permanently convert the
weight tensors to F32. The F16 subgroup-matrix kernel is disabled for these
adapters, and the regular register-tile kernel receives NO_SHADER_F16.
Its temporary arithmetic and workgroup storage use F32, with packed IQ scales
decoded using unpack2x16float().

This extends GPU eligibility to larger prompt-prefill MatMul shapes. It is
still limited to quantized types actually implemented by the pinned upstream
shader templates and to F32 activation/output shapes. Unsupported quant types,
F16-dependent operations, and other unsupported shapes remain eligible for
the CPU fallback. CI compilation and physical-device numerical/performance
validation are required before claiming this route works on every vendor.
