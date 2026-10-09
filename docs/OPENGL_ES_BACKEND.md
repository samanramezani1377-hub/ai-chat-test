# OpenGL ES GGML backend

The Android runtime uses a generic OpenGL ES 3.1 compute backend. Model-specific code is not embedded in the backend: model coverage is expressed through GGML operations and tensor/data-type support.

## Runtime contract

- GPU backend: OpenGL ES 3.1 compute.
- CPU fallback: disabled by the Android product contract.
- Vendor selection: capability-driven; no Adreno/Qualcomm-specific requirement.
- llama.cpp revision: `9871df5911a03a813518bcd623ee2eba91bf32aa`.
- Backend registration: static Android registration.
- Unsupported GPU operations must fail rather than silently moving to CPU.

## Current generic operation coverage

| Operation | Current coverage |
| --- | --- |
| VIEW / RESHAPE / PERMUTE / TRANSPOSE | metadata/no-dispatch |
| ADD / MUL | F32, equal-shape elementwise |
| SCALE / SCALE_BIAS | F32 contiguous |
| RMS_NORM | F32 |
| SIGMOID / SOFTPLUS | F32 |
| GELU / GELU_ERF / GELU_QUICK | F32 approximation |
| TANH / EXP / LOG | F32 |
| SOFT_MAX | F32 |
| ROPE | F32 + I32 positions |
| MUL_MAT | F32 x F32; Q4_0/Q6_K/Q8_0 x F32 |
| GET_ROWS | F32/F16/Q4_0/Q6_K/Q8_0 weights -> F32 |
| SILU | F32 |
| SSM_CONV | F32 |
| GATED_DELTA_NET | F32 when the device exposes at least 7 compute SSBO blocks |

This table is intentionally conservative. An operation is only advertised through `supports_op()` when the current shader implementation matches the supported tensor types and shapes.

## Capability-aware OpenGL ES

OpenGL ES 3.1 makes compute limits implementation-dependent. In particular, the minimum `MAX_COMPUTE_SHADER_STORAGE_BLOCKS` is 4. The Gated DeltaNet shader currently uses 7 storage blocks, so it is compiled/enabled only on devices that expose enough storage blocks. The rest of the backend remains usable on devices with lower limits.

The runtime records:

- GL vendor
- GL renderer
- GL version / GLSL version
- max compute SSBO blocks
- max compute workgroup X
- max SSBO block size

## Quantization policy

Quantized kernels are not tied to a model architecture. The current matrix deliberately starts with Q4_0, Q6_K and Q8_0 and can be extended with Q4_K/Q5_K and other GGUF types after backend-operation tests pass.

## Validation

The canonical llama.cpp validation mechanism is `test-backend-ops`. The backend should eventually be added to an Android/device operation matrix that compares small deterministic tensors against a CPU reference. Model-level validation then runs the same graphs for Qwen, Llama, Gemma, Phi, Mistral and other architectures.

A physical Android GPU run is still required before claiming full device-wide numerical correctness or performance parity.

## Design rule

Do not add a Qwen-specific shader to solve a model-specific symptom. If a model requires an operation, implement that GGML operation generically and add the model to the coverage tests.
