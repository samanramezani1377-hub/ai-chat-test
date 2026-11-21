# Independent GPU Inference Engine — Architecture

Status: **DESIGN BASELINE**. Any capability not implemented and validated is **UNVERIFIED**.

## 1. Target topology

```text
Android / AI Core
  |
  +-- Runtime Router
  |     +-- CPU Runtime Adapter --------------------> llama.cpp (unchanged)
  |     |
  |     +-- GPU Runtime Adapter
  |            +-- GPU Engine API
  |                 +-- Model Format Readers (GGUF first; independent parser)\n  |                 +-- Tokenizer Registry (BPE / Unigram / SentencePiece adapters)\n  |                 +-- Sampling / Generation Controller
  |                 +-- Architecture Adapters
  |                 +-- Graph Builder / Optimizer
  |                 +-- Execution Scheduler
  |                 +-- Operator Registry
  |                 +-- Kernel Registry / Compiler Cache
  |                 +-- Tensor Engine
  |                 +-- Memory Engine / KV & recurrent state
  |                 +-- Quantization Decoders
  |                 +-- Backend Registry
  |                       +-- Dawn WebGPU backend (initial execution backend)
  |                       +-- Universal GLES adapter integration
  |                       +-- future independent backends
  |
  +-- Controlled GPU failure -> existing CPU Runtime Adapter
```

The GPU engine must not call GGML execution APIs, GGML scheduler/graph APIs, or llama.cpp model/context/generation APIs. The CPU runtime may continue using them.

## 2. Layer contracts

### Runtime Router
Receives the model path, context and generation parameters. It selects GPU only after device, format, architecture, operator, quantization and memory preflight all pass. It owns fallback policy and records the exact reason for a GPU rejection/failure.

### Backend API
Backend-neutral device enumeration, capability snapshots, buffer allocation, command recording/submission, synchronization, readback/upload, timestamps where available, health/device-loss reporting, and backend diagnostics. The engine must not include Dawn/WebGPU headers outside the Dawn backend implementation.

Initial backend priority: Dawn WebGPU using the already pinned Dawn source build and OpenGL ES compatibility route. Vulkan remains disabled. The Universal GPU Adapter is a separate GLES capability/self-test facility and must be consumed through a bridge; it must not be conflated with Dawn's adapter/features.

### Tensor API
Own shape, strides, data type, layout, alignment, location, ownership and lifetime. Dense and future sparse/block-sparse representations must be explicit. Tensor metadata must not depend on `ggml_tensor`.

### Memory API
Own persistent weights, temporary arenas, staging buffers, KV cache, recurrent state, attention scratch, expert residency, pooling and budget accounting. Allocation failure must be recoverable. Weight streaming/offload is an explicit policy; do not assume all weights fit in device memory.

### Graph and Scheduler
Build a backend-independent operation DAG, validate dependencies and dynamic dimensions, plan lifetimes, reuse safe allocations, submit work in dependency order, and support prefill/decode/cancellation. Graph reuse is allowed only when all shape/state invariants match.

### Operator and Kernel registries
An operator declares input/output constraints, semantic behavior, supported dtypes/layouts, reference implementation and candidate kernels. Kernel selection considers backend capabilities, shapes, quantization and numerical requirements. Unsupported operations fail closed or are explicitly delegated through a validated CPU implementation; they must never return fabricated outputs.

### Quantization
Each format has a versioned descriptor, block layout, scale/zero-point semantics, decoder/reference implementation and kernel implementation. A format becomes supported only after independent decoder tests, numerical comparisons and real model tests. GGUF type identifiers are file-format metadata, not an execution contract.

### Architecture adapters
Adapters translate model metadata and tensor names/shapes into the generic graph/model representation. Dense, GQA/MHA, MoE, hybrid/recurrent, convolutional and multimodal families are extension points, not implied support.

### Model readers
GGUF reader first: validate header/version/metadata, parse tensor directory and offsets, check bounds/overflow/alignment, map supported tensor types and stream tensor bytes. Reuse of GGUF file-format definitions is acceptable only if it does not import GGML tensor execution or runtime dependencies. Other formats are separate reader plugins.

## 3. Universal GPU Adapter reuse

The Universal GPU Adapter remains the owner of the persistent EGL/GLES context and exposes serialized synchronous/asynchronous tasks on its dedicated context thread. A separate Universal GLES Backend consumes that API through a bridge:
- GLES context, limits and compute self-test are reported as GLES-specific facts.
- The GLES backend owns buffer allocation/lifetime, copies, upload/readback, GLSL ES dispatch and synchronization.
- Backend initialization runs an additional integer compute/readback self-test.
- No Dawn/WebGPU feature is inferred from GLES extensions.
- A backend self-test does not mean model kernels or any model architecture are validated.
- Dawn/WebGPU and Universal GLES are separate backends under the same generic contract.

## 4. Runtime API (target)

- initialize / shutdown
- enumerate devices / get capabilities
- inspect model / estimate memory / check support
- load / unload
- set context and generation parameters
- prefill / decode / cancel
- retrieve token stream
- retrieve performance, memory and diagnostic snapshots

The Android-facing API should preserve the existing model-import workflow. CPU selection must remain transparent and unchanged.

## 5. Isolation rules

- The independent engine target must compile without llama.cpp or GGML include paths.
- Only the legacy CPU adapter may include llama.cpp execution APIs.
- During migration, the old GPU path may remain compiled but must be explicitly labeled legacy.
- No GGML tensor or backend object may cross the GPU-engine API boundary.
- The Universal GPU Adapter bridge may consume its public capability struct only.
- Build-time dependencies for Dawn belong only to the Dawn backend target.
- The Android ABI remains arm64-v8a as currently configured; no Vulkan dependency is introduced.

## 6. Capability states

Use distinct states: `DISCOVERED`, `SELF_TESTED`, `BACKEND_READY`, `OPERATOR_READY`, `MODEL_VALIDATED`, `DEVICE_VALIDATED`. A lower state must not be promoted to a higher state by inference or configuration alone.

For each operator/quantization/architecture/backend combination, record one of:
- `SUPPORTED_VALIDATED`
- `SUPPORTED_LIMITED`
- `UNVERIFIED`
- `UNSUPPORTED`
- `FAILED_VALIDATION`

## 7. Failure policy

GPU initialization, memory allocation, shader compilation, command submission, synchronization, device loss, model load, prefill and decode failures must return structured errors. The router may restart the model on the existing CPU runtime only after GPU resources are safely released. Cancellation must be propagated and no GPU worker may outlive its owning runtime.

## 8. Initial implementation boundary

The first code increment introduces backend-neutral contracts and a bridge to Universal GPU Adapter diagnostics. It intentionally does not route inference through the new engine and does not claim a working Dawn inference backend. This preserves the CPU path and existing GPU inference while the independent engine is built.


### Tokenizer adapter boundary

Tokenizer behavior is separate from model architecture and from GGUF parsing. `TokenizerRegistry` selects an adapter by the exact pair `tokenizer.ggml.model` + `tokenizer.ggml.pre`. The initial SentencePiece and byte-level BPE adapters parse the GGUF token vocabulary, merges/scores and special-token metadata without llama.cpp. Their algorithms are intentionally registered as `UNVERIFIED`; `TokenizerEngine` rejects encode/decode until profile-specific parity tests promote them. Unknown tokenizer profiles fail preflight instead of silently falling back to a different algorithm.

### Android backend composition

`register_android_gpu_backends` registers Dawn before Universal GLES. `initialize_android_gpu_engine` tries the ordered preference list and records fallback diagnostics. This only establishes backend readiness and compute self-tests; it does not imply that a model graph, tokenizer, quantized kernel or full inference path is validated.


### Model graph lowering and execution

The lowering registry is keyed by backend and operator, and resolves tensors through an explicit model execution context. Built-in F32 lowerings currently cover Add, Multiply, MatMul, RMSNorm, SiLU, Softmax and RoPE candidates. Embedding and Attention are deliberately absent until token-ID upload ownership and KV-cache semantics are implemented. Lowering produces only a candidate dispatch: OperatorRegistry must independently authorize a backend-specific validated kernel before a graph node can be added. The execution wrapper compiles the complete graph before submitting it, then marks only that graph run as executed after synchronization. It never promotes a model to gpu_ready on the strength of one graph run.


### Token-ID and embedding binding

The model execution context carries token IDs and the selected backend. The Embedding lowering allocates a device token-ID buffer, uploads the exact token sequence, and binds it with the weight/output tensors. The graph compiler treats `token_ids` as a declared external host input and does not fabricate a Tensor for it. The upload buffer lifetime is held by the compute dispatch. This is a structural path only; tokenizer and Embedding kernel parity remain separate validation gates.


### Graph-ordered KV cache and attention

KVCacheAppend is represented as an internal copy-only graph node. It records two bounded device-buffer copies (K and V) in the same backend command batch as compute operations, after their producer nodes and before Attention through explicit control dependencies. The graph compiler allows side-effect outputs without fabricating tensor storage. Attention binds K/V tensor views sized to `past_length + sequence_length`. The cache valid-token counters are advanced only after the entire graph has submitted and synchronized; failed execution leaves the logical cache length unchanged so the same append range can be safely overwritten on retry. Internal copy nodes do not masquerade as validated math operators.


### Activation allocation and liveness

The activation allocator derives each output shape from the model plan and runtime token count, computes last-use indices from data dependencies, and reuses a device buffer only when its prior tensor's final consumer precedes the new producer. Retained outputs (currently logits) are pinned until the caller releases the execution context. Each unique slot is backed by a MemoryLease retained by the context. This is a simple graph-order liveness allocator for the current sequential scheduler; it does not yet model parallel branches or asynchronous multi-queue overlap.
