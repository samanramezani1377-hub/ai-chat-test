# Universal GGUF-driven GPU model runtime

Status: **architecture target and implementation contract**. This document does not claim universal model execution is already implemented.

## Goal

The GPU runtime must consume GGUF files as the source of truth and translate their contents into a backend-neutral execution representation. Adding a model must not normally require a new C++ class or a new GPU backend implementation for that model.

The CPU inference path remains llama.cpp. Vulkan remains disabled. GPU backends and model execution are independent layers.

## What GGUF gives us

The reader must expose and preserve:

- GGUF version, alignment, file size, and metadata key/value types.
- general.architecture, model identity, and architecture-prefixed hyperparameters.
- Every tensor name, dimensions, GGML type, offsets, and validated storage size.
- Tokenizer model, vocabulary, token scores/types, merges, special-token IDs, and tokenizer configuration.
- Safe bounded tensor streaming without loading the full weight file into RAM.

GGUF does not necessarily encode the complete forward-pass program. Metadata and tensor shapes do not always specify operator ordering, attention variants, gating, recurrent-state transitions, RoPE variants, masks, normalization placement, or multimodal routing. These semantics must come from a validated declarative architecture description or a verified generic inference rule. Never guess them.

## Target pipeline

1. **Container and metadata reader** — parse and validate GGUF independently of any architecture.
2. **Structure analyzer** — infer tensor roles, dimensions, layer patterns, repeated blocks, quantization formats, and candidate architecture family from metadata plus tensor evidence. Return confidence and evidence; do not silently force a match.
3. **Semantic model description** — represent the forward pass as declarative blocks and operation parameters. Reuse shared descriptions for architecture families. Architecture-specific differences should be data/configuration where possible, not duplicated C++ implementations.
4. **Backend-neutral graph IR** — construct a graph from semantic operations, shapes, dependencies, weights, control dependencies, and state/cache effects.
5. **Operator capability resolver** — resolve each graph op and each weight type to a validated implementation for the selected backend. Quantization decoding and GPU kernels are separate capabilities.
6. **Memory planner and streaming scheduler** — account for weights, activations, scratch, KV/recurrent state, alignment, and device limits. Stream weights when needed; reject unsupported single-tensor chunking rather than pretending it is supported.
7. **Numerical validation gate** — compare operation outputs and full-model logits against a trusted reference with documented tolerances and test vectors.
8. **Device execution gate** — run backend self-tests and model smoke tests. Only then may preflight mark a model/backend combination ready.

## Required invariants

- No model-specific GPU kernels unless an operation truly has unique semantics or a measured optimization; ordinary architecture variation belongs in the model description.
- No backend-specific model graph. The same IR must be lowerable by every backend that claims support.
- No assumption that all models are dense Transformers. The IR must support dense, MoE, hybrid attention/recurrent, multimodal, and future block compositions.
- Quantization format support is independent from architecture support. A model is executable only if every used tensor type has a validated decoder and selected-backend kernel or a safe, explicitly supported conversion path.
- Tokenizer parsing is independent from model architecture and is sourced from the GGUF file. Tokenizer algorithm support still requires exact reference parity.
- Every inference request must fail closed when graph semantics, tensor mapping, operator lowering, memory bounds, or numerical validation are incomplete.
- Diagnostics must identify the exact missing metadata, tensor, semantic rule, operator, quantization capability, or validation gate.

## Implementation stages and exit criteria

### Stage A — inspectability and safe discovery

Implementation on `engine`: a first-pass evidence-only analyzer now classifies tensor-name patterns for dense attention, MoE experts/router, recurrent/state-space paths, and multimodal components. It reports evidence and a deterministic evidence-strength score; it does not infer executable semantics or mark an architecture supported. Unknown architectures now report the candidate structure family and score instead of only a registry miss. Synthetic hybrid/MoE and unknown-architecture contract tests are included.
- Expose complete, bounded metadata/tensor inventory diagnostics.
- Build structure summaries without requiring an architecture adapter.
- Detect duplicate/missing/contradictory tensor patterns and arithmetic overflow.
- Add synthetic GGUF fixtures covering multiple tensor layouts and unknown architectures.

Exit: unknown architecture files can be inspected and diagnosed without being misreported as executable.

### Stage B — declarative model semantics
- Define a versioned schema for reusable blocks and operation parameters.
- Implement a generic graph builder from that schema.
- Add schema validation and graph invariants.
- Move current llama/qwen2/qwen3 descriptions toward shared declarative definitions without changing their validated behavior.

Exit: graph generation is data-driven and does not branch on a model name throughout execution code.

### Stage C — operator and quantization coverage
- Separate logical ops from backend kernel IDs.
- Build a capability matrix per backend, op, and quantization type.
- Add independent tests for dequantization, matmul, embedding, normalization, attention, RoPE, MoE routing, and state/cache operations as applicable.

Exit: graph compilation reports complete, backend-specific coverage and all required kernels have numerical validation.

### Stage D — end-to-end model validation
- Compare logits and generated token IDs against a trusted reference on representative models.
- Validate prompt prefill, single-token decode, long context, cancellation, memory pressure, and streamed weights.
- Test more than one GPU vendor/device and multiple quantization families before claiming broad compatibility.

Exit: the exact model/quantization/backend combination passes structural, numerical, and device tests.

## Implemented hybrid gated-delta-net semantic recipe

The `engine` branch now contains one reusable `HybridGatedDeltaNetRecipe` for the `qwen35` GGUF architecture. It is a family-level declarative graph recipe, not a Qwen3.5-specific GPU backend. It consumes the GGUF metadata arrays for recurrent-layer flags and multi-axis RoPE sections, checks required tensor names/shapes/storage, and emits a backend-neutral plan containing full-attention and gated-delta-net layer paths.

The graph IR now carries string, integer-array, and floating-array semantic attributes so architecture parameters such as RoPE sections remain data rather than backend-specific code. The GGUF reader also preserves typed scalar arrays instead of skipping non-tokenizer arrays. Current GGML storage layouts are recognized for TQ1_0, TQ2_0, MXFP4, NVFP4, Q1_0, Q2_0 and F8_E4M3; removed historical type IDs are identified as removed and remain non-loadable.

**Important validation boundary:** this recipe currently builds and structurally validates the graph only. The GPU implementation still lacks validated `MultiAxisRoPE`, `SSMConv1D`, `GatedDeltaNet`, `GdnL2Norm`, split, sigmoid/softplus and recurrent-state lifecycle lowerings. It deliberately leaves `gpu_ready=false` and carries explicit pending-validation blockers. No Qwen3.5 GPU execution or numerical parity is claimed. Qwen3.5-MoE is not registered by this recipe.

## Current known blocker: Qwen3.5 GGUF

The reported device log says Unsupported model architecture: qwen35. This is an architecture registry failure, not a tokenizer failure. The Qwen3.5 tokenizer profile has been merged separately; it does not prove that the Qwen3.5 forward pass is supported.

Before implementing Qwen3.5 semantics, inspect the exact GGUF's architecture metadata and tensor inventory. Qwen3.5-family models may use hybrid attention/recurrent components; the existing dense-transformer plan must not be reused by changing only its architecture name.

Required evidence for the specific file:
- all architecture-prefixed metadata keys and typed values;
- tensor names, dimensions, GGML type IDs/names, and storage sizes;
- tokenizer metadata summary (not the entire vocabulary in the diagnostic log);
- a reference implementation/version and a parity test corpus.

## Completion definition

Universal is not a claim that every future GGUF works without any new code. It means new models are handled by the generic reader, analyzer, declarative semantics, graph IR, and existing operator library whenever their operations are already known. A genuinely new operation or quantization format may require one reusable implementation, not a new model-specific runtime. Unsupported combinations remain explicitly unsupported until all validation gates pass.
