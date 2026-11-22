# Repository work rules

## Single working branch

- All ongoing work for the Android GPU inference engine MUST be committed directly to the existing `engine` branch.
- Do not create new branches from `engine` or from any other branch for this project.
- Do not open pull requests for work that is already being done on `engine`.
- Before every write, read the current `engine` head and use a compare-and-swap update so concurrent changes are not overwritten.
- Keep the CPU inference path on llama.cpp unchanged.
- Vulkan is intentionally disabled; do not add or enable Vulkan.
- Keep the Android native ABI target at `arm64-v8a` unless the user explicitly changes this requirement.

## Engineering safety

- Never infer execution semantics from architecture names alone.
- GGUF metadata, tensor names, shapes, types, and tokenizer metadata are evidence, not a complete executable model graph.
- Unknown or ambiguous structures must fail closed with actionable diagnostics.
- A model/backend/quantization combination is supported only after structural checks, operator coverage, numerical parity, and device execution validation.
- Do not mark a backend or kernel validated based only on successful compilation.
- Preserve existing behavior outside the GPU inference path.
- Run native contract tests and Android APK CI after meaningful changes; report pending, failed, and passed checks accurately.
