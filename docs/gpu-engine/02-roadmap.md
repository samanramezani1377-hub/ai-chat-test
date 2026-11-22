# Android GPU Inference Engine — Roadmap

The goal is a reliable Android GPU inference engine that can grow across GPU vendors, model architectures and quantization formats. Generality is an engineering target, not a current support claim.

## Completed project cleanup

- [x] Remove the previous AI Chat application and unrelated modules from this branch.
- [x] Keep the standalone C++ GPU engine and Android backend code.
- [x] Remove llama.cpp, GGML, Rust tokenizer and legacy CPU runtime build dependencies.
- [x] Add a minimal model-import/activation/diagnostics test app.
- [x] Add a dedicated Android APK build to CI.
- [x] Keep Vulkan disabled.

## Engineering milestones

| Stage | Work | Required evidence |
|---|---|---|
| 1 | Stabilize standalone Android build | CI builds and packages the arm64 test APK reproducibly |
| 2 | Backend device qualification | Dawn/GLES discovery, limits, self-test and dispatch/readback logs on physical devices |
| 3 | Tokenizer correctness | Qwen3.5 BPE profile has pinned Hugging Face encode/decode parity in CI; remaining BPE/SentencePiece profiles still require independent reference fixtures |
| 4 | GGUF/model architecture coverage | Complete metadata and tensor mapping for each named architecture; unsupported layouts fail closed |
| 5 | Quantization correctness | Independent block decoder tests plus backend-specific numerical parity for each format |
| 6 | End-to-end inference | Real GGUF model activation, prefill, decode, KV cache and deterministic output tests on physical GPU |
| 7 | Memory and reliability | Large-model streaming, memory pressure, cancellation, device loss, repeated load/unload and soak tests |
| 8 | Performance | Profile upload, prefill, decode, synchronization and memory; optimize only after correctness gates pass |
| 9 | Device and model expansion | Maintain a tested compatibility matrix across vendors, drivers, architectures and quantizations |

## Immediate priorities

1. Get the new standalone Android build green and keep its CI artifact downloadable.
2. Fix any compile/link/runtime issues exposed by the new test app.
3. Use the test app to capture real-device backend and model activation logs.
4. Keep the Qwen3.5 profile's pinned reference-parity test green; add independent reference suites before validating any other tokenizer profile.
5. Expand model/quantization support only with independent reference tests and backend validation.

## Acceptance rules

- A CI pass is not a physical-device pass.
- A detected GPU is not a validated compute backend.
- A compiled kernel is not a validated kernel.
- A parsed GGUF file is not an activated model.
- A model load success is not proof of correct generation.
- No support statement may claim all GPUs, all models or all quantizations without a corresponding test matrix.
- Unsupported combinations must fail closed with a useful diagnostic.
