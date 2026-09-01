# 04 — Runtime

```text
Runtime
├── ModelRuntime Contract
│   ├── load(model)
│   ├── unload()
│   ├── generate(...)
│   ├── stopGeneration()
│   └── runtimeInfo()
├── RuntimeAdapter
│   └── Domain ↔ Runtime boundary
├── Android Runtime
│   └── LlamaCppAndroidRuntimeAdapter
│       └── dev.ffmpegkit-maintained:llama-android:0.1.1
├── GGUF Backend
│   ├── initialization
│   ├── GGUF loading
│   ├── tokenizer / chat template
│   ├── context
│   ├── inference
│   ├── cancellation boundary
│   ├── unload
│   ├── memory
│   └── cleanup
└── Runtime State
    ├── Unavailable
    ├── Idle
    ├── Loading
    ├── Ready
    ├── Generating
    ├── Unloading
    └── Failed
```

## Android implementation

The Android runtime is isolated in `:core:runtime-android`. The existing JVM `:core:runtime` remains the platform-neutral contract and model-management layer. The application wires the Android implementation at the composition root.

The selected AAR is a prebuilt llama.cpp Android library distributed through Maven Central. It supports GGUF, `arm64-v8a`, Android API 24+, CPU/NEON execution and 16 KB Android page support. citeturn4search1

The AAR currently exposes full completion rather than incremental token Flow streaming in its free artifact. Therefore the adapter emits one completed response chunk through the existing `onToken` callback and does **not** claim incremental streaming. A future streaming/interrupt-capable binding can replace only the Android adapter.

The upstream llama.cpp Android example demonstrates the same architectural boundary: GGUF metadata inspection from a `Uri`/private file, loading through an inference engine, and collecting generated tokens in Kotlin `Flow`. citeturn1search6

## Model compatibility

The runtime accepts GGUF models. Qwen3 is a supported architecture in llama.cpp, and upstream model schemas include `Q6_K` tensors. citeturn0search5turn4search3

## ارجاعات

- [Architecture](01-architecture.md)
- [Model Management](02-model-management.md)
- [Model Domain](03-model-domain.md)
- [Activation](05-activation.md)
- [Storage](06-storage.md)
- [Operations](07-operations.md)
- [Concurrency & Cancellation](08-concurrency.md)
- [Errors](09-errors.md)
- [Observability](10-observability.md)
- [Capabilities](11-capabilities.md)
- [Resources](12-resources.md)
- [Testing](14-testing.md)
- [UI Integration](15-ui-integration.md)
