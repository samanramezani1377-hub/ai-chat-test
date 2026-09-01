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
├── GGUF Backend
│   ├── initialization
│   ├── GGUF loading
│   ├── tokenizer
│   ├── context
│   ├── inference
│   ├── cancellation
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

Backend واقعی باید پشت Adapter قرار گیرد. UI مستقیماً Runtime Backend را صدا نمی‌زند.

## ارجاعات

- [Architecture](01-architecture.md)
- [Model Domain](03-model-domain.md)
- [Activation](05-activation.md)
- [Concurrency & Cancellation](08-concurrency.md)
- [Errors](09-errors.md)
- [Capabilities](11-capabilities.md)
- [Resources](12-resources.md)
- [Testing](14-testing.md)
- [UI Integration](15-ui-integration.md)
