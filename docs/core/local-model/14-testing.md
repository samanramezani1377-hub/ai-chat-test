# 14 — Backend Testing

```text
Testing
├── Import
│   ├── valid GGUF
│   ├── non-GGUF
│   ├── corrupt file
│   ├── duplicate model
│   └── cancellation
├── Inspection
├── Validation
│   ├── metadata
│   ├── architecture
│   ├── quantization
│   └── resources
├── Storage
├── Lifecycle
│   ├── load
│   ├── unload
│   ├── activate
│   └── replacement
├── Inference
│   ├── generate
│   └── stop generation
├── Errors
├── Runtime Unavailable
├── Resource / Memory Failure
└── Trace / Error Propagation
```

تست‌های قابلیت واقعی نباید موفقیت Runtime را با Mock جعل کنند. Mock فقط برای Boundaryهای لازم unit test مجاز است.

## ارجاعات

- [Model Management](02-model-management.md)
- [Runtime](04-runtime.md)
- [Activation](05-activation.md)
- [Storage](06-storage.md)
- [Operations](07-operations.md)
- [Concurrency & Cancellation](08-concurrency.md)
- [Errors](09-errors.md)
- [Observability](10-observability.md)
- [Capabilities](11-capabilities.md)
- [Resources](12-resources.md)
- [Security](13-security.md)
- [UI Integration](15-ui-integration.md)
- [Definition of Done](16-definition-of-done.md)
