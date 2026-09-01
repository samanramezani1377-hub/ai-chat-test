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

ارجاع: [Errors](09-errors.md)، [Runtime](04-runtime.md)، [Observability](10-observability.md)
