# 03 — Model Domain

```text
Model Domain
├── ModelDescriptor
│   ├── id
│   ├── displayName
│   ├── source / managed path
│   ├── format
│   ├── quantization
│   ├── architecture
│   ├── size
│   ├── metadata
│   ├── validation status
│   └── runtime compatibility
├── ModelMetadata
├── ModelFormat
├── Quantization
├── Architecture
├── ModelState
└── RuntimeCompatibility
```

تمام فیلدها باید از داده واقعی پر شوند. `ModelDescriptor` قرارداد داده‌ای بین Model Management، Storage، Runtime و UI Integration است و نباید وابسته به UI باشد.

## ارجاعات

- [Documentation Tree](00-tree.md)
- [Architecture](01-architecture.md)
- [Model Management](02-model-management.md)
- [Runtime](04-runtime.md)
- [Storage](06-storage.md)
- [UI Integration](15-ui-integration.md)
