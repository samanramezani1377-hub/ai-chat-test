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

تمام فیلدها باید از داده واقعی پر شوند.

ModelDescriptor مرز داده‌ای بین Model Management و Runtime است و نباید وابسته به UI باشد.

ارجاع: [Model Management](02-model-management.md)، [Runtime](04-runtime.md)
