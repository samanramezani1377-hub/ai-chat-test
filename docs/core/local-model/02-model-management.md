# 02 — Model Management

```text
Model Management
├── Import
│   ├── Source URI
│   ├── Access
│   ├── Managed Storage
│   ├── Integrity
│   ├── Duplicate Detection
│   ├── Cancellation
│   └── Cleanup
├── Inspection
│   ├── Format
│   ├── GGUF Header
│   ├── Structure
│   ├── Metadata
│   ├── Architecture
│   ├── Quantization
│   └── Runtime Requirements
├── Validation
│   ├── File
│   ├── GGUF
│   ├── Metadata
│   ├── Architecture
│   ├── Quantization
│   ├── Runtime Compatibility
│   └── Resource Compatibility
├── Repository
│   ├── Register
│   ├── Get
│   ├── List
│   ├── Active Model
│   └── Unregister
└── Lifecycle Manager
    ├── NotImported
    ├── Importing
    ├── Validating
    ├── Imported
    ├── Ready
    ├── Loading
    ├── Active
    ├── Unloading
    ├── Invalid
    └── Failed
```

نام فایل نباید مبنای تشخیص مدل یا Quantization باشد؛ metadata واقعی فایل مرجع است.

مدل هدف اولیه: **Qwen3-1.7B · GGUF · Q6_K**.

ارجاع: [Domain](03-model-domain.md)، [Storage](06-storage.md)، [UI Integration](15-ui-integration.md)
