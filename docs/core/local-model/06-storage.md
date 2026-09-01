# 06 — Storage & File Lifecycle

```text
Storage
├── Selected Source
│   ├── URI
│   └── Access Permission
├── Managed Model File
├── Persistent Registry
├── Integrity Information
├── Recovery
├── Cleanup
├── Deletion Policy
└── Temporary URI Protection
```

مسیر موقت یا URI غیرقابل‌دسترسی نباید به‌عنوان مسیر دائمی مدل ثبت شود. چرخه استاندارد:

```text
Selected URI → Validated Source → Managed File → Registry → Runtime
```

ارجاع: [Model Management](02-model-management.md)، [Security](13-security.md)
