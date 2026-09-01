# 08 — Concurrency & Cancellation

```text
Concurrency
├── Operation Serialization / Policy
├── Duplicate Import Protection
├── Duplicate Load Protection
├── Cancellation Propagation
├── Cancellation Cleanup
├── Race Prevention
└── Safe Active Model Transition
```

Import و Load نباید همزمان به State متناقض برسند. Cancellation باید واقعی باشد و بعد از لغو، منابع و فایل‌های موقت پاک‌سازی شوند.

## ارجاعات

- [Operations](07-operations.md)
- [Model Management](02-model-management.md)
- [Activation](05-activation.md)
- [Storage](06-storage.md)
- [Runtime](04-runtime.md)
- [Errors](09-errors.md)
- [Resources](12-resources.md)
- [Testing](14-testing.md)
- [UI Integration](15-ui-integration.md)
