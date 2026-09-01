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

ارجاع: [Operations](07-operations.md)، [Activation](05-activation.md)، [Resources](12-resources.md)
