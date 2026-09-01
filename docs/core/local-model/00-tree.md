# 00 — Documentation Tree

```text
Local Model Backend Docs
│
├── README.md
├── 00-tree.md
├── 01-architecture.md
├── 02-model-management.md
├── 03-model-domain.md
├── 04-runtime.md
├── 05-activation.md
├── 06-storage.md
├── 07-operations.md
├── 08-concurrency.md
├── 09-errors.md
├── 10-observability.md
├── 11-capabilities.md
├── 12-resources.md
├── 13-security.md
├── 14-testing.md
├── 15-ui-integration.md
└── 16-definition-of-done.md
```

## وابستگی مفهومی

```text
Architecture
    ↓
Model Domain
    ↓
Model Management ──→ Storage
    ↓                    ↓
Validation          Persistence
    ↓                    ↓
Runtime ─────────→ Resources / Security
    ↓
Activation / Inference
    ↓
Operations / Cancellation
    ↓
Errors / Observability
    ↓
Testing
    ↓
UI Integration Contract
    ↓
UI
```

## ترتیب مطالعه

1. [Architecture](01-architecture.md)
2. [Model Management](02-model-management.md)
3. [Model Domain](03-model-domain.md)
4. [Runtime](04-runtime.md)
5. [Activation](05-activation.md)
6. [Storage](06-storage.md)
7. [Operations](07-operations.md)
8. [Concurrency & Cancellation](08-concurrency.md)
9. [Errors](09-errors.md)
10. [Observability](10-observability.md)
11. [Capabilities](11-capabilities.md)
12. [Resources](12-resources.md)
13. [Security](13-security.md)
14. [Testing](14-testing.md)
15. [UI Integration Contract](15-ui-integration.md)
16. [Definition of Done](16-definition-of-done.md)

## مرجع بالادستی

[Local Model Backend Index](../LOCAL_MODEL_BACKEND.md)

این فایل نقشه ساختار است؛ جزئیات هر موضوع فقط در سند همان موضوع نگهداری می‌شود.
