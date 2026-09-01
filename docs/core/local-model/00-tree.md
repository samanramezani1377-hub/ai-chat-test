# درخت مستندات Backend مدل محلی

```text
Local Model Backend Docs
│
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
    ↓
Runtime ──→ Resources / Security
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

## مرجع

`LOCAL_MODEL_BACKEND.md` قبلی به‌عنوان سند تجمیعی نگهداری می‌شود، اما مستندات خواندنی و قابل توسعه در این پوشه تفکیک شده‌اند.
