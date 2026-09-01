# 16 — Definition of Done

```text
Backend Complete
├── Import واقعی
├── GGUF Inspection واقعی
├── Validation واقعی
├── Metadata واقعی
├── Persistent Repository
├── Lifecycle Manager
├── GGUF Runtime Backend
├── Q6_K Load when supported
├── Real Inference
├── Safe Activation / Deactivation
├── Cancellation
├── Concurrency Control
├── Resource / Memory Management
├── Typed Errors
├── Action Trace / Diagnostics
├── Real Tests
├── UI Integration Contract Complete
└── UI References Updated
```

## شرط نهایی

قابلیت فقط زمانی آماده ورود به UI است که Backend موارد بالا را واقعاً پیاده و تست کرده باشد و [UI Integration Contract](15-ui-integration.md) قرارداد نهایی اتصال را توصیف کند.

## مرجع اسناد

- [Documentation Tree](00-tree.md)
- [Architecture](01-architecture.md)
- [Model Management](02-model-management.md)
- [Model Domain](03-model-domain.md)
- [Runtime](04-runtime.md)
- [Activation](05-activation.md)
- [Storage](06-storage.md)
- [Operations](07-operations.md)
- [Concurrency](08-concurrency.md)
- [Errors](09-errors.md)
- [Observability](10-observability.md)
- [Capabilities](11-capabilities.md)
- [Resources](12-resources.md)
- [Security](13-security.md)
- [Testing](14-testing.md)
- [UI Integration](15-ui-integration.md)

هیچ Mock یا Placeholder نباید موفقیت Import، Load یا Inference را جعل کند.
