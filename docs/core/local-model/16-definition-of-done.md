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

قابلیت زمانی آماده ورود به UI است که موارد Backend بالا واقعاً پیاده و تست شده باشند و `15-ui-integration.md` دقیقاً قرارداد نهایی اتصال را توصیف کند.

هیچ Mock یا Placeholder نباید موفقیت Import، Load یا Inference را جعل کند.

## ارجاعات

- [Architecture](01-architecture.md)
- [Model Management](02-model-management.md)
- [Runtime](04-runtime.md)
- [UI Integration](15-ui-integration.md)
