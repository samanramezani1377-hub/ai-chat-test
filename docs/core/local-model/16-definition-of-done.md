# 16 — Definition of Done

## Implementation status

🟡 **Backend implementation started** — domain, GGUF inspection, managed import, persistent registry and lifecycle orchestration are now being implemented independently from UI.

```text
Backend Complete
├── Domain Contract                    [implemented]
├── GGUF Inspection                    [implemented]
├── Real Import to Managed Storage     [implemented]
├── Persistent Model Registry          [implemented]
├── Validation                         [foundation present; runtime validation pending]
├── Lifecycle Manager                  [foundation implemented]
├── GGUF Runtime Backend               [pending concrete native/runtime adapter]
├── Q6_K Load                          [pending concrete runtime support]
├── Real Inference                     [pending concrete runtime support]
├── Safe Activation / Deactivation     [foundation implemented]
├── Cancellation                       [pending propagation through all layers]
├── Concurrency Control                [foundation implemented]
├── Resource / Memory Management       [pending runtime integration]
├── Typed Errors                       [implemented]
├── Action Trace / Diagnostics          [pending wiring]
├── Real Tests                         [pending]
├── UI Integration Contract             [documented]
└── UI References Updated               [pending final API wiring]
```

## شرط نهایی

قابلیت زمانی آماده ورود به UI است که موارد Backend بالا واقعاً پیاده و تست شده باشند و `15-ui-integration.md` دقیقاً قرارداد نهایی اتصال را توصیف کند.

هیچ Mock یا Placeholder نباید موفقیت Import، Load یا Inference را جعل کند.

## ارجاعات

- [Architecture](01-architecture.md)
- [Model Management](02-model-management.md)
- [Runtime](04-runtime.md)
- [UI Integration](15-ui-integration.md)
