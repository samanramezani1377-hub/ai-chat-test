# 09 — Error Model

```text
Error
├── Typed Error
├── Error Code
├── User-facing Message
├── Technical Cause
└── Diagnostics Reference

Error Types
├── FileAccessError
├── UnsupportedFormat
├── InvalidModel
├── InvalidMetadata
├── UnsupportedArchitecture
├── UnsupportedQuantization
├── RuntimeUnavailable
├── LoadFailed
├── OutOfMemory
├── ImportCancelled
├── StorageError
└── InferenceError
```

UI باید پیام کاربرپسند را دریافت کند؛ علت فنی و شناسه Diagnostics برای عیب‌یابی باقی می‌ماند.

ارجاع: [Observability](10-observability.md)، [UI Integration](15-ui-integration.md)
