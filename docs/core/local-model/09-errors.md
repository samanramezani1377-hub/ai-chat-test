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

UI پیام کاربرپسند را دریافت می‌کند؛ علت فنی، Error Code و Diagnostics Reference برای عیب‌یابی حفظ می‌شوند.

## ارجاعات

- [Model Management](02-model-management.md)
- [Runtime](04-runtime.md)
- [Operations](07-operations.md)
- [Observability](10-observability.md)
- [Resources](12-resources.md)
- [Testing](14-testing.md)
- [UI Integration](15-ui-integration.md)
