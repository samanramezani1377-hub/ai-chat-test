# 01 — Architecture

## هدف

معماری Local Model باید مستقل از UI باشد.

```text
App/UI
   ↓ contract
Model Management
   ↓
Domain
   ↓
Runtime Adapter
   ↓
GGUF Runtime Backend
```

## مرزبندی

- Domain: مدل‌ها، وضعیت‌ها و قراردادها.
- Model Management: Import، Inspect، Validate و Lifecycle.
- Runtime: Load، Generate و Unload.
- Storage: نگهداری پایدار مدل.
- Observability: Execution و Trace.
- UI Integration: تنها مرز اتصال UI به Backend.

## اصل استقلال

Composable، Screen و ViewModel نباید مسئول Parse، Validation، Persistence یا Runtime Loading باشند.

## ارجاعات

- [Model Management](02-model-management.md)
- [Model Domain](03-model-domain.md)
- [Runtime](04-runtime.md)
- [UI Integration](15-ui-integration.md)
