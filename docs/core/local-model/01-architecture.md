# 01 — Architecture

## هدف

معماری Local Model باید مستقل از UI باشد و مرز مشخصی بین Domain، Management، Storage و Runtime داشته باشد.

```text
UI / App
   │
   │ UI Integration Contract
   ▼
Model Management
   │
   ├── Model Domain
   ├── Storage
   └── Operations
          │
          ▼
     Runtime Adapter
          │
          ▼
     GGUF Runtime
```

## مرزبندی

- Domain: مدل‌ها، metadata، وضعیت‌ها و قراردادهای دامنه.
- Model Management: Import، Inspect، Validate و Lifecycle.
- Storage: فایل مدیریت‌شده و Registry پایدار.
- Operations: هویت و چرخه عملیات.
- Runtime: Load، Generate و Unload.
- Resources/Security: محدودیت منابع و مرزهای امن ورودی.
- Observability: Trace و Diagnostics.
- UI Integration: تنها مرز رسمی اتصال UI.

## اصل استقلال

Composable، Screen و ViewModel نباید مسئول Parse، Validation، Persistence یا Runtime Loading باشند.

## ارجاعات

- [Documentation Tree](00-tree.md)
- [README](README.md)
- [Model Management](02-model-management.md)
- [Model Domain](03-model-domain.md)
- [Runtime](04-runtime.md)
- [Storage](06-storage.md)
- [Operations](07-operations.md)
- [UI Integration](15-ui-integration.md)
- [Definition of Done](16-definition-of-done.md)
