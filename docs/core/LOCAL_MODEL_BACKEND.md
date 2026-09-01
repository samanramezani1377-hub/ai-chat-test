# Local AI Model Backend — Documentation Index

این فایل عمداً به‌عنوان سند بزرگ تجمیعی نگهداری نمی‌شود. مرجع اصلی و قابل توسعه Backend مدل محلی در پوشه `local-model/` قرار دارد.

## مرجع اصلی

- [درخت مستندات](local-model/00-tree.md)
- [README](local-model/README.md)
- [Architecture](local-model/01-architecture.md)
- [Model Management](local-model/02-model-management.md)
- [Model Domain](local-model/03-model-domain.md)
- [Runtime](local-model/04-runtime.md)
- [Activation](local-model/05-activation.md)
- [Storage](local-model/06-storage.md)
- [Operations](local-model/07-operations.md)
- [Concurrency & Cancellation](local-model/08-concurrency.md)
- [Errors](local-model/09-errors.md)
- [Observability / Action Trace](local-model/10-observability.md)
- [Capabilities](local-model/11-capabilities.md)
- [Resource Management](local-model/12-resources.md)
- [Security](local-model/13-security.md)
- [Testing](local-model/14-testing.md)
- [UI Integration Contract](local-model/15-ui-integration.md)
- [Definition of Done](local-model/16-definition-of-done.md)

## مدل هدف

**Qwen3-1.7B · GGUF · Q6_K**

## قانون مرجع واحد

محتوای فنی قابلیت Local AI Model باید فقط در اسناد `docs/core/local-model/` نگهداری شود. این فایل فقط Navigation/Index است تا دو نسخه متناقض از قرارداد ایجاد نشود.

هر تغییر Backend که روی UI اثر دارد باید در [UI Integration Contract](local-model/15-ui-integration.md) ثبت شود و سپس ارجاعات اسناد UI بررسی شوند.
