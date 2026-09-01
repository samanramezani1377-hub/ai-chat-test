# مستندات Backend مدل محلی

این پوشه مرجع مستندات قابلیت **Local AI Model Backend** است. مستندات به اسناد کوچک و موضوعی تقسیم شده‌اند تا خوانایی و نگهداری بهتر شود.

## ساختار

- [درخت مستندات](00-tree.md)
- [Architecture](01-architecture.md)
- [Model Management](02-model-management.md)
- [Model Domain](03-model-domain.md)
- [Runtime](04-runtime.md)
- [Activation](05-activation.md)
- [Storage](06-storage.md)
- [Operations](07-operations.md)
- [Concurrency & Cancellation](08-concurrency.md)
- [Errors](09-errors.md)
- [Observability / Action Trace](10-observability.md)
- [Capabilities](11-capabilities.md)
- [Resource Management](12-resources.md)
- [Security](13-security.md)
- [Testing](14-testing.md)
- [UI Integration Contract](15-ui-integration.md)
- [Definition of Done](16-definition-of-done.md)

## مدل هدف

**Qwen3-1.7B · GGUF · Q6_K**

## قوانین مرجع

1. Backend مدل محلی مستقل از UI طراحی و کدنویسی می‌شود.
2. `15-ui-integration.md` تنها مرز رسمی قرارداد Backend ↔ UI است.
3. هر تغییر Backend که روی UI اثر می‌گذارد باید در Integration Contract ثبت شود.
4. سپس ارجاعات اسناد UI باید بررسی و در صورت نیاز به‌روزرسانی شوند.
5. هیچ Mock یا Placeholder نباید موفقیت واقعی Import، Load یا Inference را جعل کند.
6. `Import File` عمومیِ کاربر با `Import Model` یکی نیست و فعلاً جزو این قرارداد نیست.

## مرجع بالادستی

[Local Model Backend Index](../LOCAL_MODEL_BACKEND.md)
