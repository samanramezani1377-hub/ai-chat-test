# مستندات Backend مدل محلی

این پوشه مرجع مستندات قابلیت **Local AI Model Backend** است.

اصل معماری: تمام Backend مستقل از UI طراحی و کدنویسی می‌شود. قرارداد اتصال UI فقط در سند Integration نگهداری می‌شود.

## ساختار

```text
local-model/
├── README.md
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

## مدل هدف

**Qwen3-1.7B · GGUF · Q6_K**

## قواعد

- UI نباید وابستگی اجرایی به پیاده‌سازی Backend داشته باشد.
- هر تغییر Backend که روی UI اثر می‌گذارد باید در `15-ui-integration.md` ثبت شود.
- اسناد UI باید به Integration Contract ارجاع دهند.
- هیچ Mock یا Placeholder نباید موفقیت قابلیت واقعی را جعل کند.
- `Import File` منابع کاربر با `Import Model` یکی نیست و فعلاً در این قرارداد قرار ندارد.
