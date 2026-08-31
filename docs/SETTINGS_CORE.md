# Central Settings Core

## Decision 26 — Central Settings Core

تمام تنظیمات برنامه از یک **Central Settings Core** عبور می‌کنند، اما Domainهای تنظیمات مستقل می‌مانند.

```text
Central Settings Core
├── Model
├── Inference
├── Context
├── Agent
├── Workspace
├── Logs / Debug
└── Performance / Visibility
```

`SettingsRepository` مسئول Persistence، Versioning و Migration است. UI فقط تنظیمات را نمایش و تغییر می‌دهد و Core/Runtime از Contract مرکزی مصرف می‌کنند.

### Principles

- هر Setting مستقل و قابل تنظیم باشد.
- Default مقدار اولیه است، نه محدودیت اجباری.
- عدد ثابت غیرضروری در معماری تحمیل نشود.
- محدوده واقعی هر Setting می‌تواند توسط Runtime/Device تعیین شود.
- Visibility هر Metric مستقل از Measurement باشد.
- تغییر Setting نباید به کپی منطق در UIهای مختلف منجر شود.
- Settings Versioning برای Migration به WooGit حفظ شود.
- Storage implementation پشت Repository بماند.

### Migration

Settings باید Versioned و قابل Migration باشند تا با انتقال AI Core به WooGit، UI و Storage مقصد بتوانند همان Contract را مصرف کنند بدون اینکه منطق Core تغییر اساسی کند.
