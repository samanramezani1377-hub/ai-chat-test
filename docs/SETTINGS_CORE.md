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
├── Performance / Visibility
└── Security / Confirmation
```

`SettingsRepository` مسئول Persistence، Versioning و Migration است. UI فقط تنظیمات را نمایش و تغییر می‌دهد و Core/Runtime از Contract مرکزی مصرف می‌کنند.

## Contract

هر Setting باید حداقل این مفهوم را داشته باشد:

```text
Setting {
  key
  domain
  type
  value
  defaultValue
  constraints?
  version
  source
}
```

`constraints` فقط محدودیت واقعی Runtime/Device/Platform یا Policy را بیان می‌کند؛ Default به‌تنهایی محدودیت نیست. تنظیمات User، System/Runtime و Policy باید از نظر Source قابل تشخیص باشند و precedence آن‌ها در Contract مشخص باشد.

## Versioning / Migration

Settings Schema باید Version داشته باشد. تغییر ساختار باید Migration مشخص داشته باشد و Migrationها idempotent باشند تا اجرای مجدد آن‌ها داده را خراب نکند.

```text
Stored Settings vN
       ↓
Migration Chain
       ↓
Current Settings Schema
       ↓
Central Settings Core
```

در انتقال به WooGit، مقصد باید بتواند Settings Version فعلی را بخواند یا از Migration رسمی استفاده کند؛ منطق Core نباید برای هر Storage یا UI جدید دوباره نوشته شود.

## Principles

- هر Setting مستقل و قابل تنظیم باشد.
- Default مقدار اولیه است، نه محدودیت اجباری.
- عدد ثابت غیرضروری در معماری تحمیل نشود.
- محدوده واقعی هر Setting می‌تواند توسط Runtime/Device تعیین شود.
- Visibility هر Metric مستقل از Measurement باشد.
- تغییر Setting نباید به کپی منطق در UIهای مختلف منجر شود.
- Settings Versioning برای Migration به WooGit حفظ شود.
- Storage implementation پشت Repository بماند.
- Setting Contract باید قابل تست و قابل Migration باشد.

### Migration

Settings باید Versioned و قابل Migration باشند تا با انتقال AI Core به WooGit، UI و Storage مقصد بتوانند همان Contract را مصرف کنند بدون اینکه منطق Core تغییر اساسی کند.
