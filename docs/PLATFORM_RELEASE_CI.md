# Platform, Release & CI Acceptance

## Decision 30

### Platform

حداقل نسخه Android، ABI و محدودیت‌های Platform باید بر اساس نیاز واقعی پروژه و قابلیت‌های واقعی Runtime/Model تعیین شوند و هیچ عدد ثابت و غیرضروری از ابتدا به‌عنوان محدودیت معماری فرض نشود.

ABIهای پشتیبانی‌شده نیز بر اساس Runtime/Model و قابلیت تست واقعی انتخاب شوند.

### Runtime / Model Compatibility

یک **Compatibility Contract مرکزی** وجود داشته باشد تا پیش از Load/Execution سازگاری Model، Runtime، ABI و منابع دستگاه بررسی شود و در صورت ناسازگاری دلیل قابل تشخیص ارائه شود.

```text
Model
  ↓
Runtime Compatibility
  ↓
ABI Compatibility
  ↓
Device Resources
  ↓
Compatible / Incompatible
```

### CI

تمام کنترل‌های پایه از همین حالا بخشی از CI باشند:

```text
Build
  ↓
Unit Tests
  ↓
Integration Tests
  ↓
Architecture Checks
  ↓
Static Analysis / Lint
  ↓
Package / Validation
```

تست‌های مرتبط با Action و Recovery نیز از نظر طراحی و زیرساخت آماده باشند، اما در فاز فعلی تست‌های سخت‌گیرانه آن‌ها **Release Gate اجباری نباشند**.

### Hard Gates — Final Readiness

وقتی پروژه به آخرین سطح آمادگی رسید و کیفیت خود اپ توسط مالک پروژه تأیید شد، تست‌های سخت‌گیرانه Action و Recovery به Hard Gateهای CI/Release تبدیل شوند.

فعال‌شدن این Gateها باید یک مرحله صریح از Readiness باشد و نباید صرفاً به خاطر عبور Build یا سبز بودن تست‌های پایه فعال تلقی شود.

### Definition of Done

یک قابلیت زمانی Done محسوب شود که معیارهای مرتبط آن تکمیل شده باشند، از جمله:

- Implementation
- Build
- Relevant Tests
- CI Validation
- Architecture Compliance
- Documentation
- Compatibility Validation
- Migration Readiness در موارد مرتبط

شدت و اجباری‌بودن تست‌های Action/Recovery باید مطابق فاز پروژه باشد: در توسعه فعلی زیرساخت آن‌ها آماده باشد و در مرحله نهایی، پس از تأیید کیفیت اپ، به Hard Gate تبدیل شوند.

## Principle

هدف این تصمیم این است که پروژه از همان ابتدا مسیر Release نهایی و CI کامل را داشته باشد، بدون اینکه در فاز توسعه با تست‌های سنگین و سخت‌گیرانه Action/Recovery بی‌دلیل قفل شود. هیچ‌کدام از الزامات نهایی حذف نمی‌شوند؛ فقط زمان فعال‌شدن آن‌ها به مرحله آمادگی نهایی منتقل می‌شود.
