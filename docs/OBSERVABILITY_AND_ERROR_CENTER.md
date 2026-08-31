# Observability & Central Error Center

## Decision — Central Logging and User-Facing Error Center

هسته مرکزی پروژه باید یک **Central Logging & Error System** داشته باشد. همه خطاها و Eventهای قابل ثبت از مسیر مرکزی Logging عبور کنند و UI فقط مصرف‌کننده آن باشد.

هدف این است که یک خطا هم‌زمان دو سطح نمایش داشته باشد:

1. **خطای کاربرپسند فارسی** در همان جایی که خطا رخ داده است.
2. **خطای خام ماشینی و Trace/Log مرتبط** در یک بخش اختصاصی خطاها برای توسعه و ارسال به Agentها.

```text
Runtime / Model / Agent / Action / Storage / Network
                         ↓
              Central Logging Core
                    ┌────┴────┐
                    ↓         ↓
          User-facing Error   Raw Machine Error
             (Persian)          + Logs / Trace
                    ↓                 ↓
                 UI Error        Error Center
                                      ↓
                              Copy Error Logs
                                      ↓
                               User → Agent
```

## Error Center

اپ باید یک بخش مستقل با عنوانی مانند **خطاها / گزارش خطا** داشته باشد که خطاهای ثبت‌شده را قابل مشاهده کند.

برای هر خطا، در صورت وجود، موارد زیر قابل نمایش باشند:

- پیام فارسی و قابل فهم برای کاربر
- زمان رخداد
- Component / Source
- Error Code
- Severity
- وضعیت Task/Action مرتبط
- `eventId`
- `taskId`
- `actionId`
- Raw Machine Error
- Stack Trace یا Trace فنی مرتبط
- Logهای مرتبط قبل و بعد از خطا
- وضعیت Recovery / Verification

نمایش فنی می‌تواند در حالت جزئیات/بازشونده باشد تا رابط کاربر عادی شلوغ نشود.

## Copy for Agent

در Error Center یک دکمه مشخص برای **کپی گزارش خطا** وجود داشته باشد.

با زدن آن، یک Error Report قابل ارسال ساخته و در Clipboard قرار گیرد. گزارش باید شامل اطلاعات لازم برای تشخیص مشکل باشد، مانند:

```text
App / Build Version
Runtime / Model Info
Timestamp
Event ID
Task ID
Action ID (if applicable)
Component
Error Code
User-facing Error
Raw Machine Error
Relevant Trace / Logs
Execution / Verification State
```

گزارش باید تا حد ممکن برای Agent یا توسعه‌دهنده قابل استفاده باشد و لازم نباشد کاربر خطا را دستی بازنویسی کند.

## Privacy / Redaction

Raw Log به معنی نمایش بدون فیلتر Secretها نیست. قبل از ذخیره دائمی، نمایش یا Copy باید Secret، credential، token و داده حساس غیرضروری Redact شوند. Error Center باید بین **اطلاعات فنی لازم برای Debug** و **داده‌ای که نباید افشا شود** مرز داشته باشد.

## Central Ownership

Logging نباید در هر UI یا Feature به‌صورت جداگانه پیاده‌سازی شود. `Logging/Observability Core` مالک Contract و ساختار Event/Error باشد و UI فقط آن را Subscribe/Query کند.

هر Component می‌تواند Error تولید کند، اما ثبت استاندارد، Correlation، Persistence/Retention و ساخت Error Report توسط هسته مرکزی مدیریت شود.

## Development Mode

در فاز توسعه، تمام جزئیات فنی موردنیاز برای Debug ثبت شوند. نمایش کاربر همچنان می‌تواند خلاصه و فارسی باشد، اما Error Center باید امکان مشاهده Raw Machine Error و Trace کاملِ قابل‌اشتراک را فراهم کند.

این بخش با تصمیم قبلی درباره Debug/Error Logs و Event Trace یکپارچه است و نباید یک سیستم Logging موازی و جدا ایجاد شود.
