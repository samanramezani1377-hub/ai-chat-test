# 12 — Resource Management

```text
Resources
├── Model Size
├── Memory Requirements
├── Runtime Resources
├── Load Limits
├── Resource Reservation
├── Cleanup
└── OutOfMemory Handling
```

قبل از Load باید منابع موردنیاز تا حد امکان از اطلاعات واقعی مدل و Runtime مشخص شود. خطای کمبود منابع باید به Error Contract و Diagnostics متصل شود.

## CPU و GPU

CPU یک مسیر اجرای معتبر برای مدل محلی است؛ استفاده از CPU یا انتقال اجرا از GPU به CPU (CPU fallback) به‌خودی‌خود خطا محسوب نمی‌شود. GPU یک شتاب‌دهنده اختیاری است و فعال‌بودن آن به پشتیبانی واقعی دستگاه، Driver، Build، عملیات مدل و منابع آزاد وابسته است.

در صورت در دسترس نبودن یا شکست GPU، Runtime می‌تواند در صورت امکان مدل را روی CPU اجرا کند. Diagnostics باید Backend واقعی، وضعیت GPU، تعداد لایه‌ها/عملیات واقعاً Offloadشده و علت fallback را جداگانه گزارش کند. موفقیت CPU fallback باید به‌عنوان اجرای موفق محلی گزارش شود، نه شکست صرفاً به دلیل عدم استفاده از GPU.

این سیاست فقط مربوط به انتخاب Backend درون اجرای Local است و مجوز جابه‌جایی بی‌اجازه بین منبع Local و Provider راه‌دور نیست.

## ارجاعات

- [Model Domain](03-model-domain.md)
- [Runtime](04-runtime.md)
- [Operations](07-operations.md)
- [Concurrency & Cancellation](08-concurrency.md)
- [Errors](09-errors.md)
- [Testing](14-testing.md)
- [UI Integration](15-ui-integration.md)
