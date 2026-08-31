# Persistence & Storage

## Decision 25 — Hybrid Storage

معماری ذخیره‌سازی به‌صورت ترکیبی باشد:

```text
Persistent Storage
├── Conversations
├── Tasks
├── Workspace State
├── Actions
└── Verification Results

Retention / Temporary Storage
├── Debug Logs
├── Error Logs
└── Runtime Cache
```

داده‌های اصلی کاربر و وضعیت واقعی Task از Logs و Cache جدا هستند و Lifecycle مستقل دارند. حذف یا پاک‌سازی Log/Cache نباید باعث حذف Conversation، Task یا نتیجه Verification شود.

### Persistent Data

داده‌هایی که بخشی از وضعیت واقعی برنامه و کار کاربر هستند باید به‌صورت Structured و قابل بازیابی ذخیره شوند؛ از جمله Conversation/Message و Summary، Task/Goal/State، Workspace State، Action lifecycle/result و Verification result.

### Logs

Debug و Error Logs در Storage جدا یا با Lifecycle جدا نگهداری می‌شوند. در مرحله توسعه امکان ثبت کامل Trace و جزئیات خطا وجود دارد و Retention باید قابل تنظیم باشد؛ برای نمونه ۱ روز، ۷ روز، ۳۰ روز یا بدون حذف. تنظیم نهایی این مقادیر نباید در معماری به یک عدد ثابت وابسته باشد.

### Cache

Runtime Cache داده موقتی است و حذف آن نباید State اصلی برنامه را خراب کند. Cache می‌تواند در صورت نیاز مجدداً ساخته شود.

### Lifecycle

هر نوع داده باید Policy مستقل برای Create، Update، Read، Clear/Delete و Retention داشته باشد. پاک‌سازی Logs نباید Conversation یا Task را پاک کند.

### Principle

Storage implementation باید پشت یک Repository/Storage Contract قرار گیرد تا AI Core و Domain به نوع Database یا فایل‌سیستم خاص وابسته نشوند.
