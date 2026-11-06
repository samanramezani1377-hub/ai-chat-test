# Data and Privacy Boundary

## هدف

تعریف مرز داده برای دو مسیر Inference قابل انتخاب: مدل محلی روی دستگاه و Provider راه‌دور؛ مشخص‌کردن اینکه چه داده‌ای داخل دستگاه می‌ماند و چه زمانی داده از طریق شبکه ارسال می‌شود.

## Local و Provider: مرز داده

```text
DEVICE
├── Model Files
├── Chat / Context
├── Workspace State
├── Action Logs
├── Error Reports
├── Performance Metrics
└── Tool Results
        ↓
      AI Core
```

مسیر انتخاب‌شده تعیین می‌کند داده چگونه پردازش شود:

- **Local:** متن گفتگو و ورودی‌های Inference برای تولید پاسخ به Provider راه‌دور ارسال نمی‌شوند؛ پردازش روی دستگاه انجام می‌شود، مگر قابلیت جداگانه‌ای با اجازه کاربر شبکه را درگیر کند.
- **Provider:** متن/Context لازم برای درخواست Inference به Provider انتخاب‌شده ارسال می‌شود. UI باید این وضعیت را شفاف کند و فقط داده لازم برای همان درخواست را بفرستد.

استفاده از Provider اختیاری و وابسته به انتخاب کاربر است. برنامه نباید هنگام خطای Local به‌صورت پنهانی به Provider برود یا هنگام خطای Provider بدون اطلاع کاربر به Local تغییر کند. Credentialها باید امن نگهداری و از Log/Error Report حذف یا Redact شوند.

## Network

Network Monitoring باید مصرف واقعی Network را ثبت کند. وجود Network capability به‌تنهایی به معنی ارسال Chat/Model Data نیست.

هر قابلیت Network، از جمله Inference از طریق Provider، باید صریحاً مشخص کند:

- چه داده‌ای ارسال می‌شود.
- به کجا ارسال می‌شود.
- چرا لازم است.
- آیا قابل خاموش‌کردن است.

## Storage

موارد زیر باید Lifecycle مشخص داشته باشند:

- Imported Model metadata
- Conversation History
- Performance History
- Workspace data
- Action/Verification logs
- Error Logs / Error Reports
- Exported reports

برای هر مورد باید retention و delete behavior مشخص شود.

## Logging و Error Center

تمام Error/Eventهای فنی از Central Logging/Observability عبور می‌کنند. کاربر می‌تواند خطای فارسی و قابل فهم را در محل رخداد ببیند و جزئیات Raw Machine Error/Trace را در Error Center مشاهده کند. Error Center همچنین امکان Copy یک Error Report برای ارسال به Agent/Developer را فراهم می‌کند.

Debug Logs و Error Reports نباید به‌صورت پیش‌فرض Secrets، credentialها، tokenها یا داده حساس غیرضروری را ذخیره یا Copy کنند. قبل از نمایش/Copy، اطلاعات حساس باید Redact شوند.

## User control

کاربر باید بتواند داده‌های محلی مربوط به History/Logs/Error Reports را در محدوده قابلیت‌های واقعی اپ مدیریت یا پاک کند.

## Future WooGit

در انتقال به WooGit، Data Boundary باید حفظ شود و AI Core نباید به‌صورت ضمنی داده فروشگاه را به یک سرویس Cloud ارسال کند. هر Network-backed operation باید Contract و Permission مستقل داشته باشد.
