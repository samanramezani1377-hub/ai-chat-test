# Data and Privacy Boundary

## هدف

تعریف مرز داده در Local AI و مشخص‌کردن اینکه چه داده‌ای داخل دستگاه می‌ماند و چه زمانی Network ممکن است درگیر شود.

## Local-first boundary

```text
DEVICE
├── Model Files
├── Chat / Context
├── Workspace State
├── Action Logs
├── Performance Metrics
└── Tool Results
        ↓
      AI Core
```

اصل پایه: Inference و داده‌های موردنیاز آن باید Local باشند و وابستگی اجباری به Cloud وجود نداشته باشد.

## Network

Network Monitoring باید مصرف واقعی Network را ثبت کند. وجود Network capability به‌تنهایی به معنی ارسال Chat/Model Data نیست.

هر قابلیت Network آینده باید صریحاً مشخص کند:

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
- Exported reports

برای هر مورد باید retention و delete behavior مشخص شود.

## Logging

Logهای Debug نباید به‌صورت پیش‌فرض Secrets، credentialها یا داده حساس غیرضروری را ذخیره کنند.

## User control

کاربر باید بتواند داده‌های محلی مربوط به History/Logs را در محدوده قابلیت‌های واقعی اپ پاک کند.

## Future WooGit

در انتقال به WooGit، Data Boundary باید حفظ شود و AI Core نباید به‌صورت ضمنی داده فروشگاه را به یک سرویس Cloud ارسال کند. هر Network-backed operation باید Contract و Permission مستقل داشته باشد.
