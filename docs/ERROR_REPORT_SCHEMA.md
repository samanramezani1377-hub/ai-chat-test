# Error Report Schema

## Purpose

این سند Contract استاندارد `ErrorReport` را تعریف می‌کند تا Error Center، Central Logging، Storage و Agent/Developer از یک ساختار واحد استفاده کنند.

## Canonical Schema

```text
ErrorReport {
  reportId
  appVersion
  buildId?
  timestamp
  severity
  component
  errorCode
  eventId
  taskId?
  actionId?
  userMessageFa
  rawMachineError
  trace?
  relatedLogs?
  executionState?
  recoveryState?
  verificationState?
  runtimeInfo?
  modelInfo?
  redactionStatus
}
```

## Rules

- `reportId` یکتا و مخصوص همان گزارش است.
- `eventId` شناسه Event اصلی خطاست.
- `taskId` و `actionId` در صورت ارتباط با Task/Action ثبت می‌شوند.
- `userMessageFa` پیام قابل فهم فارسی برای کاربر است.
- `rawMachineError` متن/کد خام لازم برای Debug است.
- `trace` و `relatedLogs` فقط تا حد لازم برای تشخیص مشکل اضافه می‌شوند.
- `redactionStatus` باید نشان دهد گزارش پیش از نمایش/Copy از Redaction عبور کرده است.
- Error Report نباید Secret، credential یا token خام داشته باشد.

## Lifecycle

```text
Error/Event
   ↓
Central Logging
   ↓
Correlation
   ↓
Redaction
   ↓
ErrorReport
   ├── Storage
   ├── Error Center
   └── Copy → Agent / Developer
```

## Copy Contract

دکمه Copy باید همین Contract را به یک متن قابل خواندن و ارسال تبدیل کند. UI نباید از اطلاعات پراکنده خودش گزارش متفاوتی بسازد.

## Compatibility

Version آینده این Schema باید backward-compatible یا دارای Migration مشخص باشد تا گزارش‌های قبلی قابل خواندن و تحلیل باقی بمانند.
