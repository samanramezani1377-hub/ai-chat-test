# Task State Machine

## هدف

تعریف Stateهای رسمی برای Inference، Agent و Action تا UI و Core State متناقض نسازند.

## Task states

```text
IDLE
 ↓
PLANNING
 ↓
WAITING_FOR_CONFIRMATION
 ↓
EXECUTING
 ↓
VERIFYING
 ↓
COMPLETED
```

شاخه‌های پایانی:

```text
FAILED
CANCELLED
BLOCKED
TIMEOUT
STEP_LIMIT_REACHED
```

همه Stateها برای همه Taskها الزاماً استفاده نمی‌شوند؛ State Machine باید با نوع عملیات سازگار باشد.

## Cancellation

- `STOP_GENERATION` فقط Generation را متوقف می‌کند اگر Action شروع نشده باشد.
- `CANCEL_TASK` Task را متوقف می‌کند.
- Actionهای درحال اجرا فقط اگر Executor قابلیت cancellation داشته باشد قابل توقف هستند.
- اگر Action غیرقابل‌لغو باشد، UI باید وضعیت واقعی را نمایش دهد و موفقیت جعلی ایجاد نکند.

## Transitions

هر Transition باید علت/رویداد قابل Trace داشته باشد.

```text
PLANNING → BLOCKED
reason: permission_denied

EXECUTING → FAILED
reason: execution_error

VERIFYING → FAILED
reason: verification_error
```

## Step limit

Maximum Agent Steps یک تنظیم کاربر است و نباید به یک عدد ثابت در Specification تبدیل شود. هر Retry/Attempt نیز Step مصرف می‌کند. Runtime می‌تواند Hard Safety Limit مستقل داشته باشد.

## UI mapping

UI باید State رسمی را نمایش دهد و State جدید را از روی متن مدل حدس نزند.
