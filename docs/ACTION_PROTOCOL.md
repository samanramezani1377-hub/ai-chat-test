# Action Protocol

## هدف

تعریف قرارداد استاندارد بین Agent و Action System تا مدل نتواند با متن آزاد، اجرای Action را جعل یا مبهم کند.

## ActionRequest

```json
{
  "version": 1,
  "actionId": "unique-id",
  "action": "create_file",
  "arguments": {}
}
```

### قواعد

- `version` برای تکامل قرارداد.
- `actionId` برای Trace و اتصال Request به Execution/Verification.
- `action` نام شناخته‌شده Action.
- `arguments` پارامترهای ساختاریافته.
- JSON نامعتبر یا Action ناشناخته باید قبل از Execution رد شود.

## ToolResult

```json
{
  "version": 1,
  "actionId": "unique-id",
  "success": true,
  "verified": true,
  "data": {},
  "error": null
}
```

`success` نتیجه Execution و `verified` نتیجه مستقل Verification است؛ هیچ‌کدام نباید صرفاً از متن مدل استخراج شوند.

## Action Schema

هر Action باید تعریف کند:

- نام
- Version
- Required arguments
- Optional arguments
- Type و constraints
- Read-only یا State-changing
- Risk level
- Permission موردنیاز
- نیاز به Confirmation
- قابلیت Reversible/Undo در صورت وجود
- Executor
- Verifier
- Errorهای ممکن

## Error Schema

```json
{
  "code": "VALIDATION_ERROR",
  "message": "Human readable message",
  "details": {}
}
```

کدهای پایه شامل `PARSER_ERROR`, `VALIDATION_ERROR`, `PERMISSION_DENIED`, `CONFIRMATION_REJECTED`, `EXECUTION_ERROR`, `VERIFICATION_ERROR`, `TIMEOUT`, `CANCELLED`, `STEP_LIMIT_REACHED` و `RUNTIME_ERROR` هستند.

## Action Lifecycle

```text
REQUESTED
  ↓
VALIDATED
  ↓
AUTHORIZED
  ↓
WAITING_FOR_CONFIRMATION (if required)
  ↓
EXECUTING
  ↓
VERIFYING
  ↓
COMPLETED / FAILED
```

## اصول

1. Agent درخواست می‌کند؛ Executor اجرا می‌کند.
2. Validator قبل از Execution ورودی را بررسی می‌کند.
3. Permission و Confirmation دو مفهوم جدا هستند.
4. Verifier مستقل از ادعای مدل است.
5. Action ID در تمام Traceها حفظ می‌شود.
6. Result باید machine-readable باشد و UI می‌تواند آن را به نمایش مناسب تبدیل کند.
7. افزودن Action جدید نباید Contractهای موجود را بشکند.
