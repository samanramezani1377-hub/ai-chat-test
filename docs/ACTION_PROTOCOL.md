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
AI_PREPARED
  ↓
PREVIEW_READY
  ↓
WAITING_FOR_FINAL_APPROVAL (sensitive actions)
  ↓
EXECUTING
  ↓
VERIFYING
  ↓
COMPLETED / FAILED
```

### Final Approval — sensitive actions

برای Actionهای حساس، تأیید کاربر به معنی «اجازه دادن به AI برای انجام عملیات» نیست. AI ابتدا باید نتیجه و Action قابل اجرا را آماده کند و Preview در Workspace قرار گیرد. سپس کاربر همان Snapshot آماده‌شده را بررسی و در صورت رضایت با `Final Approval` اجازه اجرای واقعی را می‌دهد.

پس از Final Approval، Executor اجرا را انجام می‌دهد؛ اما Approval هرگز به معنی موفقیت نیست. موفقیت فقط پس از Execution و Verification مستقل اعلام می‌شود.

### Approved Snapshot

برای Actionهای حساس، Approval باید به Snapshot مشخصی متصل باشد؛ حداقل شامل `actionId` و داده‌های مؤثر بر عملیات. اگر State واقعی قبل از Execution با Snapshot تأییدشده تغییر کرده باشد، سیستم نباید کورکورانه اجرا کند و باید Action را دوباره Validate/Prepare کرده و در صورت نیاز تأیید نهایی جدید بگیرد.

نمونه:

```text
Preview Snapshot:
Product #124
Price: 5,000,000 → 5,500,000

Current State before execution:
Price: 5,700,000

Result:
STALE_APPROVAL / REQUIRES_REVIEW
```

## Workspace interaction

Workspace تعاملی است اما Executor نیست. دکمه‌هایی مانند Preview، Approve، Reject، Retry، Cancel و Undo فقط Command/Intent تولید می‌کنند و آن را به AI Core می‌دهند. Core مسیر Permission، Validation، Confirmation، Execution و Verification را کنترل می‌کند.

## اصول

1. Agent درخواست می‌کند؛ Executor اجرا می‌کند.
2. Validator قبل از Execution ورودی را بررسی می‌کند.
3. Permission و Confirmation دو مفهوم جدا هستند.
4. Final Approval فقط برای Snapshot آماده‌شده و قابل اجرا صادر می‌شود.
5. Verifier مستقل از ادعای مدل است.
6. Action ID در تمام Traceها حفظ می‌شود.
7. Result باید machine-readable باشد و UI می‌تواند آن را به نمایش مناسب تبدیل کند.
8. افزودن Action جدید نباید Contractهای موجود را بشکند.
9. Approval نباید Success تلقی شود.
