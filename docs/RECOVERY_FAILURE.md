# Recovery & Failure Handling

## Decision 29 — State Machine + Checkpoint + Recovery Policy + Idempotency + Verification

سیستم Recovery بر پایه State Machine و Checkpoint باشد و در کنار آن برای هر Action، Recovery Policy، Retry Policy، Idempotency و Verification مستقل وجود داشته باشد.

```text
PENDING → PREPARING → READY_FOR_APPROVAL → APPROVED → EXECUTING → VERIFYING → COMPLETED
```

در صورت خطا:

```text
EXECUTING → FAILED → RECOVERABLE / NON_RECOVERABLE
```

## Checkpoint

Stateهای مهم Task و Action باید Persistent باشند تا Crash، Process Kill یا Restart باعث از دست رفتن وضعیت واقعی نشود. بعد از Restart سیستم باید State را بازیابی کند و قبل از ادامه، وضعیت واقعی را بررسی کند؛ نباید صرفاً بر اساس آخرین State فرضی دوباره Execute کند.

## Idempotency

هر Action قابل اجرا باید تا حد ممکن دارای `idempotencyKey` یا مکانیزم معادل باشد تا Retry ناخواسته باعث اجرای دوباره همان عملیات نشود. در عملیات چندمرحله‌ای، موفقیت هر مرحله مستقل ثبت شود و مراحل موفق در Recovery دوباره اجرا نشوند مگر اینکه Verification نشان دهد State واقعی نیازمند اجرای مجدد است.

## Recovery Policy

هر Action می‌تواند Policy مخصوص خود را برای Retry، Timeout، Recovery و Verification داشته باشد. Recovery نباید یک رفتار ثابت و یکسان برای همه Actionها فرض کند.

پس از Timeout:

```text
Did action execute?
   ├── YES → Verify
   ├── NO  → Retry if policy allows
   └── UNKNOWN → Verify actual state before retry
```

## Sensitive Actions

Recovery نباید `Final Approval` را دور بزند. اگر Action حساس قبل از Execution متوقف شده باشد، Approval مربوط به همان Snapshot/Prepared Action حفظ می‌شود؛ اگر State واقعی تغییر کرده یا Snapshot دیگر معتبر نباشد، باید Validate/Prepare مجدد انجام شود و در صورت نیاز Approval جدید گرفته شود.

اگر Crash بعد از Execution رخ دهد، سیستم ابتدا Execution status و State واقعی را Verification می‌کند و سپس تصمیم Recovery می‌گیرد؛ هرگز صرفاً به خاطر Crash عملیات را دوباره اجرا نمی‌کند.

## Partial Execution

برای Batchها و عملیات چندمرحله‌ای باید مشخص باشد کدام بخش‌ها موفق، ناموفق، در حال اجرا یا نامشخص هستند. Result و Verification هر بخش باید Traceable به `taskId` و `actionId` باشد.

## Failure Classes

Failureها حداقل باید بتوانند به دسته‌های Runtime/Model، Resource/Memory، Network/External Service، Validation، Execution، Verification، User Cancellation، App Crash و Unknown Execution State تفکیک شوند. هر Failure باید در Error Log ثبت و در صورت امکان Recovery Classification داشته باشد.

## Principle

هدف Recovery این است که **State واقعی سیستم را حفظ و از دوباره‌کاری ناخواسته جلوگیری کند**، نه اینکه صرفاً بعد از خطا Action را دوباره اجرا کند.
