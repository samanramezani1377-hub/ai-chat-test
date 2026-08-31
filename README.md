# AI Chat Test

> Prototype / Technical Feasibility Test برای Local AI روی Android و Agent قابل‌اعتماد

این repository محیط آزمایشی برای بررسی اجرای Local AI روی Android، Chat فارسی، Streaming، Agent، Action، Verification و AI Workspace است تا معماری و قابلیت‌ها پیش از انتقال به WooGit اعتبارسنجی شوند.

## نقشه مستندات

| سند | موضوع |
|---|---|
| [`PROTOTYPE_SPEC.md`](docs/PROTOTYPE_SPEC.md) | Specification اصلی Prototype |
| [`DECISIONS_21_30.md`](docs/DECISIONS_21_30.md) | تصمیم‌های نهایی ۲۱ تا ۳۰ |
| [`ARCHITECTURE.md`](docs/ARCHITECTURE.md) | معماری Core/UI و Adapterها |
| [`ACTION_PROTOCOL.md`](docs/ACTION_PROTOCOL.md) | قرارداد Action و Tool Result |
| [`ACTION_REGISTRY.md`](docs/ACTION_REGISTRY.md) | Registry دسته‌بندی‌شده Actionها |
| [`ACTION_EXECUTION_VERIFICATION.md`](docs/ACTION_EXECUTION_VERIFICATION.md) | Executor، Verifier و شواهد اجرا |
| [`TASK_STATE_MACHINE.md`](docs/TASK_STATE_MACHINE.md) | State Machine Task/Agent/Execution |
| [`RECOVERY_FAILURE.md`](docs/RECOVERY_FAILURE.md) | Checkpoint، Recovery، Idempotency و Verification |
| [`OBSERVABILITY_AND_ERROR_CENTER.md`](docs/OBSERVABILITY_AND_ERROR_CENTER.md) | Central Logging، Error Center و گزارش قابل ارسال به Agent |
| [`SECURITY_MODEL.md`](docs/SECURITY_MODEL.md) | Permission، Confirmation و Risk |
| [`DATA_AND_PRIVACY.md`](docs/DATA_AND_PRIVACY.md) | Data Boundary، Logging و Privacy |
| [`AI_WORKSPACE.md`](docs/AI_WORKSPACE.md) | میز کار تعاملی AI |
| [`PERFORMANCE_METRICS.md`](docs/PERFORMANCE_METRICS.md) | Performance و Network Monitoring |
| [`WOOGIT_INTEGRATION.md`](docs/WOOGIT_INTEGRATION.md) | WooGit Adapter + Capability Contract |
| [`PLATFORM_RELEASE_CI.md`](docs/PLATFORM_RELEASE_CI.md) | Platform، Compatibility و CI |
| [`TEST_MATRIX.md`](docs/TEST_MATRIX.md) | ماتریس تست و Benchmark |
| [`IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) | برنامه پیاده‌سازی و مهاجرت |
| [`QUESTION_BANK.md`](docs/QUESTION_BANK.md) | سؤالات و تصمیم‌های ثبت‌شده |

## معماری اصلی

اصل کلیدی: **AI Core کاملاً مستقل از UI و Runtime است.**

```text
UI
├── Chat
├── Settings
├── Error Center
├── Technical Error Panel (hideable)
└── AI Workspace
        ↓
     AI Core
     ├── Model / Inference
     ├── Context / Conversation
     ├── Agent
     ├── Action Protocol / Registry
     └── Central Observability / Logging
        ↓
   Action System
   ├── Permission
   ├── Validation
   ├── Capability Check
   ├── Adapter / Executor
   └── Verifier
```

Runtime، Storage و WooGit نیز از طریق Contract/Adapter از Core جدا می‌مانند.

## Context

Context ترکیبی است:

```text
System Context
+ Persistent Task Context
+ Conversation Summary
+ Recent Messages (user-configurable)
+ Workspace Context (query/select)
```

هیچ عدد ثابت غیرضروری نباید به‌عنوان محدودیت معماری فرض شود.

## Action و تأیید

Actionها در Registry مرکزی دسته‌بندی می‌شوند و هر Category قابلیت اضافه‌کردن Executor/Adapterهای بیشتر را دارد. Capability Check پیش از Execution مشخص می‌کند محیط فعلی قابلیت لازم را دارد.

مسیر مرجع:

```text
Action Registry → Action Contract → Capability Check → Adapter / Executor → Verifier
```

برای Action حساس:

```text
Prepare → Preview / Snapshot → Validate → Final Approval → Execute → Verify → Result
```

تأیید کاربر اجازه اجرای همان عملیات آماده‌شده را می‌دهد؛ AI بعد از تأیید نباید دوباره تصمیم‌گیری کند. Approval به معنی Success نیست و Success فقط پس از Verification اعلام می‌شود.

## AI Workspace

Workspace یک میز کار تعاملی برای Task، Goal، Action، Result، Verification، Preview، Before/After، فایل و Artifact است. Workspace Executor نیست و Interactionها را به Core برمی‌گرداند.

## Central Logging & Error Center

تمام Error/Eventهای فنی باید از **Central Logging/Observability Core** عبور کنند.

هر خطا دو سطح دارد:

```text
خطای فارسی و قابل فهم در محل رخداد
                 +
Raw Machine Error / Trace / Logs در Error Center
```

در **Error Center / بخش خطاها** کاربر باید بتواند خطاهای ثبت‌شده، Raw Machine Error، Trace، `eventId`، `taskId`، `actionId` و وضعیت Execution/Recovery/Verification را ببیند.

همچنین یک دکمه **Copy Error Report** وجود دارد تا گزارش استاندارد شامل اطلاعات فنی لازم در Clipboard قرار گیرد و کاربر بتواند آن را برای Agent یا Developer ارسال کند.

پنل نمایش کامل خطاهای فنی و Raw Machine Error یک بخش مستقل و **قابل مخفی‌سازی بدون حذف** است. در آینده می‌توان آن را با تغییر Visibility/Feature Flag از UI مخفی کرد، در حالی که Logging Core، Error Store و Error Report همچنان فعال می‌مانند و پنل در Developer/Debug Mode دوباره قابل نمایش است.

Secret، credential، token و داده حساس غیرضروری باید قبل از نمایش یا Copy Redact شوند.

جزئیات: [`OBSERVABILITY_AND_ERROR_CENTER.md`](docs/OBSERVABILITY_AND_ERROR_CENTER.md)

## Recovery

Recovery بر پایه:

```text
State Machine + Checkpoint + Recovery Policy + Idempotency + Verification
```

Crash یا Restart نباید باعث اجرای دوباره کورکورانه Action شود. در وضعیت نامشخص، ابتدا State واقعی Verify می‌شود.

## WooGit Migration

مهاجرت به WooGit با `Adapter + Capability Contract` انجام می‌شود. AI Core مستقیماً به WooGit وابسته نیست و Action Contractها بین Prototype و WooGit قابل استفاده مجدد هستند.

```text
AI Core → Action Registry → Capability Check → WooGit Adapter → WooGit API/Service → Verifier
```

## CI و Release

CI پایه از ابتدا فعال است: Build، Unit Test، Integration Test، Architecture Check، Static Analysis/Lint و Validation.

تست‌های سخت‌گیرانه Action/Recovery از نظر زیرساخت آماده می‌شوند، اما تا رسیدن پروژه به آخرین سطح آمادگی و تأیید کیفیت اپ توسط مالک پروژه، Hard Gate اجباری نیستند.

## Baseline

Baseline فعلی Prototype می‌تواند Qwen3-1.7B در GGUF/Q4_K_M با llama.cpp باشد؛ این انتخاب نهایی محصول نیست و Runtime/Model باید قابل تعویض بماند.

**مرجع اصلی تصمیم‌های فنی:** [`PROTOTYPE_SPEC.md`](docs/PROTOTYPE_SPEC.md)
