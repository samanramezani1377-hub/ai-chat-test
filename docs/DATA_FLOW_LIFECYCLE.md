# Data Flow & Lifecycle

## User request

```text
User Message
 ↓
UI
 ↓
AI Core
 ↓
Context Builder
 ↓
ModelRuntime Contract
 ↓
Agent Decision
 ↓
Action Registry
 ↓
Action Contract
 ↓
Capability Check
 ↓
Prepare / Validate
 ↓
Preview
 ↓
Final Approval (if required)
 ↓
Adapter / Executor
 ↓
Verifier
 ↓
Result
 ↓
Central Observability
 ↓
UI / Workspace
```

## Important rule

Approval فقط اجازه اجرای operation آماده‌شده را می‌دهد. بعد از Approval، AI نباید دوباره operation را طراحی یا انتخاب کند.

## Failure path

هر مرحله می‌تواند Error تولید کند. Error به Central Observability می‌رود و هم‌زمان UI پیام فارسی مناسب دریافت می‌کند. Error Center می‌تواند ErrorReport کامل را نمایش و Copy کند.

## Persistence

Stateهای لازم برای Resume، Recovery و Verification باید از طریق Persistence Contract ثبت شوند. Restart نباید باعث اجرای کورکورانه Action شود.
