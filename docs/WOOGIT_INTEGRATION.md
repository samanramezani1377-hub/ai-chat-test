# WooGit Integration Contract

## Decision 28 — Adapter + Capability Contract

برای مهاجرت نهایی به WooGit، اتصال AI Core به WooGit به‌صورت **Adapter + Capability Contract** انجام شود.

AI Core نباید مستقیماً به API، Service یا implementation داخلی WooGit وابسته باشد. Core فقط Action Contract و Capability Contract را می‌شناسد و Adapter محیط اجرا مسئول نگاشت آن‌ها به عملیات واقعی است.

```text
AI Core
  ↓
Action Contract
  ↓
Action Registry
  ↓
Capability Contract
  ↓
WooGit Adapter
  ↓
WooGit API / Service
  ↓
WooCommerce
```

## Capabilities

هر Adapter باید بتواند قابلیت‌های قابل پشتیبانی خود را اعلام کند. Core قبل از اجرای Action می‌تواند بررسی کند که محیط فعلی آن Capability را پشتیبانی می‌کند یا خیر.

نمونه:

```text
supports("change_product_price") → true
supports("rewrite_product_description") → true
supports("delete_order") → false
```

Capabilityها باید قابل توسعه باشند و به یک فهرست ثابت و غیرقابل تغییر در Core تبدیل نشوند.

## Adapterها

Prototype می‌تواند Adapter آزمایشی داشته باشد و WooGit Adapter بعداً بدون تغییر Action Contract جایگزین یا اضافه شود.

```text
                 AI Core
                    │
             Action Contract
                    │
              Action Registry
                    │
        ┌───────────┼───────────┐
        ↓           ↓           ↓
   Test Adapter  WooGit       Future Adapter
```

## Execution Lifecycle

برای Actionهای واقعی، به‌خصوص Actionهای حساس، Adapter فقط در مرحله Execution واقعی وارد عمل می‌شود و Workflow کامل باید وضعیت هر مرحله را مشخص کند:

```text
Prepare
  ↓
Preview / Snapshot
  ↓
Validate
  ↓
Final Approval
  ↓
Execute
  ↓
Verify
  ↓
Result
```

`Final Approval` اجازه اجرای همان عملیات آماده و Snapshot‌شده است؛ تأیید نباید باعث شود AI دوباره تصمیم بگیرد چه عملیاتی انجام دهد. Approval به معنی Success نیست و Success فقط بعد از Execute و Verify اعلام می‌شود.

## Migration Principle

هدف این Contract این است که انتقال از Prototype به WooGit عمدتاً با جایگزینی/افزودن Adapter و Capability Mapping انجام شود و نیاز به تغییر در AI Core، UI و Action Contract حداقل باشد.

Action عمومی مانند `change_product_price` در Core مستقل می‌ماند و WooGit Adapter آن را به Operation واقعی WooGit نگاشت می‌کند.

## Extensibility

ساختار باید اجازه دهد در آینده محیط‌های دیگری نیز اضافه شوند؛ برای مثال Test، WooGit یا Adapterهای آینده. همچنین Capability و Action جدید باید بدون بازنویسی Core قابل اضافه‌شدن باشند.
