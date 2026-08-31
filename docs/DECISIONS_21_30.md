# Decisions 21–30 — Consolidated Architecture Decisions

این سند خلاصه یکدست تصمیم‌های نهایی سؤال‌های ۲۱ تا ۳۰ است. سندهای تخصصی هر تصمیم همچنان مرجع جزئیات اجرایی هستند.

| سؤال | تصمیم | سند تخصصی |
|---|---|---|
| 21 | AI Core مستقل از Runtime + Runtime Adapter Contract | Architecture / Runtime Contract |
| 22 | Context ترکیبی با Recent Messages قابل تنظیم، Summary، Persistent Task Context و Workspace Context | Architecture / Context |
| 23 | Workspace تعاملی ولی غیر Executor؛ Sensitive Action با Preview/Snapshot و Final Approval | AI Workspace / Action Protocol |
| 24 | نمایش Eventهای مهم در Workspace و ثبت کامل Event/Error در توسعه | Observability |
| 25 | Persistent Storage ترکیبی + Lifecycle جدا برای Log/Cache | Persistence |
| 26 | Central Settings Core + Domain-specific settings + Versioning/Migration | Settings / Architecture |
| 27 | Action Registry + Adapter؛ دسته‌بندی داخلی و Executorهای قابل توسعه زیر هر Category | Action Registry |
| 28 | Adapter + Capability Contract برای اتصال به WooGit | WooGit Integration |
| 29 | State Machine + Checkpoint + Recovery Policy + Idempotency + Verification | Recovery & Failure |
| 30 | Platform/Compatibility بر اساس نیاز واقعی؛ CI پایه از ابتدا؛ Hard Gate سخت Action/Recovery در آمادگی نهایی پس از تأیید کیفیت اپ | Platform / Release / CI |

## اصول مشترک

- هیچ عدد ثابت و غیرضروری نباید به‌عنوان محدودیت معماری فرض شود؛ مقادیر کاربر قابل تنظیم‌اند و Runtime می‌تواند Safety Limit مستقل داشته باشد.
- AI Core از UI، Runtime، Storage و محیط مقصد مستقل است.
- Workspace فقط نمایش‌دهنده و دریافت‌کننده Interaction است و Executor نیست.
- Actionهای حساس ابتدا Prepare/Preview/Snapshot/Validate می‌شوند؛ سپس Final Approval همان Snapshot را مجاز می‌کند و بعد Execute و Verify انجام می‌شود.
- Approval به معنی Success نیست و Success فقط پس از Verification اعلام می‌شود.
- Actionها از طریق Registry و Contractهای machine-readable مدیریت می‌شوند.
- Adapter محیط اجرا را از Core جدا می‌کند و Capabilityها مشخص می‌کنند محیط فعلی چه Actionهایی را پشتیبانی می‌کند.
- Recovery باید State واقعی را حفظ کند و از اجرای دوباره ناخواسته جلوگیری کند.
- CI از ابتدا باید مسیر Release نهایی را آماده کند، اما تست‌های سخت‌گیرانه Action/Recovery تا آخرین سطح آمادگی و تأیید کیفیت اپ توسط مالک پروژه Hard Gate نمی‌شوند.
- تمام تصمیم‌ها باید در مستندات تخصصی خود منعکس شوند و README فقط نقشه و نقطه ورود مستندات باشد.
