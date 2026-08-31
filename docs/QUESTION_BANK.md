# AI Chat Test — Question Bank

این فایل فقط سؤالات تصمیم‌گیری Prototype را نگهداری می‌کند. تصمیم‌های نهایی پس از پاسخ‌گویی باید در سند تخصصی مربوط به خود ثبت شوند و Question Bank فقط وضعیت و متن سؤال را نگه می‌دارد.

## وضعیت سؤالات قبلی

تمام سؤالات ۱ تا ۲۰ پاسخ داده و تصمیم‌گیری شده‌اند.

- [x] 1–20. سؤالات اولیه Prototype

## سؤالات جدید — Contract و Implementation

### 21. قرارداد Model و Runtime

قرارداد دقیق بین `AI Core` و `Runtime Adapter` چگونه باشد؟ چه اطلاعاتی باید از مدل دریافت و چه APIهایی برای Load، Unload، Generate، Streaming، Stop و تنظیمات Inference ارائه شود؟

**تصمیم نهایی:** AI Core کاملاً مستقل از Runtime باشد و Runtime Adapter این Contract را پیاده‌سازی کند. بنابراین Core مستقیماً به `llama.cpp` وابسته نیست و Runtime از طریق Adapter/Contract قابل تعویض است.

- [x] 21. Model / Runtime Contract

### 22. Conversation و Context

تاریخچه Chat و Context چگونه مدیریت شود؟ چه چیزی وارد Context مدل شود، Context Window چگونه کنترل شود، و در زمان پرشدن Context چه رفتاری داشته باشیم (حذف، خلاصه‌سازی، شروع Context جدید یا ترکیب این روش‌ها)؟

**تصمیم نهایی:** سیستم Context ترکیبی باشد و از چهار بخش اصلی تشکیل شود: System Context، Persistent Task Context، Conversation Summary، Recent Messages و Workspace Context. تعداد Recent Messages مقدار پیش‌فرض دارد اما باید توسط کاربر قابل تنظیم باشد و عدد ثابتی در معماری فرض نشود. پیام‌های قدیمی‌تر به Summary تبدیل می‌شوند. Persistent Task Context هدف، کار فعلی، Intent کاربر و اطلاعات مهم را مستقل از تاریخچه نگه می‌دارد. Workspace نیز به‌عنوان Context قابل Query/Select در اختیار Core/Agent قرار می‌گیرد و لازم نیست کل داده Workspace کورکورانه داخل Prompt قرار گیرد. Context هر درخواست از Source of Truth ساخته می‌شود تا Prompt به‌صورت بی‌نهایت رشد نکند.

- [x] 22. Conversation / Context Contract

### 23. Workspace Data Contract

`AI Workspace` دقیقاً چه ساختار داده‌ای از Core دریافت کند؟ برای نمایش Chat، Action، Tool Result، Verification، Before/After، Preview و Confirmation چه نوع `WorkspaceItem` یا Eventهایی لازم است؟

**تصمیم نهایی:** Workspace طبق گزینه B تعاملی باشد، اما Executor نباشد. Workspace می‌تواند وضعیت Task، Goal، Current Subject، Actionها، Resultها، Verification، Preview، Before/After، Confirmation، Error، File و Artifact را نمایش دهد و Interactionهایی مانند Approve، Reject، Retry، Cancel و Undo را دریافت کند؛ اما این Interactionها فقط Command/Intent به AI Core می‌فرستند و اجرای واقعی همیشه از مسیر Core و Action System انجام می‌شود. برای Actionهای حساس، Workflow شامل آماده‌سازی توسط AI، Preview/Snapshot، انتظار برای `Final Approval` و سپس Execution واقعی است. تأیید نهایی قبل از Execution واقعی انجام می‌شود، نه به معنی اجازه‌ای مبهم برای اینکه AI بعداً تصمیم بگیرد چه کاری انجام دهد. Approval به معنی Success نیست و موفقیت فقط پس از Execution و Verification مستقل اعلام می‌شود. Approval باید به Snapshot مشخص متصل باشد و اگر State واقعی قبل از Execution تغییر کرده باشد، اجرای کورکورانه ممنوع و نیازمند Validate/Prepare و در صورت لزوم تأیید نهایی جدید است.

- [x] 23. Workspace Data Contract

### 24. Event و Trace

یک اجرای Agent یا Action چگونه از ابتدا تا انتها Trace شود؟ چه `eventId`، `taskId`، `actionId`، timestamp، state و metadataهایی لازم است و چه چیزی باید در Debug Log یا Workspace قابل مشاهده باشد؟

**تصمیم نهایی:** گزینه B. در Workspace عادی فقط Eventهای مهم و قابل فهم نمایش داده شوند تا محیط کار شلوغ نشود. در عین حال در مرحله توسعه، تمام Eventهای داخلی و جزئی و همچنین جزئیات کامل خطاها در Debug/Error Logs ثبت شوند. هر Event باید قابل ردیابی با `eventId`، `taskId` و در صورت مرتبط بودن `actionId` باشد و لاگ خطا حداقل component، error code، message، timestamp و metadata فنی مرتبط را نگه دارد. بنابراین نمایش کاربر ساده است، اما هیچ اطلاعات لازم برای توسعه و عیب‌یابی از دست نمی‌رود.

- [x] 24. Event / Trace Schema

### 25. Persistence و ذخیره‌سازی

کدام داده‌ها باید ذخیره شوند و Lifecycle هرکدام چیست؟ برای Model Metadata، Chat History، Performance History، Workspace State، Action/Verification Logs و Exportها چه Storage و چه رفتار Delete/Clear در نظر گرفته شود؟

**تصمیم نهایی:** گزینه C. Storage به‌صورت ترکیبی باشد: داده‌های اصلی و وضعیت واقعی برنامه در Persistent Storage نگهداری شوند (Conversation، Task، Workspace State، Action و Verification)، در حالی که Debug/Error Logs و Runtime Cache Lifecycle و Retention جدا داشته باشند. پاک‌سازی Log یا Cache نباید Conversation، Task یا Verification را حذف یا خراب کند. Retention لاگ‌ها باید قابل تنظیم باشد و معماری نباید به یک عدد ثابت وابسته شود. Storage implementation نیز پشت Repository/Storage Contract قرار گیرد تا Core به Database یا فایل‌سیستم خاص وابسته نباشد.

- [x] 25. Persistence Schema

### 26. Configuration

تمام تنظیمات کاربر دقیقاً چه چیزهایی باشند و چگونه ذخیره و اعمال شوند؟ آیا تنظیمات Model، Inference، Agent، Visibility، Network/Debug و Workspace باید از هم مستقل باشند و تغییر هرکدام بدون تغییر بقیه ممکن باشد؟

**تصمیم نهایی:** گزینه C. یک **Central Settings Core** هسته مرکزی مدیریت تنظیمات باشد. تنظیمات از نظر Domain به بخش‌های مستقل مانند Model، Inference، Context، Agent، Workspace، Logs/Debug و Performance تقسیم شوند؛ اما همه از طریق Settings Core مدیریت شوند. `SettingsRepository` مسئول Storage، Versioning و Migration باشد. هر تنظیم مستقل قابل تغییر باشد و Defaultها در Schema/Settings Provider متمرکز تعریف شوند، نه به‌صورت پراکنده در منطق برنامه. Default مقدار اولیه است و نباید به محدودیت اجباری تبدیل شود؛ مقادیر کاربر باید تا محدوده‌ای که Runtime/Device واقعاً پشتیبانی می‌کند قابل تنظیم باشند. این ساختار باید Migration به WooGit را نیز آسان کند؛ داده‌های Settings نسخه‌بندی و قابل تبدیل باشند و Core به UI یا Storage خاص وابسته نباشد.

- [x] 26. Configuration Schema

### 27. Action Registry

فهرست Actionها و Metadata آن‌ها چگونه نگهداری شود؟ آیا هر Action باید Schema، Version، Permission، Risk Level، Confirmation Policy، Executor و Verifier مخصوص خود را در یک Registry مرکزی داشته باشد؟ اضافه‌کردن Action جدید دقیقاً چگونه انجام شود؟

**تصمیم نهایی:** گزینه C. معماری Actionها به‌صورت `Registry + Adapter Architecture` باشد. `ActionRegistry` مرجع مرکزی تعریف و کشف Actionهاست و Executor نیست. Actionها داخل خود Registry به‌صورت دسته‌بندی‌شده سازمان‌دهی شوند و هر Category بتواند در آینده Executor/Adapterهای بیشتری داشته باشد. بنابراین می‌توان Category جدید، Action جدید، Executor جدید یا Adapter محیط اجرا اضافه کرد بدون اینکه AI Core یا UI بازنویسی شوند. هر Action Contract و Metadata مستقل خود را دارد و اجرای واقعی از مسیر Adapter/Executor و سپس Verifier انجام می‌شود. این ساختار برای انتقال به WooGit نیز مناسب است، چون Action Contract می‌تواند ثابت بماند و Adapter/Executor محیط جدید اضافه یا جایگزین شود.

- [x] 27. Action Registry

### 28. WooGit Integration Contract

AI Core هنگام انتقال به WooGit دقیقاً با چه Interface یا Contractی با Operationهای WooGit ارتباط برقرار کند؟ چگونه Actionهای عمومی Prototype به Operationهای واقعی WooGit مانند تغییر قیمت، تغییر توضیحات و مدیریت محصول نگاشت شوند؟

- [ ] 28. WooGit Integration Contract

### 29. Recovery و Failure Handling

اگر وسط Inference، Agent، Action یا Verification یکی از این اتفاق‌ها رخ دهد چه شود: App بسته شود، Process کشته شود، Model Unload شود، RAM کافی نباشد، Runtime خطا بدهد، Action نصفه اجرا شود، Verification ناموفق باشد یا کاربر عملیات را متوقف کند؟

- [ ] 29. Recovery / Failure Strategy

### 30. Platform / Release / CI Acceptance

حداقل نسخه Android، ABIهای پشتیبانی‌شده، Runtime/Model Compatibility و شرایط Release چه باشند؟ همچنین برای هر تغییر چه تست‌هایی باید در CI اجرا شوند و Definition of Done دقیقاً چه مواردی را شامل شود؟

- [ ] 30. Platform / Release / CI Acceptance

## روش ادامه

پس از پاسخ به سؤالات ۲۱ تا ۳۰، هر تصمیم به سند تخصصی مربوط به خودش منتقل می‌شود و سپس `QUESTION_BANK.md` فقط وضعیت پاسخ‌گویی را ثبت خواهد کرد.
