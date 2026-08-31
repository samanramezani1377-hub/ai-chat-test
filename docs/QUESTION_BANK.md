# AI Chat Test — Question Bank

این فایل فقط سؤالات تصمیم‌گیری Prototype را نگهداری می‌کند. تصمیم‌های نهایی پس از پاسخ‌گویی باید در سند تخصصی مربوط به خود ثبت شوند و Question Bank فقط وضعیت و متن سؤال را نگه می‌دارد.

## وضعیت سؤالات قبلی
- [x] 1–20. سؤالات اولیه Prototype

## سؤالات جدید — Contract و Implementation

### 21. قرارداد Model و Runtime
**تصمیم نهایی:** AI Core کاملاً مستقل از Runtime باشد و Runtime Adapter این Contract را پیاده‌سازی کند. Core مستقیماً به `llama.cpp` وابسته نیست و Runtime از طریق Adapter/Contract قابل تعویض است.
- [x] 21. Model / Runtime Contract

### 22. Conversation و Context
**تصمیم نهایی:** سیستم Context ترکیبی باشد: System Context، Persistent Task Context، Conversation Summary، Recent Messages و Workspace Context. تعداد Recent Messages توسط کاربر قابل تنظیم باشد و عدد ثابت در معماری فرض نشود. پیام‌های قدیمی‌تر به Summary تبدیل شوند و Workspace به‌صورت Query/Select قابل استفاده باشد تا Prompt بی‌نهایت رشد نکند.
- [x] 22. Conversation / Context Contract

### 23. Workspace Data Contract
**تصمیم نهایی:** Workspace طبق گزینه B تعاملی باشد، اما Executor نباشد. Interactionها Command/Intent به Core می‌فرستند و اجرای واقعی از Core/Action System عبور می‌کند. Action حساس: Prepare/Preview/Snapshot/Validate/Final Approval/Execute/Verify. Approval فقط برای همان Snapshot است و Approval به معنی Success نیست.
- [x] 23. Workspace Data Contract

### 24. Event و Trace
**تصمیم نهایی:** گزینه B. Workspace فقط Eventهای مهم را نمایش دهد، اما در مرحله توسعه تمام Eventهای داخلی و جزئیات خطا در Debug/Error Logs ثبت شوند. Eventها با `eventId`، `taskId` و در صورت نیاز `actionId` قابل Trace باشند.
- [x] 24. Event / Trace Schema

### 25. Persistence و ذخیره‌سازی
**تصمیم نهایی:** گزینه C. داده‌های اصلی و وضعیت واقعی در Persistent Storage و Debug/Error Logs و Runtime Cache با Lifecycle و Retention جدا باشند. پاک‌سازی Log/Cache نباید Conversation، Task یا Verification را حذف یا خراب کند. Storage پشت Repository/Storage Contract باشد.
- [x] 25. Persistence Schema

### 26. Configuration
**تصمیم نهایی:** گزینه C. یک **Central Settings Core** هسته مرکزی مدیریت تنظیمات باشد. تنظیمات Domain-specific مستقل باشند، اما همه از Settings Core مدیریت شوند. `SettingsRepository` مسئول Storage، Versioning و Migration باشد. Defaultها متمرکز باشند و Default مقدار اولیه است، نه محدودیت اجباری. ساختار Settings برای Migration به WooGit نیز Versioned و قابل تبدیل باشد.
- [x] 26. Configuration Schema

### 27. Action Registry
**تصمیم نهایی:** گزینه C. معماری `Registry + Adapter Architecture` باشد. `ActionRegistry` مرجع مرکزی تعریف و کشف Actionهاست و Executor نیست. Actionها داخل Registry دسته‌بندی شوند و هر Category بتواند در آینده Executor/Adapterهای بیشتری داشته باشد. Category، Action، Executor و Adapter جدید بدون بازنویسی Core/UI قابل اضافه‌شدن باشند.
- [x] 27. Action Registry

### 28. WooGit Integration Contract
**تصمیم نهایی:** گزینه C. اتصال به WooGit به‌صورت `Adapter + Capability Contract` باشد. AI Core مستقیماً به WooGit وابسته نباشد و فقط Action/Capability Contract را بشناسد. WooGit Adapter قابلیت‌ها را اعلام و Actionهای عمومی را به عملیات واقعی نگاشت کند. Adapterهای آینده نیز بدون تغییر Core قابل اضافه‌شدن باشند.
- [x] 28. WooGit Integration Contract

### 29. Recovery و Failure Handling
**تصمیم نهایی:** گزینه C. Recovery بر پایه **State Machine + Checkpoint + Recovery Policy + Idempotency + Verification** باشد. Stateهای مهم Task و Action Persistent باشند؛ هر Action Policy مخصوص Retry/Timeout/Recovery/Verification داشته باشد؛ عملیات چندمرحله‌ای مرحله‌به‌مرحله ثبت شوند؛ و قبل از Retry در وضعیت نامشخص، State واقعی Verification شود. Recovery نباید Final Approval را دور بزند و Crash نباید باعث اجرای دوباره کورکورانه شود.
- [x] 29. Recovery / Failure Strategy

### 30. Platform / Release / CI Acceptance
**تصمیم نهایی:** Platform، ABI و محدودیت‌های Runtime/Model بر اساس نیاز و قابلیت واقعی پروژه تعیین شوند و عدد ثابت غیرضروری از ابتدا تحمیل نشود. یک Compatibility Contract مرکزی وجود داشته باشد. CI پایه از همین حالا شامل Build، Unit Tests، Integration Tests، Architecture Checks، Static Analysis/Lint و Package/Validation باشد. تست‌های سخت‌گیرانه Action و Recovery از نظر طراحی و زیرساخت آماده باشند، اما فعلاً Release Gate اجباری نباشند. پس از رسیدن پروژه به آخرین سطح آمادگی و تأیید کیفیت خود اپ توسط مالک پروژه، این تست‌ها به Hard Gateهای CI/Release تبدیل شوند. Definition of Done شامل Implementation، Build، تست‌های مرتبط، CI، Architecture Compliance، Documentation، Compatibility و در موارد مرتبط Migration Readiness باشد.
- [x] 30. Platform / Release / CI Acceptance

## روش ادامه
تمام سؤالات ۲۱ تا ۳۰ پاسخ داده و تصمیم‌گیری شده‌اند. تصمیم‌های نهایی در اسناد تخصصی مربوط به خود نیز ثبت شده‌اند.
