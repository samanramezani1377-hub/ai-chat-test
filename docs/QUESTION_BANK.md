# AI Chat Test — Question Bank

این فایل فقط سؤالات تصمیم‌گیری Prototype را نگهداری می‌کند. تصمیم‌های نهایی پس از پاسخ‌گویی باید در سند تخصصی مربوط به خود ثبت شوند و Question Bank فقط وضعیت و متن سؤال را نگه می‌دارد.

## وضعیت سؤالات قبلی

تمام سؤالات ۱ تا ۲۰ پاسخ داده و تصمیم‌گیری شده‌اند.

- [x] 1–20. سؤالات اولیه Prototype

## سؤالات جدید — Contract و Implementation

این ۱۰ سؤال برای تکمیل جزئیات اجرایی پروژه طراحی شده‌اند. فعلاً هیچ‌کدام پاسخ‌داده‌شده محسوب نمی‌شوند.

### 21. قرارداد Model و Runtime

قرارداد دقیق بین `AI Core` و `Runtime Adapter` چگونه باشد؟ چه اطلاعاتی باید از مدل دریافت و چه APIهایی برای Load، Unload، Generate، Streaming، Stop و تنظیمات Inference ارائه شود؟

**تصمیم موردنیاز:** ساختار Model/Runtime Contract و میزان استقلال Core از llama.cpp.

- [ ] 21. Model / Runtime Contract

### 22. Conversation و Context

تاریخچه Chat و Context چگونه مدیریت شود؟ چه چیزی وارد Context مدل شود، Context Window چگونه کنترل شود، و در زمان پرشدن Context چه رفتاری داشته باشیم (حذف، خلاصه‌سازی، شروع Context جدید یا ترکیب این روش‌ها)؟

**تصمیم موردنیاز:** Lifecycle کامل Conversation و Context.

- [ ] 22. Conversation / Context Contract

### 23. Workspace Data Contract

`AI Workspace` دقیقاً چه ساختار داده‌ای از Core دریافت کند؟ برای نمایش Chat، Action، Tool Result، Verification، Before/After، Preview و Confirmation چه نوع `WorkspaceItem` یا Eventهایی لازم است؟

**تصمیم موردنیاز:** Data Contract مستقل Workspace.

- [ ] 23. Workspace Data Contract

### 24. Event و Trace

یک اجرای Agent یا Action چگونه از ابتدا تا انتها Trace شود؟ چه `eventId`، `taskId`، `actionId`، timestamp، state و metadataهایی لازم است و چه چیزی باید در Debug Log یا Workspace قابل مشاهده باشد؟

**تصمیم موردنیاز:** Event/Trace Schema مشترک برای Chat، Agent، Action و Verification.

- [ ] 24. Event / Trace Schema

### 25. Persistence و ذخیره‌سازی

کدام داده‌ها باید ذخیره شوند و Lifecycle هرکدام چیست؟ برای Model Metadata، Chat History، Performance History، Workspace State، Action/Verification Logs و Exportها چه Storage و چه رفتار Delete/Clear در نظر گرفته شود؟

**تصمیم موردنیاز:** Persistence Schema و Data Lifecycle.

- [ ] 25. Persistence Schema

### 26. Configuration

تمام تنظیمات کاربر دقیقاً چه چیزهایی باشند و چگونه ذخیره و اعمال شوند؟ آیا تنظیمات Model، Inference، Agent، Visibility، Network/Debug و Workspace باید از هم مستقل باشند و تغییر هرکدام بدون تغییر بقیه ممکن باشد؟

**تصمیم موردنیاز:** Configuration Schema و مرز تنظیمات User/Runtime.

- [ ] 26. Configuration Schema

### 27. Action Registry

فهرست Actionها و Metadata آن‌ها چگونه نگهداری شود؟ آیا هر Action باید Schema، Version، Permission، Risk Level، Confirmation Policy، Executor و Verifier مخصوص خود را در یک Registry مرکزی داشته باشد؟ اضافه‌کردن Action جدید دقیقاً چگونه انجام شود؟

**تصمیم موردنیاز:** Action Registry و Extension Contract.

- [ ] 27. Action Registry

### 28. WooGit Integration Contract

AI Core هنگام انتقال به WooGit دقیقاً با چه Interface یا Contractی با Operationهای WooGit ارتباط برقرار کند؟ چگونه Actionهای عمومی Prototype به Operationهای واقعی WooGit مانند تغییر قیمت، تغییر توضیحات و مدیریت محصول نگاشت شوند؟

**تصمیم موردنیاز:** مرز AI Core و WooGit و روش Integration بدون وابستگی Core به UI.

- [ ] 28. WooGit Integration Contract

### 29. Recovery و Failure Handling

اگر وسط Inference، Agent، Action یا Verification یکی از این اتفاق‌ها رخ دهد چه شود: App بسته شود، Process کشته شود، Model Unload شود، RAM کافی نباشد، Runtime خطا بدهد، Action نصفه اجرا شود، Verification ناموفق باشد یا کاربر عملیات را متوقف کند؟

**تصمیم موردنیاز:** Recovery، Resume، Rollback/Compensation در صورت امکان و رفتار نهایی هر Failure.

- [ ] 29. Recovery / Failure Strategy

### 30. Platform / Release / CI Acceptance

حداقل نسخه Android، ABIهای پشتیبانی‌شده، Runtime/Model Compatibility و شرایط Release چه باشند؟ همچنین برای هر تغییر چه تست‌هایی باید در CI اجرا شوند و Definition of Done دقیقاً چه مواردی را شامل شود؟

**تصمیم موردنیاز:** Compatibility Matrix، Release Policy و CI/Acceptance Criteria.

- [ ] 30. Platform / Release / CI Acceptance

## روش ادامه

پس از پاسخ به سؤالات ۲۱ تا ۳۰، هر تصمیم به سند تخصصی مربوط به خودش منتقل می‌شود و سپس `QUESTION_BANK.md` فقط وضعیت پاسخ‌گویی را ثبت خواهد کرد.
