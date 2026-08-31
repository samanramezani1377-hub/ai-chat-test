# AI Chat Test — Question Bank

این فایل مخزن ۲۰ سؤال تصمیم‌گیری برای Prototype است. سؤالات باید **یکی‌یکی** مطرح شوند. پس از پاسخ کاربر، پاسخ در همین سند ثبت و سؤال مربوطه با `[x]` علامت‌گذاری می‌شود و سپس سؤال بعدی مطرح خواهد شد.

هدف این پرسش‌ها مشخص‌کردن نیازمندی‌های واقعی Prototype قبل از توسعه است.

## وضعیت

- [x] 01. هدف اصلی Prototype چیست؟
- [x] 02. مدل پایه دقیقاً کدام است؟
- [x] 03. Quantization اولیه کدام باشد؟
- [x] 04. Runtime اجرای مدل چه باشد؟
- [x] 05. مدل چگونه وارد اپ شود؟
- [x] 06. مدل‌ها کجا نگهداری شوند؟
- [x] 07. آیا APK باید بدون مدل قابل نصب و اجرا باشد؟
- [x] 08. حداقل قابلیت‌های Chat چه باشند؟
- [x] 09. Streaming پاسخ لازم است؟
- [x] 10. Stop Generation لازم است؟
- [x] 11. چه تنظیمات Inference در UI نمایش داده شود؟
- [x] 12. Agent در Prototype دقیقاً چه کاری انجام دهد؟
- [x] 13. فرمت ارتباط مدل با Actionها چه باشد؟
- [x] 14. اولین Actionهای قابل اجرا کدام باشند؟
- [x] 15. Agent چند مرحله اجازه اجرای Action داشته باشد؟
- [x] 16. در صورت خطای Action چه اتفاقی بیفتد؟
- [ ] 17. چه اطلاعات Performance اندازه‌گیری شود؟
- [ ] 18. تست کاملاً Offline چگونه تأیید شود؟
- [ ] 19. معیار موفقیت Prototype چیست؟
- [ ] 20. چه زمانی قابلیت Local AI ارزش انتقال به WooGit را دارد؟

## پاسخ‌ها

### 16 — در صورت خطای Action چه اتفاقی بیفتد؟
**وضعیت:** تصمیم ثبت شد

**پاسخ:**

خطای Action باید به‌صورت واقعی و ساختاریافته از Executor به Agent برگردد. Agent باید بتواند خطا را بررسی کرده و در صورت امکان مسیر دیگری را انتخاب کند، اما Retry و ادامه اجرا باید محدود و قابل‌کنترل باشند.

**رفتار اصلی:**

1. خطای واقعی Executor به Agent برگردد.
2. Agent خطا را دریافت کند و بتواند بر اساس آن تصمیم بعدی بگیرد.
3. خطاهای قابل Retry با فیلد `retryable` مشخص شوند.
4. برای خطاهای قابل Retry حداکثر **۲ Retry برای همان Action** مجاز باشد.
5. خطاهای غیرقابل Retry نباید به‌صورت خودکار تکرار شوند.
6. Agent بتواند به‌جای Retry، در صورت امکان یک Action جایگزین انتخاب کند.
7. تمام خطاها در Agent Debug Log ثبت شوند.
8. خطای Parser قبل از Executor مدیریت شود و Action اجرا نشود.
9. Action حساسی که کاربر اجرای آن را رد کرده است، نباید خودکار Retry شود.
10. Maximum Agent Steps سؤال ۱۵ همچنان Hard Limit اصلی باشد.
11. بعد از رسیدن به Step Limit، Action جدید اجرا نشود.
12. Agent هرگز نباید موفقیت Action شکست‌خورده را جعل کند.
13. اگر خطا باعث ناقص‌ماندن کار شود، Final Answer باید این موضوع را صادقانه اعلام کند.

**ساختار پیشنهادی خطای Tool:**

```json
{
  "type": "tool_result",
  "action": "read_file",
  "success": false,
  "error": {
    "code": "FILE_NOT_FOUND",
    "message": "File does not exist",
    "retryable": false
  }
}
```

برای خطای موقتی:

```json
{
  "type": "tool_result",
  "action": "create_file",
  "success": false,
  "error": {
    "code": "TEMPORARY_IO_ERROR",
    "message": "Temporary I/O failure",
    "retryable": true
  }
}
```

**Retry:**

Retry باید از Step Limit جدا باشد. هر اجرای واقعی Action یک Agent Step مصرف می‌کند و هر Action حداکثر ۲ Retry مجاز دارد؛ Retry نیز اجرای واقعی Action است و بنابراین Step Budget را مصرف می‌کند.

**Parser Error:**

اگر مدل JSON نامعتبر تولید کند یا Schema را رعایت نکند، Executor نباید اجرا شود. Agent می‌تواند یک فرصت محدود برای اصلاح Action Request داشته باشد، ولی خروجی نامعتبر نباید مستقیماً اجرا شود.

**Confirmation Error:**

اگر Action حساس نیاز به تأیید داشته باشد و کاربر آن را رد کند، Action باید `Cancelled/Rejected` تلقی شود و Agent نباید همان Action را خودکار Retry کند.

**Step Limit:**

اگر Agent به Maximum Agent Steps قابل تنظیم کاربر برسد، Action بعدی Block می‌شود؛ حتی اگر مدل آن را درخواست کرده باشد. در این وضعیت موفقیت جعلی ممنوع است.

**مثال تصمیم‌گیری Agent:**

```text
read_file("test.txt")
        ↓
FILE_NOT_FOUND
        ↓
Agent receives real Tool Result
        ↓
list_files()
        ↓
["test1.txt", "notes.txt"]
        ↓
Agent determines test.txt does not exist
        ↓
Final Answer: فایل test.txt پیدا نشد.
```

**تصمیم نهایی:**

مدیریت خطا به‌صورت **Agent-aware** انتخاب شد: خطای واقعی و ساختاریافته → تصمیم Agent → Retry محدود یا Action جایگزین در صورت امکان → ثبت کامل در Debug Log. حداکثر ۲ Retry برای هر Action، Hard Step Limit قابل تنظیم، عدم Retry خودکار برای خطاهای غیرقابل Retry یا Action ردشده و ممنوعیت کامل جعل موفقیت.

### 15 — Agent چند مرحله اجازه اجرای Action داشته باشد؟
**وضعیت:** تصمیم ثبت شد

**پاسخ:**

Agent باید از Multi-Step Execution پشتیبانی کند و Maximum Agent Steps قابل تنظیم توسط کاربر باشد. مقدار پیش‌فرض ۵ Step است و Runtime باید یک سقف امن مستقل نیز داشته باشد. رسیدن به سقف باعث Block شدن Action بعدی می‌شود و جعل موفقیت ممنوع است.

### 14 — اولین Actionهای قابل اجرا کدام باشند؟
**وضعیت:** تصمیم ثبت شد

**پاسخ:**

Actionهای نسخه اول: `calculate`، `create_file`، `read_file`، `list_files`، `get_time`، `get_device_info`، `get_model_info`، `get_performance_stats` و `delete_file` با تأیید اجباری کاربر. Web Search، HTTP Request و Actionهای وابسته به Cloud فعلاً خارج از Prototype هستند.

### 13 — فرمت ارتباط مدل با Actionها چه باشد؟
**وضعیت:** تصمیم ثبت شد

**پاسخ:**

ارتباط مدل با Actionها باید با JSON ساختاریافته و Schema مشخص انجام شود. انواع اصلی `action`، `tool_result` و `final` هستند و Parser باید قبل از Executor اعتبارسنجی کامل انجام دهد. Structured Output / Grammar فقط در صورت پشتیبانی واقعی Runtime استفاده شود.

### 12 — Agent در Prototype دقیقاً چه کاری انجام دهد؟
**وضعیت:** تصمیم ثبت شد

**پاسخ:**

Agent باید یک Agent واقعی و قابل ارزیابی باشد؛ تشخیص نیاز به Action، انتخاب Action، تولید درخواست ساختاریافته، Parser، Executor واقعی، Tool Result واقعی، Final Answer، Multi-Step، Confirmation برای Actionهای حساس، Debug Mode و عدم جعل موفقیت از الزامات هستند.

### 11 — چه تنظیمات Inference در UI نمایش داده شود؟
**وضعیت:** تصمیم ثبت شد

**پاسخ:**

تنظیمات Inference در دو سطح Basic و Advanced ارائه شوند. Basic شامل Temperature و Max Tokens / Max New Tokens است. Advanced شامل Top-K، Top-P، Min-P، Repeat Penalty، Seed، Stop Sequences، Context Length و Structured Output / Grammar در صورت پشتیبانی واقعی Runtime است. تنظیمات بدون اثر واقعی در Runtime نباید نمایش داده شوند.

### 09 — Streaming پاسخ
**وضعیت:** تصمیم ثبت شد

**پاسخ:**

Streaming واقعی پاسخ الزامی است و باید امکان اندازه‌گیری First Token Time و Generation Speed را فراهم کند و با Stop Generation سازگار باشد.

### 10 — Stop Generation
**وضعیت:** تصمیم ثبت شد

**پاسخ:**

Stop Generation الزامی است و باید در سطح واقعی Runtime انجام شود. متن تولیدشده تا لحظه توقف حفظ و با وضعیت Stopped علامت‌گذاری شود.

### 08 — حداقل قابلیت‌های Chat
**وضعیت:** تصمیم ثبت شد

**پاسخ:**

قابلیت‌های ۱ تا ۲۰ مورد توافق قرار گرفتند:

1. ارسال پیام
2. دریافت پاسخ مدل
3. تاریخچه مکالمه / Context
4. New Chat
5. پاک‌کردن مکالمه
6. Streaming پاسخ
7. Stop Generation
8. نمایش وضعیت مدل شامل Loaded / Loading / Unloaded / Error
9. نمایش خطاهای واقعی
10. اجرای کاملاً Offline
11. نمایش وضعیت در حال تولید پاسخ
12. غیرفعال‌سازی ارسال در وضعیت‌های نامعتبر
13. امکان ادامه Generation پس از خطا یا Stop، در صورت پشتیبانی Runtime
14. شمارش توکن‌های ورودی و خروجی
15. نمایش سرعت Generation برحسب tok/s
16. اندازه‌گیری زمان رسیدن اولین Token
17. اندازه‌گیری زمان کل Generation
18. نمایش میزان مصرف Context
19. نمایش مدل فعال و مشخصات آن
20. نمایش تنظیمات فعلی Inference

**قابلیت‌های Debug و Agent مورد توافق:**

33. نمایش Raw Model Output
34. Agent Debug Mode برای نمایش Model → Action Request → Parser → Executor → Tool Result → Final Answer
35. درخواست تأیید کاربر قبل از اجرای Actionهای حساس
36. ثبت Action Log
37. نمایش Tool Result جدا از پاسخ نهایی AI

این قابلیت‌ها باید واقعی باشند و Mock یا رفتار جعلی وجود نداشته باشد.
