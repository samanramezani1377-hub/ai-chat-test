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
- [ ] 16. در صورت خطای Action چه اتفاقی بیفتد؟
- [ ] 17. چه اطلاعات Performance اندازه‌گیری شود؟
- [ ] 18. تست کاملاً Offline چگونه تأیید شود؟
- [ ] 19. معیار موفقیت Prototype چیست؟
- [ ] 20. چه زمانی قابلیت Local AI ارزش انتقال به WooGit را دارد؟

## پاسخ‌ها

### 15 — Agent چند مرحله اجازه اجرای Action داشته باشد؟
**وضعیت:** تصمیم ثبت شد

**پاسخ:**

Agent باید از **Multi-Step Execution** پشتیبانی کند و برای جلوگیری از Loop بی‌نهایت، مصرف بی‌دلیل منابع و اجرای کنترل‌نشده Actionها، یک **Hard Step Limit** داشته باشد.

مقدار پیش‌فرض پیشنهادی و مورد توافق: **۵ Step**.

اما این مقدار باید **توسط خود کاربر قابل تنظیم** باشد.

**رفتار تنظیمات:**

- مقدار پیش‌فرض: 5
- کاربر می‌تواند Maximum Agent Steps را از UI تغییر دهد.
- مقدار انتخابی کاربر باید واقعاً توسط Agent Runtime اعمال شود.
- تنظیمات باید در بخش مناسب Agent/Advanced قابل مشاهده و تغییر باشند.
- مقدار تنظیم‌شده باید در Agent Debug Mode نیز نمایش داده شود؛ مثلاً `Step 2 / 5`.
- UI نباید اجازه مقدار نامعتبر یا بدون محدودیت واقعی را بدهد.
- یک حد بالای امن Runtime باید وجود داشته باشد تا کاربر نتواند با تنظیم UI عملاً Agent را بدون محدودیت کند.

**نمونه UI:**

```text
Agent Settings

Maximum Agent Steps
[ 5 ]

Agent Debug Mode
[ ON ]
```

**نمونه اجرای چندمرحله‌ای:**

```text
Step 1 / 5 → list_files
Step 2 / 5 → read_file
Step 3 / 5 → calculate
Step 4 / 5 → create_file
Step 5 / 5 → read_file
```

اگر Agent در هر مرحله قبل از رسیدن به Limit به Final Answer برسد، اجرا طبیعی تمام می‌شود.

**رسیدن به Limit:**

اگر Agent به Maximum Agent Steps برسد، Action بعدی نباید اجرا شود.

مثلاً:

```text
Step 5 / 5
        ↓
MAX_STEPS_REACHED
        ↓
Action #6 = BLOCKED
```

در Debug Mode باید مشخص شود که اجرای Action بعدی به دلیل Step Limit مسدود شده است.

اگر کار هنوز کامل نشده باشد، Agent نباید موفقیت را جعل کند. باید وضعیت ناقص/متوقف‌شده را به کاربر اعلام کند.

**جلوگیری از Loop:**

- Step Limit سخت و واقعی است.
- هر اجرای واقعی Action یک Step مصرف می‌کند.
- Actionهای نامعتبر نباید Step معتبر محسوب شوند و نباید به Executor برسند.
- پس از رسیدن به Limit، هیچ Action جدیدی اجرا نمی‌شود.
- Agent نباید امکان Loop بی‌نهایت داشته باشد.

**تصمیم نهایی:**

Multi-Step Agent با **Maximum Agent Steps قابل تنظیم توسط کاربر** انتخاب شد؛ مقدار پیش‌فرض ۵ است و Runtime باید یک سقف امن مستقل نیز داشته باشد. رسیدن به سقف باعث Block شدن Action بعدی می‌شود و جعل موفقیت ممنوع است.

### 14 — اولین Actionهای قابل اجرا کدام باشند؟
**وضعیت:** تصمیم ثبت شد

**پاسخ:**

Actionهای نسخه اول:

**ضروری:**
- `calculate`
- `create_file`
- `read_file`
- `list_files`
- `get_time`
- `get_device_info`
- `get_model_info`
- `get_performance_stats`

**Safety Test:**
- `delete_file` با تأیید اجباری کاربر

**فعلاً خارج از Prototype:**
- Web Search
- HTTP Request
- Internet Tools
- Actionهای وابسته به Cloud

همه Actionها باید واقعی باشند و Mock یا موفقیت جعلی ممنوع است.

### 13 — فرمت ارتباط مدل با Actionها چه باشد؟
**وضعیت:** تصمیم ثبت شد

**پاسخ:**

ارتباط مدل با Actionها باید با **JSON ساختاریافته و Schema مشخص** انجام شود. مدل نباید برای درخواست اجرای Action از متن آزاد استفاده کند.

ساختار اصلی Action Request:

```json
{
  "type": "action",
  "action": {
    "name": "calculate",
    "arguments": {
      "expression": "125 * 37"
    }
  }
}
```

ساختار Final Answer:

```json
{
  "type": "final",
  "content": "حاصل ۱۲۵ × ۳۷ برابر با ۴۶۲۵ است."
}
```

ساختار Tool Result:

```json
{
  "type": "tool_result",
  "action": "calculate",
  "success": true,
  "result": "4625"
}
```

ساختار خطای Tool Result نیز باید نتیجه واقعی Executor را منتقل کند.

**الزامات Parser:**

- JSON معتبر باشد.
- `type` معتبر باشد.
- Action در فهرست Actionهای مجاز باشد.
- `arguments` مطابق Schema همان Action اعتبارسنجی شود.
- Action نامعتبر یا ناقص به Executor ارسال نشود.
- Parser بر اساس حدس، خروجی خراب را به Action معتبر تبدیل نکند.

**زنجیره:**

`LLM → JSON Action Request → Parser → Schema Validation → Permission/Confirmation → Executor → Tool Result → LLM → Final Answer`

Structured Output / Grammar در صورت پشتیبانی واقعی Runtime استفاده شود و Mock ممنوع است.

### 12 — Agent در Prototype دقیقاً چه کاری انجام دهد؟
**وضعیت:** تصمیم ثبت شد

**پاسخ:**

Agent باید یک Agent واقعی و قابل ارزیابی باشد؛ نه صرفاً یک Chat با عنوان Agent. باید نیاز به Action را تشخیص دهد، Action مناسب را انتخاب کند، درخواست ساختاریافته تولید کند، Parser آن را اعتبارسنجی کند، Executor آن را واقعاً اجرا کند، Tool Result واقعی را دریافت کند و Final Answer را بر اساس نتیجه واقعی تولید کند.

قابلیت‌ها شامل Multi-Step Execution، تأیید کاربر برای Actionهای حساس، Agent Debug Mode، انتقال خطای واقعی، عدم جعل موفقیت، اجرای Local و عدم استفاده از Mock هستند.

### 11 — چه تنظیمات Inference در UI نمایش داده شود؟
**وضعیت:** تصمیم ثبت شد

**پاسخ:**

تنظیمات Inference در دو سطح Basic و Advanced ارائه شوند.

**Basic:** Temperature، Max Tokens / Max New Tokens

**Advanced:** Top-K، Top-P، Min-P، Repeat Penalty، Seed، Stop Sequences، Context Length و Structured Output / Grammar در صورت پشتیبانی واقعی Runtime.

تنظیمات بدون اثر واقعی در Runtime نباید نمایش داده شوند.

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
