# AI Chat Test — Prototype Specification

این سند تصمیم‌های نهایی ثبت‌شده برای Prototype را به‌صورت Specification اجرایی نگهداری می‌کند. تصمیم‌ها از Question Bank استخراج شده‌اند و از اینجا به بعد مرجع طراحی و پیاده‌سازی هستند.

## 1. Chat

Prototype باید یک Chat واقعی و قابل ارزیابی داشته باشد.

قابلیت‌های توافق‌شده:

- ارسال پیام
- دریافت پاسخ مدل
- تاریخچه مکالمه و Context
- New Chat
- پاک‌کردن مکالمه
- Streaming واقعی پاسخ
- Stop Generation واقعی در سطح Runtime
- نمایش وضعیت مدل: Loaded / Loading / Unloaded / Error
- نمایش خطاهای واقعی
- اجرای Local AI بدون وابستگی اجباری به شبکه
- نمایش وضعیت در حال تولید پاسخ
- غیرفعال‌سازی ارسال در وضعیت‌های نامعتبر
- ادامه Generation پس از خطا یا Stop در صورت پشتیبانی Runtime
- شمارش Tokenهای ورودی و خروجی
- نمایش سرعت Generation برحسب tok/s
- اندازه‌گیری First Token Time
- اندازه‌گیری Total Generation Time
- نمایش Context Usage
- نمایش مدل فعال و مشخصات آن
- نمایش تنظیمات فعلی Inference

Streaming باید با Stop Generation و Performance Monitoring سازگار باشد.

## 2. Inference Settings

تنظیمات در دو سطح ارائه شوند.

### Basic

- Temperature
- Max Tokens / Max New Tokens

### Advanced

- Top-K
- Top-P
- Min-P
- Repeat Penalty
- Seed
- Stop Sequences
- Context Length
- Structured Output / Grammar در صورت پشتیبانی واقعی Runtime

تنظیمی که Runtime واقعاً پشتیبانی یا اعمال نمی‌کند نباید در UI به‌عنوان تنظیم فعال نمایش داده شود.

## 3. Agent

Agent باید واقعی و قابل ارزیابی باشد و فقط یک شبیه‌سازی UI نباشد.

مسیر اصلی:

`Model → Action Request → Parser → Executor → Tool Result → Final Answer`

Agent باید بتواند:

- تشخیص دهد آیا برای درخواست به Action نیاز دارد یا خیر.
- Action مناسب را انتخاب کند.
- درخواست ساختاریافته تولید کند.
- درخواست را قبل از اجرا Parse و Validate کند.
- Action واقعی را اجرا کند.
- Tool Result واقعی دریافت کند.
- بر اساس Tool Result تصمیم مرحله بعدی را بگیرد.
- Multi-Step Execution انجام دهد.
- در Actionهای حساس از کاربر تأیید بگیرد.
- Final Answer واقعی تولید کند.
- در Debug Mode مسیر اجرای خود را قابل مشاهده کند.
- هرگز موفقیت ساختگی اعلام نکند.

## 4. Action Protocol

ارتباط مدل با Actionها با JSON ساختاریافته و Schema مشخص انجام می‌شود.

انواع اصلی:

- `action`
- `tool_result`
- `final`

Parser باید قبل از Executor اعتبارسنجی کامل انجام دهد. JSON نامعتبر یا Schema نامعتبر نباید مستقیماً اجرا شود.

Structured Output / Grammar فقط در صورت پشتیبانی واقعی Runtime استفاده می‌شود.

## 5. Initial Actions

Actionهای نسخه اول:

- `calculate`
- `create_file`
- `read_file`
- `list_files`
- `get_time`
- `get_device_info`
- `get_model_info`
- `get_performance_stats`
- `delete_file`

`delete_file` نیازمند تأیید اجباری کاربر است.

Web Search، HTTP Request و Actionهای وابسته به Cloud فعلاً خارج از Prototype هستند.

## 6. Agent Step Limit

Agent باید Multi-Step باشد.

- Maximum Agent Steps توسط کاربر قابل تنظیم است.
- مقدار پیش‌فرض: 5 Step
- Runtime باید یک سقف امن مستقل نیز داشته باشد.
- رسیدن به Maximum Agent Steps باعث Block شدن Action بعدی می‌شود.
- جعل موفقیت پس از رسیدن به Limit ممنوع است.

هر اجرای واقعی Action یک Agent Step مصرف می‌کند.

## 7. Action Error Handling

مدیریت خطا به‌صورت Agent-aware است:

`Real Error → Agent Decision → Limited Retry / Alternative Action → Final Result`

الزامات:

- خطای واقعی Executor به Agent برگردد.
- خطا ساختاریافته باشد.
- خطاهای قابل Retry با `retryable` مشخص شوند.
- هر Action حداکثر 2 Retry داشته باشد.
- Retry نیز اجرای واقعی Action است و Step Budget را مصرف می‌کند.
- خطای غیرقابل Retry خودکار تکرار نشود.
- Agent در صورت امکان بتواند Action جایگزین انتخاب کند.
- همه خطاها در Agent Debug Log ثبت شوند.
- Parser Error قبل از Executor متوقف شود.
- Actionی که کاربر اجرای آن را رد کرده است خودکار Retry نشود.
- Step Limit همچنان Hard Limit باشد.
- موفقیت Action شکست‌خورده هرگز جعل نشود.
- اگر کار ناقص بماند، Final Answer باید صادقانه آن را اعلام کند.

نمونه Tool Error:

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

## 8. Debug & Agent Observability

قابلیت‌های توافق‌شده:

- نمایش Raw Model Output
- Agent Debug Mode با نمایش زنجیره Model → Action Request → Parser → Executor → Tool Result → Final Answer
- تأیید کاربر برای Actionهای حساس
- Action Log
- نمایش Tool Result جدا از پاسخ نهایی AI

تمام این اطلاعات باید از اجرای واقعی سیستم به‌دست آیند و Mock نباشند.

## 9. Performance Monitoring

همه Performance Metricها باید اندازه‌گیری شوند.

### Inference Metrics

- First Token Time
- Input Tokens
- Output Tokens
- Generation Time
- Tokens/sec
- Context Usage

### Model Metrics

- Load Time
- Unload Time، در صورت پشتیبانی Runtime

### Device Metrics

- RAM
- CPU
- GPU/NPU، در صورت دسترسی واقعی
- Backend

### Agent Metrics

- Total Agent Time
- Step Count
- Action Time
- Retry Count
- Error Count

### History & Reporting

- Performance History
- مقایسه تست‌ها
- Export گزارش Performance

### مستقل بودن Visibility

اندازه‌گیری و نمایش از هم مستقل باشند. همه Metricها اندازه‌گیری می‌شوند، اما نمایش هر Metric باید مستقل و قابل تنظیم باشد. Visibility هر Metric باید با یک تغییر بسیار کوچک در یک محل مرکزی کد قابل روشن/خاموش شدن باشد.

برای نمونه:

```kotlin
const val SHOW_FIRST_TOKEN_TIME = true
const val SHOW_INPUT_TOKENS = true
const val SHOW_OUTPUT_TOKENS = true
const val SHOW_TOKENS_PER_SECOND = true
const val SHOW_CONTEXT_USAGE = true
const val SHOW_RAM = true
const val SHOW_CPU = true
const val SHOW_GPU_NPU = true
const val SHOW_AGENT_TIME = true
const val SHOW_ACTION_TIME = true
```

خاموش‌کردن Visibility یک Metric نباید اندازه‌گیری آن را غیرفعال کند.

هر Metric باید از Runtime/Android و مسیر واقعی اندازه‌گیری شود. اگر اندازه‌گیری یک Metric واقعاً ممکن نباشد، `Unavailable` نمایش داده شود و مقدار تخمینی یا ساختگی مجاز نیست.

## 10. Network Monitoring & Offline

مسیر Local AI نباید برای کارکرد اصلی خود وابستگی اجباری به شبکه داشته باشد.

در Prototype:

- تمام Network Usage اپ باید به‌صورت واقعی مانیتور شود.
- Network Request/Connectionهای واقعی و میزان مصرف شبکه، در حد اطلاعاتی که Android/Runtime واقعاً ارائه می‌کند، ثبت شوند.
- اطلاعات شبکه در Performance/Debug قابل مشاهده باشند.
- تست Offline واقعی با قطع اینترنت توسط کاربر انجام می‌شود.
- تست‌های اضافی مانند Firewall، DNS و Fresh Install جزو Requirement نیستند.
- هیچ مقدار ساختگی برای Network Usage مجاز نیست.

## 11. اصول عمومی Prototype

- رفتار Mock یا جعلی در مسیرهای اصلی مجاز نیست.
- هر موفقیت یا شکست باید بر اساس نتیجه واقعی Runtime/Executor باشد.
- قابلیت‌هایی که Runtime واقعاً پشتیبانی نمی‌کند نباید به‌صورت ظاهری و گمراه‌کننده ارائه شوند.
- Debug/Performance باید اطلاعات قابل ردیابی از اجرای واقعی سیستم ارائه دهد.
