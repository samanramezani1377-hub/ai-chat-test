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

- **هیچ مقدار عددی ثابت یا پیش‌فرض اجباری برای Maximum Agent Steps تعریف نمی‌شود.**
- Maximum Agent Steps توسط کاربر قابل تنظیم است.
- رسیدن به Limit انتخاب‌شده توسط کاربر باعث Block شدن Action بعدی می‌شود.
- Runtime می‌تواند یک Hard Safety Limit مستقل برای جلوگیری از Loop بی‌نهایت یا اجرای غیرعادی داشته باشد؛ این Safety Limit جایگزین تنظیم کاربر نیست.
- جعل موفقیت پس از رسیدن به Limit ممنوع است.
- هر اجرای واقعی Action یک Agent Step مصرف می‌کند.

## 7. Action Error Handling

مدیریت خطا به‌صورت Agent-aware است:

`Real Error → Agent Decision → Limited Retry / Alternative Action → Final Result`

الزامات:

- خطای واقعی Executor به Agent برگردد.
- خطا ساختاریافته باشد.
- خطاهای قابل Retry با `retryable` مشخص شوند.
- Retry باید محدود باشد و تعداد آن توسط سیاست Retry سیستم کنترل شود.
- Retry نیز اجرای واقعی Action است و Step Budget را مصرف می‌کند.
- خطای غیرقابل Retry خودکار تکرار نشود.
- Agent در صورت امکان بتواند Action جایگزین انتخاب کند.
- همه خطاها در Agent Debug Log ثبت شوند.
- Parser Error قبل از Executor متوقف شود.
- Actionی که کاربر اجرای آن را رد کرده است خودکار Retry نشود.
- Step Limit همچنان Hard Limit باشد.
- موفقیت Action شکست‌خورده هرگز جعل نشود.
- اگر کار ناقص بماند، Final Answer باید صادقانه آن را اعلام کند.

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

## 11. معیار موفقیت Prototype — سؤال 19

هیچ Threshold، امتیاز، حداقل/حداکثر عددی یا معیار خودکار برای اعلام موفقیت Prototype تعریف نمی‌شود.

Prototype باید قابلیت‌ها و Metricهای واقعی موردنیاز را اجرا و ارائه کند. ارزیابی نهایی کیفیت، سرعت، مصرف منابع، پایداری، کیفیت فارسی، عملکرد Agent و Offline بودن **شخصاً توسط کاربر انجام می‌شود**.

سیستم نباید به‌صورت خودکار اعلام کند که Prototype موفق یا ناموفق شده است.

## 12. تصمیم انتقال به WooGit — سؤال 20

هیچ شرط، Threshold، امتیاز یا معیار عددی از پیش تعیین‌شده‌ای برای انتقال Local AI به WooGit تعریف نمی‌شود.

Prototype نتایج و شواهد واقعی را ارائه می‌کند و **تصمیم نهایی درباره مناسب بودن و انتقال Local AI به WooGit کاملاً با کاربر است**.

در صورت تصمیم به انتقال، معماری، قرارداد Actionها، Agent، Model Management، Performance Monitoring و تجربه فنی Prototype به‌عنوان مرجع مهاجرت استفاده می‌شوند.

## 13. معماری مهاجرت به WooGit و جداسازی Core از UI

### هدف معماری

این Prototype نباید به‌گونه‌ای طراحی شود که هسته Local AI به UI چت وابسته شود. هدف نهایی این است که قابلیت AI در WooGit هم از طریق Chat و هم از طریق قابلیت‌های مستقیم داخل UI قابل استفاده باشد.

نمونه قابلیت‌های آینده:

- اجرای عملیات WooCommerce از طریق Chat، مانند تغییر قیمت، موجودی، وضعیت یا اطلاعات محصول.
- قابلیت‌های مستقیم داخل UI، مانند یک دکمه کنار توضیحات محصول برای بازنویسی توضیحات با AI.
- قابلیت‌های مشابه برای عنوان، متن، محتوا و سایر بخش‌های محصول در آینده.

بنابراین معماری باید از ابتدا این دو مسیر را از یک AI Core مشترک تغذیه کند:

```text
                         AI CORE
                            │
                 ┌──────────┴──────────┐
                 ↓                     ↓
               Chat              UI AI Operations
                 ↓                     ↓
               Agent            Direct AI Operation
                 └──────────┬──────────┘
                            ↓
                     WooGit Services
                            ↓
                    WooCommerce API
```

### چرا Core باید از UI جدا باشد؟

1. **قابل استفاده بودن AI در چند نقطه**
   هسته AI نباید فقط برای صفحه Chat ساخته شود. Chat، صفحه محصول، دکمه بازنویسی، تنظیمات و قابلیت‌های آینده باید بتوانند همان Core را مصرف کنند.

2. **جلوگیری از وابستگی معماری**
   UI باید مصرف‌کننده AI باشد، نه صاحب منطق AI. اگر Agent، Model Management، Inference یا Action Execution داخل UI قرار بگیرد، استفاده مجدد و توسعه قابلیت‌های جدید سخت می‌شود.

3. **مهاجرت تمیز از Prototype به WooGit**
   هدف مهاجرت، کپی کل اپ `ai-chat-test` نیست. هسته قابل‌استفاده AI باید استخراج و به‌عنوان ماژول/Library مستقل در WooGit مصرف شود و `ai-chat-test` به‌عنوان محیط تست و مرجع باقی بماند.

4. **تعویض مدل بدون تغییر UI**
   UI نباید بداند مدل دقیقاً چگونه اجرا می‌شود. تغییر مدل، Quantization یا Runtime نباید مستلزم بازنویسی Chat یا UIهای WooGit باشد.

5. **تعویض یا توسعه UI بدون تغییر Core**
   اگر طراحی Chat یا صفحه محصول تغییر کند، نباید Agent و Inference تغییر کنند. همچنین اضافه شدن UI جدید نباید باعث کپی‌کردن منطق AI شود.

6. **تفکیک Agent از عملیات واقعی WooGit**
   Agent باید درخواست ساختاریافته تولید کند؛ WooGit باید اعتبارسنجی و اجرای واقعی عملیات را انجام دهد. بنابراین AI نباید مستقیماً APIهای WooCommerce را کنترل کند.

7. **قابل تست بودن مستقل Core**
   Core باید بدون نیاز به UI کامل قابل تست باشد. این موضوع اجازه می‌دهد رفتار مدل، Agent، Parser، Executor و Performance جداگانه Regression Test شوند.

8. **کاهش ریسک مهاجرت**
   اگر Core مستقل باشد، انتقال آن به WooGit مرحله‌ای است و لازم نیست کل پیام‌رسان Prototype وارد پروژه اصلی شود.

9. **جلوگیری از دو نسخه متفاوت از منطق AI**
   Chat و دکمه‌های AI داخل UI نباید پیاده‌سازی‌های جدا داشته باشند. هر دو باید از APIهای یک Core مشترک استفاده کنند.

10. **آمادگی برای قابلیت‌های آینده**
    قابلیت‌هایی که هنوز در Prototype وجود ندارند، مانند عملیات بیشتر روی محصولات، سفارش‌ها، مشتریان و محتوا، باید بتوانند به Core متصل شوند بدون اینکه معماری UI از نو طراحی شود.

### ساختار مفهومی پیشنهادی

```text
WooGit
│
├── app / UI
│   ├── Chat
│   ├── Product Editor
│   └── AI UI Operations
│
├── AI Core
│   ├── ModelManager
│   ├── AIEngine / Inference
│   ├── AgentManager
│   ├── AgentParser
│   ├── Action Protocol
│   ├── ActionExecutor
│   └── Performance / Observability
│
└── WooGit Operations
    ├── Product Operations
    ├── Order Operations
    ├── Customer Operations
    ├── Content Operations
    └── Media Operations
```

این ساختار مفهومی است و به معنی الزام به نام‌گذاری یا مسیر دقیق فایل‌ها نیست؛ اصل الزام، **جداسازی مسئولیت Core از UI و اتصال کنترل‌شده آن به سرویس‌های WooGit** است.

### Chat و UI باید از یک Operation مشترک استفاده کنند

مثلاً عملیات `update_product` نباید فقط برای Chat نوشته شود. Chat می‌تواند آن را درخواست کند و UI هم می‌تواند مستقیماً همان Operation را اجرا کند.

```text
Chat
  ↓
Agent
  ↓
update_product
  ↓
WooGit Operation
```

و:

```text
Product Editor
  ↓
AI UI Action
  ↓
update_product / rewrite_description / ...
  ↓
WooGit Operation
```

### عملیات تغییر داده

برای عملیات واقعی و حساس، مسیر کلی باید چنین باشد:

```text
AI Request
    ↓
Structured Action
    ↓
Validation
    ↓
Preview / Confirmation در صورت نیاز
    ↓
Real Execution
    ↓
Real Result
    ↓
AI Final Answer
```

برای مثال تغییر قیمت محصولات باید به عملیات واقعی WooGit متصل شود و نتیجه اجرای واقعی به Agent برگردد. AI نباید صرفاً بر اساس تولید متن ادعا کند که قیمت تغییر کرده است.

برای عملیات محتوایی مانند `rewrite_description` نیز بهتر است خروجی ابتدا به‌صورت Preview در اختیار کاربر قرار گیرد تا کاربر بتواند آن را قبول، ویرایش یا رد کند.

### اصل استقلال مدل

Operationهای WooGit نباید به مدل خاص وابسته باشند. اگر مدل از Qwen3-1.7B به مدل دیگری تغییر کند، قرارداد Operationهایی مانند `update_product` و `rewrite_description` نباید به‌خاطر تغییر مدل بازنویسی شوند.

### نقش ai-chat-test پس از مهاجرت

`ai-chat-test` نباید بعد از انتقال به‌عنوان یک کپی بلااستفاده باقی بماند. این پروژه باید تا حد امکان به‌عنوان **Test Harness / Reference** برای AI Core باقی بماند تا تغییرات Core قبل از ورود یا همراه با ورود به WooGit قابل تست و Regression باشند.

## 14. اصول عمومی Prototype

- رفتار Mock یا جعلی در مسیرهای اصلی مجاز نیست.
- هر موفقیت یا شکست باید بر اساس نتیجه واقعی Runtime/Executor باشد.
- قابلیت‌هایی که Runtime واقعاً پشتیبانی نمی‌کند نباید به‌صورت ظاهری و گمراه‌کننده ارائه شوند.
- Debug/Performance باید اطلاعات قابل ردیابی از اجرای واقعی سیستم ارائه دهد.
