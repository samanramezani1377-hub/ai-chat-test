# AI Chat Test

> **Prototype / Technical Feasibility Test برای Local AI روی Android و Agent قابل‌اعتماد**

این repository یک محیط آزمایشی برای بررسی فنی اجرای هوش مصنوعی کاملاً Local روی Android است؛ نه محصول نهایی.

هدف اصلی این است که قبل از انتقال قابلیت Local AI به **WooGit**، بتوانیم اجرای مدل، Chat فارسی، Streaming، Performance، Agent، Actionهای واقعی، Verification و یک **AI Workspace** را روی دستگاه واقعی آزمایش و ارزیابی کنیم.

## فهرست مستندات

این README نمای کلی پروژه است و جزئیات تصمیم‌ها در مستندات زیر نگهداری می‌شوند:

| سند | موضوع |
|---|---|
| [`PROTOTYPE_SPEC.md`](docs/PROTOTYPE_SPEC.md) | Specification اصلی و تصمیم‌های نهایی Prototype، Chat، Inference، Agent، Action Protocol، Safety، Performance، Network و مسیر مهاجرت به WooGit |
| [`ACTION_EXECUTION_VERIFICATION.md`](docs/ACTION_EXECUTION_VERIFICATION.md) | اثبات اجرای واقعی Action، Executor، Verifier، Before/After و جلوگیری از Success جعلی |
| [`AI_WORKSPACE.md`](docs/AI_WORKSPACE.md) | طراحی AI Workspace / میز کار، Preview، تعامل کاربر و ارتباط آن با Core و Action System |
| [`PERFORMANCE_METRICS.md`](docs/PERFORMANCE_METRICS.md) | Metricهای کامل Performance، Network، History، Export و تنظیم مستقل Visibility |
| [`QUESTION_BANK.md`](docs/QUESTION_BANK.md) | Question Bank و وضعیت تصمیم‌های پروژه |

**مرجع اصلی تصمیم‌های فنی:** [`PROTOTYPE_SPEC.md`](docs/PROTOTYPE_SPEC.md)

## 1. هدف پروژه

پروژه باید مشخص کند Local AI روی Android تا چه حد برای استفاده آینده در WooGit عملی و قابل‌اتکا است.

تمرکز Prototype:

- اجرای Local Model
- Chat فارسی
- Streaming واقعی
- Stop Generation
- تنظیمات Inference
- Agent واقعی
- Actionهای واقعی
- Action Verification
- Performance Monitoring
- Network Monitoring و Offline
- AI Workspace
- آماده‌سازی معماری برای مهاجرت به WooGit

این پروژه عمداً یک Prototype است و نباید بدون نیاز به هدف اصلی پیچیده شود.

جزئیات کامل: [`PROTOTYPE_SPEC.md`](docs/PROTOTYPE_SPEC.md)

## 2. مدل Baseline

مدل پایه فعلی:

- **Qwen3-1.7B**
- Format: **GGUF**
- Initial Quantization: **Q4_K_M**
- Runtime: **llama.cpp**
- مدل داخل APK قرار نمی‌گیرد.
- کاربر مدل را به‌صورت فایل جداگانه Import می‌کند.

Qwen3-1.7B فعلاً Baseline Prototype است و انتخاب آن قطعی برای محصول نهایی نیست. معماری باید امکان تعویض مدل و در آینده استفاده از مدل‌های بزرگ‌تر را بدون بازطراحی اساسی UI و Core فراهم کند.

## 3. معماری کلان

اصل مهم معماری این است که **AI Core از UI جدا باشد**.

```text
                         AI CORE
                            │
              ┌─────────────┴─────────────┐
              ↓                           ↓
            Chat                      Workspace
              ↓                           ↓
           Agent                  AI UI Operations
              └─────────────┬─────────────┘
                            ↓
                      Action System
                            ↓
                   Executor + Verifier
                            ↓
                       Real Result
                            ↓
                    UI / Workspace
```

AI Core باید مسئول منطق Local AI، Model Management، Inference، Agent، Action Protocol، Executor، Verification و Performance/Observability باشد؛ UI فقط مصرف‌کننده این قابلیت‌ها و نمایش‌دهنده State/Result واقعی باشد.

### چرا Core از UI جداست؟

- Chat تنها مصرف‌کننده AI نیست.
- قابلیت‌هایی مانند «بازنویسی توضیحات با AI» باید بتوانند مستقیماً از Core استفاده کنند.
- Chat و UI باید Operationهای مشترک داشته باشند و منطق AI را دوباره پیاده‌سازی نکنند.
- تعویض مدل، Quantization یا Runtime نباید UI را مجبور به بازنویسی کند.
- UI جدید نباید باعث کپی‌شدن Agent یا Inference شود.
- Core باید مستقل تست و Regression شود.
- مهاجرت به WooGit باید استخراج Core باشد، نه کپی کل اپ.

جزئیات کامل معماری و مسیر مهاجرت: [`PROTOTYPE_SPEC.md`](docs/PROTOTYPE_SPEC.md)

## 4. Model Management

کاربر باید بتواند Model Import، Validate، Load و Unload کند، مدل‌های Import‌شده را ببیند و وضعیت مدل را مشاهده کند.

مدل نباید در کد hardcode شود و مسیر فایل باید با Android Storage Access Framework مدیریت شود.

## 5. Local Chat

پس از Load مدل:

- پیام کاربر دریافت شود.
- Inference کاملاً روی دستگاه اجرا شود.
- پاسخ به‌صورت Streaming واقعی نمایش داده شود.
- Generation واقعاً قابل Stop باشد.
- خطاهای واقعی نمایش داده شوند.
- وضعیت مدل و Generation قابل مشاهده باشد.

برای Inference نباید وابستگی اجباری به Cloud یا API خارجی وجود داشته باشد.

## 6. Inference Settings

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

تنظیمی که Runtime واقعاً اعمال نمی‌کند نباید در UI به‌عنوان قابلیت فعال نمایش داده شود.

## 7. Agent

Agent باید واقعی باشد و صرفاً شبیه‌سازی UI نباشد.

```text
User Request
     ↓
Model
     ↓
Action Request
     ↓
Parser / Validator
     ↓
ActionExecutor
     ↓
Real Execution
     ↓
ActionVerifier
     ↓
Verified Tool Result
     ↓
Agent
     ↓
Final Answer
```

Agent باید Multi-Step باشد و Maximum Agent Steps توسط کاربر قابل تنظیم باشد؛ هیچ مقدار عددی ثابت یا پیش‌فرض اجباری برای آن تعریف نمی‌شود. Runtime می‌تواند Hard Safety Limit مستقل برای جلوگیری از Loop غیرعادی داشته باشد.

## 8. Actionهای Prototype

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

## 9. Action واقعی و Verification

مدل نباید صرفاً بگوید Action انجام شده است. اجرای واقعی باید قابل اثبات باشد.

**Executor انجام می‌دهد → Verifier ثابت می‌کند → Agent نتیجه را تفسیر می‌کند → UI شواهد را نشان می‌دهد.**

برای Actionهای State-changing در صورت امکان وضعیت Before و After بررسی می‌شود.

مثلاً:

```text
BEFORE
exists: false

ACTION
create_file
file: test.txt

EXECUTOR
✓ completed

VERIFICATION
✓ file exists
✓ content matches

AFTER
exists: true

VERIFIED ✓
```

اگر Verification شکست بخورد، Agent نباید موفقیت را اعلام کند.

جزئیات کامل: [`ACTION_EXECUTION_VERIFICATION.md`](docs/ACTION_EXECUTION_VERIFICATION.md)

## 10. Action Debug / Test Panel

برای هر اجرای Action حداقل این موارد باید قابل مشاهده باشند:

1. Action Request
2. پارامترهای ورودی
3. نتیجه واقعی Executor
4. نتیجه Verification
5. Final Answer

در صورت امکان Before/After State نیز نمایش داده شود.

هدف این است که کاربر بتواند با مشاهده شواهد واقعی تشخیص دهد Action واقعاً اجرا شده یا فقط مدل ادعای اجرای آن را کرده است.

جزئیات: [`ACTION_EXECUTION_VERIFICATION.md`](docs/ACTION_EXECUTION_VERIFICATION.md)

## 11. AI Workspace — میز کار

Prototype باید علاوه بر Chat یک **AI Workspace** داشته باشد.

Workspace فضای کاری قابل مشاهده و تعاملی برای:

- نتایج Action
- Tool Result
- Verification
- خروجی‌های چندمرحله‌ای Agent
- فایل‌ها و داده‌های مورد استفاده
- Preview
- تأیید یا رد عملیات
- Before / After

Workspace **اجراکننده Action نیست**. اجرای واقعی از مسیر Core → Operation → Executor → Verifier انجام می‌شود.

در آینده در WooGit می‌تواند برای تغییر قیمت، نمایش Before/After، بازنویسی توضیحات، تولید عنوان، اصلاح محتوا و سایر AI Operations استفاده شود.

جزئیات کامل: [`AI_WORKSPACE.md`](docs/AI_WORKSPACE.md)

## 12. Performance & Observability

Metricهای واقعی باید اندازه‌گیری شوند؛ Metric صوری، تخمینی یا Mock مجاز نیست.

### Inference

- First Token Time
- Input Tokens
- Output Tokens
- Generation Time
- Tokens/sec
- Context Usage

### Model

- Load Time
- Unload Time در صورت پشتیبانی واقعی Runtime

### Device

- RAM
- CPU
- GPU/NPU در صورت دسترسی واقعی
- Backend

### Agent

- Total Agent Time
- Step Count
- Action Time
- Retry Count
- Error Count

### Network

- Network Request/Connectionهای واقعی
- Network Usage در حد اطلاعاتی که Android/Runtime واقعاً ارائه می‌کند

### History / Reporting

- ذخیره نتایج قبلی
- مقایسه Performance
- Export گزارش

جزئیات: [`PERFORMANCE_METRICS.md`](docs/PERFORMANCE_METRICS.md)

## 13. Visibility قابل تنظیم

تمام Metricها باید اندازه‌گیری شوند، اما نمایش هر Metric باید **کاملاً مستقل** قابل روشن/خاموش‌شدن باشد.

مثلاً:

```kotlin
const val SHOW_CPU = false
```

باید فقط نمایش CPU را خاموش کند؛ Measurement نباید غیرفعال شود.

اگر Metric واقعاً قابل اندازه‌گیری نباشد، `Unavailable` نمایش داده شود یا بر اساس Visibility همان Metric مخفی شود؛ مقدار ساختگی مجاز نیست.

جزئیات: [`PERFORMANCE_METRICS.md`](docs/PERFORMANCE_METRICS.md)

## 14. Network & Offline

- تمام Network Usage اپ باید واقعاً مانیتور شود.
- Network Request/Connectionهای واقعی و میزان مصرف شبکه، در حد اطلاعاتی که Android/Runtime ارائه می‌کند، ثبت شوند.
- Network information در Performance/Debug قابل مشاهده باشد.
- Offline با قطع واقعی اینترنت تست می‌شود.
- Firewall، DNS، Fresh Install و سناریوهای پیچیده خارج از Requirement هستند.

جزئیات: [`PROTOTYPE_SPEC.md`](docs/PROTOTYPE_SPEC.md) و [`PERFORMANCE_METRICS.md`](docs/PERFORMANCE_METRICS.md)

## 15. مهاجرت به WooGit

هدف، انتقال کل `ai-chat-test` به WooGit نیست.

```text
ai-chat-test
      │
      ↓
  AI Core
      │
 ┌────┴─────┐
 ↓          ↓
ai-chat-test WooGit
 Test       Production
Harness
```

AI Core باید بتواند در WooGit به‌عنوان ماژول/Library مستقل مصرف شود.

### Chat

```text
Chat
 ↓
Agent
 ↓
WooGit Operation
 ↓
Executor
 ↓
Verifier
 ↓
Real Result
```

### UI AI Operation

```text
Product Editor
 ↓
[✨ بازنویسی با AI]
 ↓
AI Core
 ↓
rewrite_description
 ↓
Preview
 ↓
Accept / Edit / Reject
```

هم Chat و هم UI باید بتوانند از Operationهای مشترک استفاده کنند. برای عملیات حساس مانند تغییر قیمت، Validation و در صورت نیاز Preview/Confirmation لازم است.

جزئیات معماری مهاجرت: [`PROTOTYPE_SPEC.md`](docs/PROTOTYPE_SPEC.md) و [`AI_WORKSPACE.md`](docs/AI_WORKSPACE.md)

## 16. تصمیم درباره انتقال

برای Prototype هیچ Threshold یا امتیاز عددی از پیش تعیین‌شده‌ای برای اعلام موفقیت یا انتقال وجود ندارد.

Prototype باید شواهد واقعی ارائه کند و **تصمیم نهایی درباره کیفیت Prototype و مناسب بودن انتقال به WooGit با کاربر است**.

## 17. تست‌های پایه

### Chat فارسی

```text
سلام، خودت را معرفی کن و بگو چه کارهایی می‌توانی انجام بدهی.
```

### محاسبه

```text
125 × 37 چند می‌شود؟
```

### Action

```text
یک فایل متنی با نام test.txt بساز و داخل آن دقیقاً عبارت Hello World را قرار بده.
```

در این تست علاوه بر Final Answer، اجرای واقعی و Verification باید در Debug/Workspace قابل مشاهده باشد.

### WooCommerce-style Action

```text
یک محصول جدید ایجاد کن با عنوان «تشک طبی فنری»، قیمت ۵ میلیون تومان، موجودی ۱۲ عدد و وضعیت پیش‌نویس.
```

این تست در Prototype صرفاً تبدیل دستور طبیعی به درخواست ساختاریافته را بررسی می‌کند و اتصال واقعی WooCommerce فعلاً لازم نیست.

## 18. محدوده خارج از Prototype

فعلاً خارج از محدوده:

- Cloud AI
- API خارجی
- Multi-agent
- RAG
- Vector Database
- Voice
- Image Generation
- حساب کاربری
- Marketplace مدل
- سیستم Plugin پیچیده
- Autonomous Background Agent
- تعداد زیاد Tool

## 19. اصول توسعه

هر تغییر باید:

1. با هدف Prototype مرتبط باشد.
2. Build را خراب نکند.
3. قابلیت قبلی را بدون دلیل حذف نکند.
4. نتیجه واقعی تست را گزارش کند.
5. Mock یا گزارش جعلی ایجاد نکند.
6. با Specification و مستندات پروژه سازگار باشد.

## 20. وضعیت

**Baseline Model:** Qwen3-1.7B  
**Format:** GGUF  
**Initial Quantization:** Q4_K_M  
**Runtime:** llama.cpp  
**Purpose:** Android Local AI + Agent Feasibility Test  
**Architecture Direction:** AI Core جدا از UI + Chat + AI Workspace + Action Verification  
**Future Target:** امکان انتقال کنترل‌شده AI Core به WooGit

برای جزئیات اجرایی و تصمیم‌های کامل، از [`docs/PROTOTYPE_SPEC.md`](docs/PROTOTYPE_SPEC.md) شروع کنید.
