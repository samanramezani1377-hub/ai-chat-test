# AI Chat Test

> **Prototype / Technical Feasibility Test برای Local AI روی Android و Agent قابل‌اعتماد**

این repository یک محیط آزمایشی برای بررسی فنی اجرای هوش مصنوعی کاملاً Local روی Android است؛ نه محصول نهایی.

هدف اصلی این است که قبل از انتقال قابلیت Local AI به **WooGit**، اجرای مدل، Chat فارسی، Streaming، Performance، Agent، Actionهای واقعی، Verification و یک **AI Workspace** روی دستگاه واقعی آزمایش و ارزیابی شوند.

## فهرست مستندات

README نقشه کلی پروژه است. جزئیات هر بخش در سند تخصصی خودش نگهداری می‌شود:

| سند | موضوع |
|---|---|
| [`PROTOTYPE_SPEC.md`](docs/PROTOTYPE_SPEC.md) | Specification اصلی و تصمیم‌های نهایی Prototype |
| [`ARCHITECTURE.md`](docs/ARCHITECTURE.md) | معماری Core/UI، مرزبندی لایه‌ها، Adapterها و مسیر معماری WooGit |
| [`ACTION_PROTOCOL.md`](docs/ACTION_PROTOCOL.md) | قرارداد ActionRequest/ToolResult، Schema، Lifecycle و Errorها |
| [`ACTION_EXECUTION_VERIFICATION.md`](docs/ACTION_EXECUTION_VERIFICATION.md) | اجرای واقعی Action، Executor، Verifier و شواهد Before/After |
| [`TASK_STATE_MACHINE.md`](docs/TASK_STATE_MACHINE.md) | State Machine مربوط به Task، Agent، Execution و Cancellation |
| [`SECURITY_MODEL.md`](docs/SECURITY_MODEL.md) | Permission، Confirmation، Risk Level و مرز اعتماد Actionها |
| [`DATA_AND_PRIVACY.md`](docs/DATA_AND_PRIVACY.md) | مرز داده Local، Network، Storage، Logging و Privacy |
| [`AI_WORKSPACE.md`](docs/AI_WORKSPACE.md) | طراحی AI Workspace / میز کار و تعامل آن با Core و Action System |
| [`PERFORMANCE_METRICS.md`](docs/PERFORMANCE_METRICS.md) | Performance، Network، History، Export و Visibility مستقل Metricها |
| [`TEST_MATRIX.md`](docs/TEST_MATRIX.md) | ماتریس تست، سناریوهای موفق/ناموفق و Benchmark Protocol |
| [`IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) | مسیر مرحله‌ای تبدیل Specification به پیاده‌سازی و سپس مهاجرت به WooGit |
| [`QUESTION_BANK.md`](docs/QUESTION_BANK.md) | Question Bank و تصمیم‌های ثبت‌شده پروژه |

### ترتیب پیشنهادی مطالعه

```text
README
  ↓
PROTOTYPE_SPEC
  ↓
ARCHITECTURE
  ├── ACTION_PROTOCOL
  ├── TASK_STATE_MACHINE
  └── SECURITY_MODEL
  ↓
ACTION_EXECUTION_VERIFICATION
  ↓
AI_WORKSPACE
  ↓
PERFORMANCE_METRICS + DATA_AND_PRIVACY
  ↓
TEST_MATRIX
  ↓
IMPLEMENTATION_PLAN
```

**مرجع اصلی تصمیم‌های فنی:** [`PROTOTYPE_SPEC.md`](docs/PROTOTYPE_SPEC.md)

---

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

جزئیات: [`PROTOTYPE_SPEC.md`](docs/PROTOTYPE_SPEC.md)

---

## 2. مدل Baseline

مدل پایه فعلی:

- **Qwen3-1.7B**
- Format: **GGUF**
- Initial Quantization: **Q4_K_M**
- Runtime: **llama.cpp**
- مدل داخل APK قرار نمی‌گیرد.
- کاربر مدل را به‌صورت فایل جداگانه Import می‌کند.

Qwen3-1.7B فعلاً Baseline Prototype است و انتخاب قطعی محصول نهایی نیست. معماری باید امکان تعویض مدل و Runtime را بدون بازطراحی اساسی UI و Core فراهم کند.

جزئیات مدل و Prototype: [`PROTOTYPE_SPEC.md`](docs/PROTOTYPE_SPEC.md)

---

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

AI Core باید مسئول منطق Local AI، Model Management، Inference، Agent، Action Protocol، Executor، Verification و Performance/Observability باشد؛ UI فقط مصرف‌کننده قابلیت‌ها و نمایش‌دهنده State/Result واقعی باشد.

### چرا Core از UI جداست؟

- Chat تنها مصرف‌کننده AI نیست.
- قابلیت‌هایی مانند «بازنویسی توضیحات با AI» باید بتوانند مستقیماً از Core استفاده کنند.
- Chat و UI باید Operationهای مشترک داشته باشند.
- تعویض مدل، Quantization یا Runtime نباید UI را مجبور به بازنویسی کند.
- Core باید مستقل تست و Regression شود.
- مهاجرت به WooGit باید استخراج Core باشد، نه کپی کل اپ.

جزئیات: [`ARCHITECTURE.md`](docs/ARCHITECTURE.md)

---

## 4. Model Management

کاربر باید بتواند Model را:

- Import کند.
- Validate کند.
- Load کند.
- Unload کند.
- مشاهده و مدیریت کند.

مدل نباید در کد hardcode شود و مسیر فایل باید با Android Storage Access Framework مدیریت شود.

---

## 5. Local Chat

پس از Load مدل:

- پیام کاربر دریافت شود.
- Inference کاملاً روی دستگاه اجرا شود.
- پاسخ به‌صورت Streaming واقعی نمایش داده شود.
- Generation واقعاً قابل Stop باشد.
- خطاهای واقعی نمایش داده شوند.
- وضعیت مدل و Generation قابل مشاهده باشد.

برای Inference نباید وابستگی اجباری به Cloud یا API خارجی وجود داشته باشد.

جزئیات: [`PROTOTYPE_SPEC.md`](docs/PROTOTYPE_SPEC.md)

---

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

هر تنظیمی فقط زمانی باید در UI فعال معرفی شود که Runtime واقعاً آن را اعمال کند.

---

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

Agent باید Multi-Step باشد و Maximum Agent Steps توسط کاربر قابل تنظیم باشد. هیچ عدد ثابت اجباری به‌عنوان تنظیم کاربر تعریف نمی‌شود. Runtime می‌تواند Hard Safety Limit مستقل برای جلوگیری از Loop غیرعادی داشته باشد.

جزئیات State: [`TASK_STATE_MACHINE.md`](docs/TASK_STATE_MACHINE.md)

---

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

`delete_file` نیازمند Confirmation اجباری کاربر است.

Web Search، HTTP Request و Actionهای وابسته به Cloud فعلاً خارج از Prototype هستند.

قرارداد Actionها: [`ACTION_PROTOCOL.md`](docs/ACTION_PROTOCOL.md)

---

## 9. Action واقعی و Verification

مدل نباید صرفاً بگوید Action انجام شده است. اجرای واقعی باید قابل اثبات باشد.

**Executor انجام می‌دهد → Verifier ثابت می‌کند → Agent نتیجه را تفسیر می‌کند → UI شواهد را نشان می‌دهد.**

برای Actionهای State-changing در صورت امکان وضعیت Before و After بررسی می‌شود.

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

جزئیات: [`ACTION_EXECUTION_VERIFICATION.md`](docs/ACTION_EXECUTION_VERIFICATION.md)

---

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

---

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

جزئیات: [`AI_WORKSPACE.md`](docs/AI_WORKSPACE.md)

---

## 12. Task State Machine

Stateهای اصلی Task:

```text
IDLE
 ↓
PLANNING
 ↓
WAITING_FOR_CONFIRMATION
 ↓
EXECUTING
 ↓
VERIFYING
 ↓
COMPLETED
```

و حالت‌های پایانی/خطا:

```text
FAILED
CANCELLED
BLOCKED
TIMEOUT
STEP_LIMIT_REACHED
```

Stop Generation، Cancel Task و Cancellation یک Action درحال اجرا مفاهیم جدا هستند و باید State واقعی نمایش داده شود.

جزئیات: [`TASK_STATE_MACHINE.md`](docs/TASK_STATE_MACHINE.md)

---

## 13. Security Model

هر Action باید از نظر امنیتی مشخص کند:

- Risk Level
- Read Only / State-changing
- Permission
- Confirmation
- Reversible بودن در صورت وجود

**Permission** تعیین می‌کند Action اصولاً مجاز است یا نه.

**Confirmation** تعیین می‌کند Action مجاز، قبل از اجرای واقعی نیاز به تأیید کاربر دارد یا نه.

مسیر اعتماد:

```text
Model Output
   ↓ untrusted
Parser
   ↓
Validator
   ↓
Permission
   ↓
Confirmation
   ↓
Executor
   ↓
Verifier
```

جزئیات: [`SECURITY_MODEL.md`](docs/SECURITY_MODEL.md)

---

## 14. Performance & Observability

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
- Network Usage در حد اطلاعاتی که Android/Runtime ارائه می‌کند

### History / Reporting

- ذخیره نتایج قبلی
- مقایسه Performance
- Export گزارش

جزئیات: [`PERFORMANCE_METRICS.md`](docs/PERFORMANCE_METRICS.md)

---

## 15. Visibility قابل تنظیم

تمام Metricها باید اندازه‌گیری شوند، اما نمایش هر Metric باید **کاملاً مستقل** قابل روشن/خاموش‌شدن باشد.

مثلاً:

```kotlin
const val SHOW_CPU = false
```

این تغییر باید فقط نمایش CPU را خاموش کند؛ Measurement نباید غیرفعال شود.

اگر Metric واقعاً قابل اندازه‌گیری نباشد، مقدار ساختگی مجاز نیست و باید `Unavailable` یا رفتار Visibility تعریف‌شده نمایش داده شود.

جزئیات: [`PERFORMANCE_METRICS.md`](docs/PERFORMANCE_METRICS.md)

---

## 16. Network & Offline

- Network Usage واقعی مانیتور شود.
- Network Request/Connectionهای واقعی و مصرف شبکه، در حد اطلاعاتی که Android/Runtime ارائه می‌کند، ثبت شوند.
- Network information در Performance/Debug قابل مشاهده باشد.
- Offline با قطع واقعی اینترنت تست می‌شود.
- Firewall، DNS، Fresh Install و سناریوهای پیچیده خارج از Requirement هستند.

مرز داده و Privacy: [`DATA_AND_PRIVACY.md`](docs/DATA_AND_PRIVACY.md)

Metricهای Network: [`PERFORMANCE_METRICS.md`](docs/PERFORMANCE_METRICS.md)

---

## 17. Data & Privacy Boundary

اصل پروژه Local-first است:

```text
DEVICE
├── Model Files
├── Chat / Context
├── Workspace State
├── Action Logs
├── Performance Metrics
└── Tool Results
        ↓
      AI Core
```

Inference و داده‌های موردنیاز آن باید Local باشند و وابستگی اجباری به Cloud وجود نداشته باشد.

هر قابلیت Network آینده باید مشخص کند چه داده‌ای، به کجا و برای چه هدفی ارسال می‌شود و آیا قابل خاموش‌کردن است یا نه.

Logهای Debug نباید بدون نیاز Secrets، credentialها یا داده حساس را ذخیره کنند.

جزئیات: [`DATA_AND_PRIVACY.md`](docs/DATA_AND_PRIVACY.md)

---

## 18. Test Matrix

تست‌ها باید هم مسیر موفق و هم مسیر شکست را پوشش دهند.

مهم‌ترین سناریوها:

- Import و Validate مدل
- Chat فارسی
- Stop Generation
- Action موفق
- JSON نامعتبر
- Action ناشناخته
- Permission denied
- Confirmation rejected
- Execution failure
- Verification failure
- Multi-step Agent
- Retry
- Step Limit
- Cancellation
- Workspace evidence
- Before/After
- Performance metrics
- Offline
- Network usage
- Persistence
- Export

هیچ تستی نباید فقط بر اساس Final Answer مدل موفق اعلام شود.

جزئیات: [`TEST_MATRIX.md`](docs/TEST_MATRIX.md)

---

## 19. مهاجرت به WooGit

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

هم Chat و هم UI باید بتوانند از Operationهای مشترک استفاده کنند.

برای عملیات حساس مانند تغییر قیمت، Validation و در صورت نیاز Preview/Confirmation لازم است.

جزئیات معماری: [`ARCHITECTURE.md`](docs/ARCHITECTURE.md)

جزئیات Workspace: [`AI_WORKSPACE.md`](docs/AI_WORKSPACE.md)

---

## 20. تصمیم درباره انتقال

برای Prototype هیچ Threshold یا امتیاز عددی از پیش تعیین‌شده‌ای برای اعلام موفقیت یا انتقال وجود ندارد.

Prototype باید شواهد واقعی ارائه کند و **تصمیم نهایی درباره کیفیت Prototype و مناسب بودن انتقال به WooGit با کاربر است**.

---

## 21. تست‌های پایه

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

---

## 22. Implementation Plan

مسیر پیشنهادی پیاده‌سازی:

```text
Foundation
   ↓
Chat
   ↓
Action System
   ↓
Agent
   ↓
Workspace
   ↓
Observability
   ↓
QA
   ↓
WooGit extraction
```

هر Phase باید Build/CI سالم، قابلیت واقعی، تست مسیر خطا، Trace/Result قابل مشاهده و مستندات به‌روز داشته باشد.

جزئیات: [`IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md)

---

## 23. محدوده خارج از Prototype

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

---

## 24. اصول توسعه

هر تغییر باید:

1. با هدف Prototype مرتبط باشد.
2. Build را خراب نکند.
3. قابلیت قبلی را بدون دلیل حذف نکند.
4. نتیجه واقعی تست را گزارش کند.
5. Mock یا گزارش جعلی ایجاد نکند.
6. با Specification و Contractهای پروژه سازگار باشد.
7. مستندات مربوطه را هم‌زمان به‌روز نگه دارد.

---

## 25. وضعیت

**Baseline Model:** Qwen3-1.7B  
**Format:** GGUF  
**Initial Quantization:** Q4_K_M  
**Runtime:** llama.cpp  
**Purpose:** Android Local AI + Agent Feasibility Test  
**Architecture Direction:** AI Core جدا از UI + Chat + AI Workspace + Action Verification  
**Future Target:** امکان انتقال کنترل‌شده AI Core به WooGit

برای شروع مطالعه از [`PROTOTYPE_SPEC.md`](docs/PROTOTYPE_SPEC.md) استفاده کنید؛ سپس معماری و Contractهای اجرایی را بخوانید.
