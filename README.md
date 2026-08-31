# AI Chat Test

> **Prototype / Technical Feasibility Test برای Local AI روی Android و Agent قابل‌اعتماد**

این repository یک محیط آزمایشی برای بررسی فنی اجرای هوش مصنوعی کاملاً Local روی Android است؛ نه محصول نهایی.

هدف اصلی این است که قبل از انتقال قابلیت Local AI به **WooGit**، اجرای مدل، Chat فارسی، Streaming، Performance، Agent، Actionهای واقعی، Verification و یک **AI Workspace** روی دستگاه واقعی آزمایش و ارزیابی شوند.

## فهرست مستندات

README نقشه کلی پروژه است. جزئیات هر بخش در سند تخصصی خودش نگهداری می‌شود:

| سند | موضوع |
|---|---|
| [`PROTOTYPE_SPEC.md`](docs/PROTOTYPE_SPEC.md) | Specification اصلی و تصمیم‌های نهایی Prototype |
| [`DECISIONS_21_30.md`](docs/DECISIONS_21_30.md) | خلاصه یکدست تصمیم‌های نهایی ۲۱ تا ۳۰ |
| [`ARCHITECTURE.md`](docs/ARCHITECTURE.md) | معماری Core/UI، مرزبندی لایه‌ها، Adapterها و مسیر معماری WooGit |
| [`ACTION_PROTOCOL.md`](docs/ACTION_PROTOCOL.md) | قرارداد ActionRequest/ToolResult، Schema، Lifecycle و Errorها |
| [`ACTION_REGISTRY.md`](docs/ACTION_REGISTRY.md) | Registry دسته‌بندی‌شده Actionها و Executor/Adapterهای قابل توسعه |
| [`ACTION_EXECUTION_VERIFICATION.md`](docs/ACTION_EXECUTION_VERIFICATION.md) | اجرای واقعی Action، Executor، Verifier و شواهد Before/After |
| [`TASK_STATE_MACHINE.md`](docs/TASK_STATE_MACHINE.md) | State Machine مربوط به Task، Agent، Execution و Cancellation |
| [`RECOVERY_FAILURE.md`](docs/RECOVERY_FAILURE.md) | Checkpoint، Recovery Policy، Idempotency و Verification در خطا |
| [`SECURITY_MODEL.md`](docs/SECURITY_MODEL.md) | Permission، Confirmation، Risk Level و مرز اعتماد Actionها |
| [`DATA_AND_PRIVACY.md`](docs/DATA_AND_PRIVACY.md) | مرز داده Local، Network، Storage، Logging و Privacy |
| [`AI_WORKSPACE.md`](docs/AI_WORKSPACE.md) | طراحی AI Workspace / میز کار و تعامل آن با Core و Action System |
| [`PERFORMANCE_METRICS.md`](docs/PERFORMANCE_METRICS.md) | Performance، Network، History، Export و Visibility مستقل Metricها |
| [`WOOGIT_INTEGRATION.md`](docs/WOOGIT_INTEGRATION.md) | Adapter + Capability Contract برای مهاجرت به WooGit |
| [`PLATFORM_RELEASE_CI.md`](docs/PLATFORM_RELEASE_CI.md) | Platform، Compatibility، CI و Release Acceptance |
| [`TEST_MATRIX.md`](docs/TEST_MATRIX.md) | ماتریس تست، سناریوهای موفق/ناموفق و Benchmark Protocol |
| [`IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) | مسیر مرحله‌ای تبدیل Specification به پیاده‌سازی و سپس مهاجرت به WooGit |
| [`QUESTION_BANK.md`](docs/QUESTION_BANK.md) | Question Bank و وضعیت تصمیم‌های ثبت‌شده پروژه |

### ترتیب پیشنهادی مطالعه

```text
README
  ↓
PROTOTYPE_SPEC
  ↓
DECISIONS_21_30
  ↓
ARCHITECTURE
  ├── ACTION_PROTOCOL
  ├── ACTION_REGISTRY
  ├── TASK_STATE_MACHINE
  ├── RECOVERY_FAILURE
  └── SECURITY_MODEL
  ↓
ACTION_EXECUTION_VERIFICATION
  ↓
AI_WORKSPACE
  ↓
WOOGIT_INTEGRATION
  ↓
PERFORMANCE_METRICS + DATA_AND_PRIVACY
  ↓
TEST_MATRIX + PLATFORM_RELEASE_CI
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

AI Core باید منطق Local AI، Model Management، Inference، Agent، Action Protocol و Observability را مستقل از UI ارائه کند. UI فقط مصرف‌کننده قابلیت‌ها و نمایش‌دهنده State/Result واقعی است.

Chat تنها مصرف‌کننده AI نیست؛ قابلیت‌هایی مانند «بازنویسی توضیحات با AI» یا تغییر قیمت باید بتوانند از همان Core استفاده کنند. تعویض Model، Quantization یا Runtime نباید UI را مجبور به بازنویسی کند و مهاجرت به WooGit باید استخراج Core و Contractها باشد، نه کپی کل اپ.

جزئیات: [`ARCHITECTURE.md`](docs/ARCHITECTURE.md)

---

## 4. Model Management

کاربر باید بتواند Model را Import، Validate، Load، Unload و مدیریت کند. مدل نباید hardcode شود و مسیر فایل باید با Android Storage Access Framework مدیریت شود.

---

## 5. Local Chat

پس از Load مدل:

- پیام کاربر دریافت شود.
- Inference کاملاً روی دستگاه اجرا شود.
- پاسخ به‌صورت Streaming واقعی نمایش داده شود.
- Generation واقعاً قابل Stop باشد.
- خطاهای واقعی نمایش داده شوند.
- وضعیت مدل و Generation قابل مشاهده باشد.

برای Inference وابستگی اجباری به Cloud یا API خارجی وجود ندارد.

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

هر تنظیم فقط زمانی در UI فعال معرفی شود که Runtime واقعاً آن را اعمال کند.

---

## 7. Agent و Context

Agent باید واقعی و Multi-Step باشد و Maximum Agent Steps توسط کاربر قابل تنظیم باشد. هیچ عدد ثابت اجباری به‌عنوان تنظیم کاربر تعریف نمی‌شود و Runtime می‌تواند Hard Safety Limit مستقل داشته باشد.

Context به‌صورت ترکیبی ساخته می‌شود:

```text
System Context
      +
Persistent Task Context
      +
Conversation Summary
      +
Recent Messages (user-configurable)
      +
Workspace Context (query/select)
```

Recent Messages قابل تنظیم است و عدد ثابتی در معماری فرض نمی‌شود. پیام‌های قدیمی‌تر به Summary تبدیل می‌شوند. Persistent Task Context هدف، کار فعلی، Intent و اطلاعات مهم را مستقل از Chat History نگه می‌دارد. Workspace نیز منبع Context قابل Query/Select است.

---

## 8. Action System

Actionها با Contract ساختاریافته مدیریت می‌شوند و از طریق `ActionRegistry` مرکزی کشف و دسته‌بندی می‌شوند. Registry خودش Executor نیست و هر Category می‌تواند در آینده Executor/Adapterهای بیشتری داشته باشد.

نمونه Actionهای Prototype:

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

جزئیات: [`ACTION_PROTOCOL.md`](docs/ACTION_PROTOCOL.md) و [`ACTION_REGISTRY.md`](docs/ACTION_REGISTRY.md)

---

## 9. Action واقعی و Verification

مدل نباید صرفاً بگوید Action انجام شده است.

**Executor انجام می‌دهد → Verifier ثابت می‌کند → Agent نتیجه را تفسیر می‌کند → UI شواهد را نشان می‌دهد.**

برای Actionهای State-changing در صورت امکان Before/After بررسی می‌شود. اگر Verification شکست بخورد، Agent نباید موفقیت را اعلام کند.

جزئیات: [`ACTION_EXECUTION_VERIFICATION.md`](docs/ACTION_EXECUTION_VERIFICATION.md)

---

## 10. Sensitive Actions و Final Approval

برای Actionهای حساس:

```text
Prepare → Preview / Snapshot → Validate → Final Approval → Execute → Verify → Result
```

`Final Approval` اجازه اجرای همان عملیات آماده‌شده و Snapshot‌شده را می‌دهد؛ تأیید نباید به معنی اجازه تصمیم‌گیری مجدد AI باشد. Approval به معنی Success نیست و Success فقط پس از Execution و Verification اعلام می‌شود.

اگر State واقعی قبل از Execute تغییر کند، Snapshot باید دوباره Validate/Prepare شود و در صورت نیاز تأیید جدید گرفته شود.

---

## 11. AI Workspace — میز کار

Prototype علاوه بر Chat یک **AI Workspace** تعاملی دارد برای:

- Task و Goal
- Actionها و Tool Resultها
- Verification
- Preview و Before/After
- خروجی‌های چندمرحله‌ای Agent
- فایل‌ها و Artifactها
- Accept / Edit / Reject / Approve / Cancel و سایر Interactionها

Workspace Executor نیست. Interactionهای آن Command/Intent به Core می‌فرستند و اجرای واقعی از مسیر Core/Action System انجام می‌شود.

این ساختار برای قابلیت‌های آینده WooGit مانند تغییر قیمت، بازنویسی توضیحات، تولید عنوان و اصلاح محتوا آماده است.

جزئیات: [`AI_WORKSPACE.md`](docs/AI_WORKSPACE.md)

---

## 12. Task State Machine و Recovery

Stateهای اصلی Task:

```text
IDLE → PLANNING → WAITING_FOR_CONFIRMATION → EXECUTING → VERIFYING → COMPLETED
```

حالت‌های خطا/پایان شامل `FAILED`، `CANCELLED`، `BLOCKED`، `TIMEOUT` و `STEP_LIMIT_REACHED` هستند.

Recovery بر پایه:

```text
State Machine + Checkpoint + Recovery Policy + Idempotency + Verification
```

Stateهای مهم Persistent هستند. Crash یا Restart نباید باعث اجرای دوباره کورکورانه شود. در وضعیت نامشخص، ابتدا State واقعی Verify می‌شود و سپس درباره Retry تصمیم‌گیری می‌شود. Recovery نباید Final Approval را دور بزند.

جزئیات: [`TASK_STATE_MACHINE.md`](docs/TASK_STATE_MACHINE.md) و [`RECOVERY_FAILURE.md`](docs/RECOVERY_FAILURE.md)

---

## 13. Security Model

هر Action باید Risk Level، Read Only/State-changing، Permission، Confirmation و Reversible بودن در صورت وجود را مشخص کند.

مسیر اعتماد:

```text
Model Output → Parser → Validator → Permission → Confirmation → Executor → Verifier
```

Permission و Confirmation دو مفهوم مستقل هستند.

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

---

## 15. Visibility مستقل

تمام Metricها باید اندازه‌گیری شوند، اما نمایش هر Metric باید **کاملاً مستقل و قابل تنظیم** باشد. تغییر نمایش یک Metric باید با تغییر بسیار کوچک در یک محل مرکزی کد انجام شود و خاموش‌کردن Visibility نباید Measurement را غیرفعال کند.

اگر Metric واقعاً قابل اندازه‌گیری نباشد، `Unavailable` نمایش داده شود و مقدار ساختگی مجاز نیست.

---

## 16. Network & Offline

- Network Usage واقعی مانیتور شود.
- Network Request/Connectionهای واقعی و مصرف شبکه، در حد اطلاعاتی که Android/Runtime ارائه می‌کند، ثبت شوند.
- Network information در Performance/Debug قابل مشاهده باشد.
- Offline با قطع واقعی اینترنت توسط کاربر تست می‌شود.
- Firewall، DNS، Fresh Install و سناریوهای پیچیده خارج از Requirement هستند.

---

## 17. Data & Privacy Boundary

اصل پروژه Local-first است. Inference و داده‌های موردنیاز آن باید Local باشند و وابستگی اجباری به Cloud وجود نداشته باشد.

هر قابلیت Network آینده باید مشخص کند چه داده‌ای، به کجا و برای چه هدفی ارسال می‌شود و آیا قابل خاموش‌کردن است یا نه.

Logهای Debug نباید بدون نیاز Secrets، credentialها یا داده حساس را ذخیره کنند.

---

## 18. Test Matrix

تست‌ها باید مسیر موفق و شکست را پوشش دهند؛ از جمله Model Import/Validate، Chat فارسی، Stop Generation، Action موفق، JSON نامعتبر، Action ناشناخته، Permission، Confirmation، Execution/Verification Failure، Multi-step Agent، Retry، Step Limit، Cancellation، Workspace Evidence، Before/After، Performance، Offline، Network Usage، Persistence و Export.

هیچ تستی نباید فقط بر اساس Final Answer مدل موفق اعلام شود.

در فاز فعلی، تست‌های سخت‌گیرانه Action/Recovery زیرساخت و سناریو دارند اما تا مرحله آمادگی نهایی Release Gate اجباری نیستند.

---

## 19. WooGit Integration و Migration

هدف انتقال کل `ai-chat-test` به WooGit نیست. AI Core و Contractها باید قابل استخراج و مصرف مستقل باشند.

اتصال به WooGit با **Adapter + Capability Contract** انجام می‌شود:

```text
AI Core
  ↓
Action Contract
  ↓
Action Registry
  ↓
Capability Contract
  ↓
WooGit Adapter
  ↓
WooGit API / Service
```

محیط Prototype می‌تواند Test Adapter داشته باشد و WooGit Adapter بعداً بدون تغییر Action Contract اضافه شود. Capabilityها مشخص می‌کنند محیط فعلی چه Actionهایی را پشتیبانی می‌کند.

جزئیات: [`WOOGIT_INTEGRATION.md`](docs/WOOGIT_INTEGRATION.md)

---

## 20. معیار موفقیت و تصمیم انتقال

هیچ Threshold، امتیاز یا معیار عددی از پیش تعیین‌شده‌ای برای اعلام موفقیت Prototype یا انتقال به WooGit وجود ندارد.

Prototype باید شواهد واقعی ارائه کند و **تصمیم نهایی درباره کیفیت اپ و مناسب بودن انتقال شخصاً توسط کاربر انجام می‌شود**.

---

## 21. Platform / Release / CI

Platform، ABI و محدودیت‌های Runtime/Model بر اساس نیاز واقعی و قابلیت واقعی پروژه تعیین می‌شوند و عدد ثابت غیرضروری از ابتدا تحمیل نمی‌شود.

یک Compatibility Contract مرکزی باید سازگاری Model، Runtime، ABI و منابع دستگاه را پیش از Load/Execution بررسی کند.

CI پایه از همین حالا شامل Build، Unit Tests، Integration Tests، Architecture Checks، Static Analysis/Lint و Package/Validation است.

تست‌های سخت‌گیرانه Action و Recovery از نظر طراحی و زیرساخت آماده‌اند، اما **فعلاً Release Gate اجباری نیستند**. وقتی پروژه به آخرین سطح آمادگی برسد و کاربر کیفیت خود اپ را تأیید کند، این تست‌ها به Hard Gateهای CI/Release تبدیل می‌شوند.

جزئیات: [`PLATFORM_RELEASE_CI.md`](docs/PLATFORM_RELEASE_CI.md)

---

## 22. Decisions 21–30

تصمیم‌های ۲۱ تا ۳۰ به‌صورت یکپارچه ثبت شده‌اند:

- Runtime مستقل و Adapter-based
- Context ترکیبی و قابل تنظیم
- Workspace تعاملی و غیر Executor
- Observability کامل در توسعه
- Persistence ترکیبی
- Central Settings Core
- Action Registry دسته‌بندی‌شده و قابل توسعه
- WooGit Adapter + Capability Contract
- Recovery با Checkpoint/Idempotency/Verification
- CI پایه از ابتدا و Hard Gate نهایی پس از تأیید کیفیت اپ

جزئیات کامل: [`DECISIONS_21_30.md`](docs/DECISIONS_21_30.md)

---

## 23. تست‌های پایه

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

## 24. Implementation Plan

مسیر پیشنهادی پیاده‌سازی:

```text
Foundation → Chat → Action System → Agent → Workspace → Observability → QA → WooGit extraction
```

هر Phase باید Build/CI سالم، قابلیت واقعی، تست مسیر خطا، Trace/Result قابل مشاهده و مستندات به‌روز داشته باشد.

جزئیات: [`IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md)

---

## 25. محدوده خارج از Prototype

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

## 26. اصول توسعه

هر تغییر باید:

1. با هدف Prototype مرتبط باشد.
2. Build را خراب نکند.
3. قابلیت قبلی را بدون دلیل حذف نکند.
4. نتیجه واقعی تست را گزارش کند.
5. Mock یا گزارش جعلی ایجاد نکند.
6. با Specification و Contractهای پروژه سازگار باشد.
7. مستندات مربوطه را هم‌زمان به‌روز نگه دارد.

---

## 27. وضعیت

**Baseline Model:** Qwen3-1.7B  
**Format:** GGUF  
**Initial Quantization:** Q4_K_M  
**Runtime:** llama.cpp  
**Purpose:** Android Local AI + Agent Feasibility Test  
**Architecture Direction:** AI Core جدا از UI + Chat + AI Workspace + Action Registry + Action Verification + Recovery  
**Future Target:** امکان انتقال کنترل‌شده AI Core به WooGit با Adapter + Capability Contract

برای شروع مطالعه از [`PROTOTYPE_SPEC.md`](docs/PROTOTYPE_SPEC.md) و سپس [`DECISIONS_21_30.md`](docs/DECISIONS_21_30.md) استفاده کنید؛ سپس معماری و Contractهای اجرایی را بخوانید.
