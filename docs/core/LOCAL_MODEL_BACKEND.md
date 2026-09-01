# قرارداد Backend مدل هوش مصنوعی محلی

**وضعیت:** 🔒 قرارداد اجرایی قبل از پیاده‌سازی UI
**دامنه:** Core / Runtime / Model Management
**UI:** عمداً مستقل از این بخش طراحی و پیاده‌سازی می‌شود.
**مرجع ساختاری UI:** [`../ui/UI_TREE.md`](../ui/UI_TREE.md)
**مرجع UI مدل:** [`../ui/LOCAL_MODEL_IMPORT_UI.md`](../ui/LOCAL_MODEL_IMPORT_UI.md)

---

## 1. هدف

این بخش تمام کمبودهای Backend لازم برای وارد کردن، اعتبارسنجی، نگهداری، مدیریت و اجرای یک مدل زبانی محلی را مشخص می‌کند.

مدل هدف اولیه:

**Qwen3-1.7B · GGUF · Q6_K**

این نام صرفاً انتخاب مدل هدف است؛ Backend باید مشخصات فایل واقعی را از metadata/validation استخراج کند و نباید صرفاً بر اساس نام فایل نتیجه‌گیری کند.

---

## 2. اصل معماری

این بخش باید **کاملاً جدا از UI** طراحی و کدنویسی شود.

```text
UI
 │
 │ قرارداد اتصال
 ▼
Model Management API
 │
 ├── Import
 ├── Validate
 ├── Inspect
 ├── Register
 ├── Activate / Deactivate
 ├── Delete
 └── Status
        │
        ▼
     Runtime
        │
        ▼
   GGUF Backend
        │
        ▼
 Qwen3-1.7B Q6_K
```

UI فقط انتخاب فایل، نمایش وضعیت و ارسال فرمان را انجام می‌دهد. Parse، validation، persistence، loading و inference مسئولیت Core/Runtime است.

---

## 3. کمبودهای فعلی که باید تکمیل شوند

### 3.1 Model Import Service

یک سرویس واقعی برای دریافت مدل از File URI/منبع انتخاب‌شده لازم است.

مسئولیت‌ها:

- دریافت ورودی فایل از لایه Android
- بررسی دسترسی خواندن
- ایجاد/ثبت شناسه یکتا برای مدل
- انتقال یا ثبت امن فایل در storage داخلی برنامه در صورت نیاز
- جلوگیری از ثبت ناقص
- مدیریت cancellation و failure
- برگرداندن نتیجه واقعی Import

### 3.2 Model Validator

Validator باید فایل را واقعاً بررسی کند:

- وجود و دسترسی فایل
- خوانایی فایل
- فرمت
- ساختار GGUF
- صحت header و metadata
- سازگاری version/metadata با Backend
- مشخصات معماری مدل
- quantization واقعی
- اندازه و محدودیت‌های Runtime

فقط نام فایل نباید مبنای validation باشد.

### 3.3 GGUF Metadata Reader

برای GGUF باید Reader واقعی اضافه شود تا metadata لازم برای `ModelDescriptor` استخراج شود، از جمله در صورت موجود بودن:

- model architecture
- model name
- quantization type
- context information
- tensor/file information
- tokenizer metadata
- سایر metadataهای لازم Runtime

### 3.4 Model Repository

یک Repository پایدار برای مدل‌های Import‌شده لازم است.

باید بتواند:

- ثبت مدل
- پیدا کردن مدل
- لیست مدل‌ها
- پیدا کردن مدل فعال
- حذف/Unregister
- نگهداری وضعیت مدل
- نگهداری metadata معتبر

را انجام دهد.

### 3.5 Model Manager

Manager باید چرخه عمر مدل را کنترل کند:

```text
NotImported
   ↓
Importing
   ↓
Validating
   ↓
Imported
   ↓
Ready
   ↓
Loading
   ↓
Loaded / Active
```

و مسیرهای خطا/لغو نیز باید واقعی باشند:

```text
Importing ──X──> Failed
Validating ──X──> Invalid
Loading ──X──> LoadFailed
```

### 3.6 Runtime Backend واقعی

`RuntimeAdapter` فعلی فقط مرز معماری است. باید Backend واقعی GGUF در پشت آن قرار گیرد.

Backend باید حداقل این قابلیت‌ها را فراهم کند:

- load model
- unload model
- generate
- stop generation
- runtime information
- وضعیت آماده/در حال Load/خطا
- مدیریت منابع
- آزادسازی حافظه

### 3.7 Inference Contract

Inference باید از Agent/Conversation جدا از جزئیات Backend باقی بماند.

```text
Agent
  ↓
ModelRuntime.generate()
  ↓
RuntimeAdapter
  ↓
GGUF Runtime
```

هیچ کد UI نباید مستقیماً Runtime Backend را صدا بزند.

### 3.8 Model Activation

فعال کردن مدل باید یک عملیات واقعی باشد.

باید مشخص باشد:

- مدل انتخاب‌شده
- مدل فعلی
- وضعیت Runtime
- موفقیت/شکست Activation
- مدل قبلی در صورت تعویض

نباید صرفاً با تغییر یک مقدار UI، مدل «فعال» فرض شود.

### 3.9 Storage و File Lifecycle

Backend باید مشخص کند فایل مدل کجا نگهداری می‌شود و چرخه عمر آن چیست:

```text
Selected URI
   ↓
Validated Source
   ↓
Managed Model File
   ↓
Registered Model
   ↓
Runtime Load
```

باید از ثبت مسیر موقت/غیرقابل‌دسترسی به‌عنوان مسیر دائمی جلوگیری شود.

### 3.10 Concurrency و Cancellation

Import و Load نباید باعث چند Load همزمان یا وضعیت race شوند.

لازم است:

- یک سیاست مشخص برای عملیات همزمان
- cancellation واقعی
- cleanup بعد از cancellation
- جلوگیری از duplicate registration
- جلوگیری از unload ناخواسته مدل فعال

وجود داشته باشد.

### 3.11 Error Model

تمام خطاها باید typed و قابل نگاشت باشند.

نمونه دسته‌ها:

- FileAccessError
- UnsupportedFormat
- InvalidModel
- InvalidMetadata
- UnsupportedQuantization
- RuntimeUnavailable
- LoadFailed
- OutOfMemory
- ImportCancelled
- StorageError
- InferenceError

UI متن فارسی را از Error Contract می‌گیرد؛ متن فنی و علت دقیق باید برای Diagnostics قابل دسترسی باشد.

### 3.12 Observability / Trace

Import، Validation، Registration، Activation، Load و Inference باید Execution/Trace واقعی تولید کنند تا در معماری فعلی observability قابل مشاهده باشند.

حداقل باید قابل ثبت باشد:

- operation id
- model id
- stage
- start/end
- duration در صورت اندازه‌گیری واقعی
- status
- error code در صورت خطا
- metadata غیرحساس لازم برای عیب‌یابی

---

## 4. قرارداد ModelDescriptor

`ModelDescriptor` باید مرجع اطلاعات مدل باشد و اطلاعات حداقلی زیر را پوشش دهد:

```text
ModelDescriptor
├── id
├── displayName
├── source / managed path
├── format
├── quantization
├── architecture
├── size
├── metadata
├── validation status
└── runtime compatibility
```

فیلدها باید فقط با داده واقعی پر شوند.

---

## 5. قرارداد وضعیت مدل

Backend باید وضعیت قابل مشاهده و پایدار داشته باشد:

```text
UNKNOWN
IMPORTED
VALIDATING
INVALID
READY
LOADING
ACTIVE
UNLOADING
FAILED
```

نام نهایی enumها باید با Domain واقعی پروژه هماهنگ شود، ولی semantics باید حفظ شود.

---

## 6. قابلیت‌های لازم Backend

قبل از شروع UI باید این Capabilityها وجود داشته باشند یا قرارداد دقیق و قابل پیاده‌سازی آن‌ها تثبیت شده باشد:

```text
ModelCapabilities
├── import
├── inspect
├── validate
├── list
├── get
├── activate
├── deactivate
├── unload
├── delete
├── runtimeStatus
├── runtimeInfo
└── inference
```

هر Capability باید نتیجه واقعی، State و Error قابل دسترسی داشته باشد.

---

## 7. تست‌های Backend

باید تست واقعی برای حداقل این سناریوها نوشته شود:

1. Import فایل معتبر GGUF
2. Import فایل غیر GGUF
3. فایل خراب
4. metadata ناقص
5. quantization پشتیبانی‌نشده
6. Q6_K معتبر
7. duplicate model
8. cancellation هنگام Import
9. failure هنگام Storage
10. Load موفق
11. Load ناموفق
12. unload
13. activation مدل
14. تعویض مدل فعال
15. حذف مدل غیرفعال
16. جلوگیری از حذف مدل فعال بدون سیاست مشخص
17. generate
18. stop generation
19. runtime unavailable
20. خطاهای منابع/حافظه

هیچ تستی نباید Mock موفقیت Runtime را جایگزین تست واقعی Backend کند؛ Mock فقط برای Boundaryهای ضروری و تست‌های unit مجاز است و نباید نتیجه قابلیت را جعل کند.

---

# 8. بخش اجباری: UI Integration Contract

این بخش باید **همزمان با طراحی و کدنویسی Backend** تکمیل شود و تا پایان Backend به‌عنوان قرارداد اتصال به UI نگهداری شود.

این قسمت برای جلوگیری از وضعیت «Backend تمام شد ولی UI نمی‌داند چه چیزی باید وصل کند» اجباری است.

## 8.1 چیزهایی که باید به UI متصل شوند

```text
UI Integration
├── Model List
│   ├── list models
│   └── model item state
│
├── Current Model
│   ├── get active model
│   └── active state
│
├── Import Model
│   ├── start import
│   ├── import state
│   ├── validation state
│   ├── result
│   ├── cancellation
│   └── error
│
├── Model Details
│   ├── metadata
│   ├── format
│   ├── quantization
│   ├── size
│   ├── compatibility
│   └── runtime status
│
├── Model Actions
│   ├── activate
│   ├── deactivate
│   ├── unload
│   └── delete
│
├── Runtime
│   ├── status
│   ├── loading state
│   ├── runtime info
│   └── capability state
│
└── Diagnostics
    ├── execution id
    ├── trace id
    ├── error code
    ├── error message
    └── report navigation target
```

## 8.2 Reactive State Contract

UI نباید وضعیت مدل را حدس بزند. Backend باید State قابل مشاهده ارائه کند.

نمونه:

```text
ModelUiState
├── models
├── currentModel
├── importState
├── validationState
├── runtimeState
├── activeOperation
└── error
```

State باید reactive باشد و UI بتواند تغییرات واقعی را observe کند.

## 8.3 Import Result Contract

نتیجه Import باید حداقل بتواند این موارد را به UI بدهد:

```text
ImportResult
├── success / failure
├── model descriptor در صورت موفقیت
├── operation/execution id
├── validation result
└── error در صورت شکست
```

## 8.4 Model Details Contract

UI باید بتواند بدون دسترسی مستقیم به فایل، جزئیات واقعی مدل را دریافت کند:

```text
ModelDetails
├── name
├── file name
├── format
├── quantization
├── size
├── architecture
├── validation
├── compatibility
└── runtime state
```

## 8.5 Diagnostics Navigation Contract

اگر Import/Validation/Load شکست خورد، UI باید شناسه گزارش متناظر را داشته باشد:

```text
Workspace / Settings
      ↓
مشاهده جزئیات
      ↓
Diagnostics
      ↓
همان Execution / Trace
```

این اتصال باید بر اساس ID واقعی باشد، نه route یا گزارش ساختگی.

## 8.6 Event/Trace Contract برای UI

برای نمایش وضعیت عملیات در UI، Backend باید Eventهای قابل مشاهده و پایدار داشته باشد؛ حداقل:

```text
MODEL_IMPORT_STARTED
MODEL_VALIDATION_STARTED
MODEL_VALIDATION_COMPLETED
MODEL_IMPORTED
MODEL_IMPORT_FAILED
MODEL_LOAD_STARTED
MODEL_LOADED
MODEL_LOAD_FAILED
MODEL_UNLOADED
MODEL_ACTIVATED
MODEL_DEACTIVATED
```

نام دقیق Eventها می‌تواند مطابق Event Bus پروژه نهایی شود، اما semantics و payload موردنیاز باید حفظ شود.

---

## 9. UI Mapping

مرجع UI باید این قرارداد را به‌صورت مستقیم مصرف کند:

```text
UI_TREE.md
   ↓
Settings
└── هوش مصنوعی
    └── مدل
        ├── مدل فعلی
        ├── Import Model
        └── مدیریت مدل‌ها
```

جزئیات بصری در `LOCAL_MODEL_IMPORT_UI.md` است؛ قرارداد داده/عملیات در این سند است.

---

## 10. Definition of Done برای Backend

این بخش زمانی تمام‌شده محسوب می‌شود که:

- Import واقعی کار کند.
- Validation واقعی GGUF کار کند.
- Metadata واقعی استخراج شود.
- ModelRepository پایدار باشد.
- ModelManager چرخه عمر را کنترل کند.
- GGUF Runtime واقعی متصل باشد.
- Q6_K در صورت پشتیبانی Backend واقعاً Load شود.
- Inference واقعی کار کند.
- Error Model کامل باشد.
- Cancellation و concurrency کنترل شده باشد.
- Trace/Observability واقعی ثبت شود.
- تست‌های لازم سبز باشند.
- هیچ قابلیت موفقیت ساختگی یا Mock جایگزین Runtime واقعی نشده باشد.
- **UI Integration Contract در همین سند کامل و نهایی شده باشد.**
- تمام ارجاعات UI به این قرارداد ثبت شده باشند.

تا قبل از تحقق موارد بالا، UI نباید قابلیت Import Model را به‌عنوان قابلیت عملیاتی کامل اعلام کند.

---

## 11. ارجاعات اجباری

این سند مرجع Backend قابلیت Local Model است.

اسناد UI باید به این سند ارجاع دهند:

- `docs/ui/UI_TREE.md`
- `docs/ui/SCREENS.md`
- `docs/ui/LOCAL_MODEL_IMPORT_UI.md`

اسناد Core/Architecture/Contracts نیز در صورت اشاره به Local Model باید به همین سند ارجاع دهند و قرارداد موازی نسازند.

**قاعده:** اگر Backend Contract تغییر کرد، بخش `UI Integration Contract` باید در همان تغییر به‌روزرسانی شود و سپس ارجاعات UI بررسی شوند.
