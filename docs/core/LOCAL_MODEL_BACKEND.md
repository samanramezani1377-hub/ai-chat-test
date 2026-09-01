# درخت استاندارد Backend مدل هوش مصنوعی محلی

**وضعیت:** 🔒 قرارداد اجرایی قبل از پیاده‌سازی UI  
**دامنه:** Core / Runtime / Model Management  
**اصل:** این بخش کاملاً مستقل از UI طراحی و کدنویسی می‌شود.

## ۱. درخت مرجع کامل

```text
Local AI Model Backend
│
├── 1. Model Management
│   ├── Import
│   │   ├── دریافت URI / فایل انتخاب‌شده
│   │   ├── بررسی دسترسی خواندن
│   │   ├── ایجاد Model ID
│   │   ├── انتقال به Storage مدیریت‌شده در صورت نیاز
│   │   ├── جلوگیری از Import ناقص
│   │   ├── Cancellation
│   │   └── Import Result
│   │
│   ├── Validation
│   │   ├── File existence / accessibility
│   │   ├── File readability
│   │   ├── Format detection
│   │   ├── GGUF header validation
│   │   ├── GGUF structure validation
│   │   ├── Metadata validation
│   │   ├── Architecture validation
│   │   ├── Real quantization detection
│   │   └── Runtime compatibility
│   │
│   ├── Inspection
│   │   └── GGUF Metadata Reader
│   │       ├── model architecture
│   │       ├── model name
│   │       ├── quantization
│   │       ├── context information
│   │       ├── tensor/file information
│   │       ├── tokenizer metadata
│   │       └── سایر metadataهای موردنیاز Runtime
│   │
│   ├── Registry / Repository
│   │   ├── register
│   │   ├── get
│   │   ├── list
│   │   ├── get active
│   │   ├── unregister
│   │   └── persistent metadata/state
│   │
│   └── Lifecycle Manager
│       ├── NotImported
│       ├── Importing
│       ├── Validating
│       ├── Imported
│       ├── Ready
│       ├── Loading
│       ├── Active
│       ├── Unloading
│       ├── Invalid
│       └── Failed
│
├── 2. Model Descriptor
│   ├── id
│   ├── displayName
│   ├── source / managed path
│   ├── format
│   ├── quantization
│   ├── architecture
│   ├── size
│   ├── metadata
│   ├── validation status
│   └── runtime compatibility
│
├── 3. Runtime
│   ├── Runtime Contract
│   │   ├── load
│   │   ├── unload
│   │   ├── generate
│   │   ├── stopGeneration
│   │   └── runtimeInfo
│   │
│   ├── Runtime Adapter
│   │   └── ModelRuntime / RuntimeAdapter boundary
│   │
│   ├── GGUF Backend
│   │   ├── model loading
│   │   ├── tensor access
│   │   ├── tokenizer/runtime setup
│   │   ├── inference
│   │   ├── stop generation
│   │   ├── unload
│   │   ├── memory management
│   │   └── resource cleanup
│   │
│   └── Runtime State
│       ├── unavailable
│       ├── idle
│       ├── loading
│       ├── ready
│       ├── generating
│       ├── unloading
│       └── failed
│
├── 4. Model Activation
│   ├── activate
│   ├── deactivate
│   ├── active model
│   ├── previous model هنگام تعویض
│   ├── runtime state
│   └── activation result
│
├── 5. Storage & File Lifecycle
│   ├── selected source
│   ├── managed model file
│   ├── persistence
│   ├── integrity
│   ├── cleanup
│   ├── حذف مدل
│   └── جلوگیری از استفاده از مسیر موقت به‌عنوان مسیر دائمی
│
├── 6. Concurrency & Cancellation
│   ├── operation serialization / policy
│   ├── duplicate import protection
│   ├── duplicate load protection
│   ├── cancellation propagation
│   ├── cancellation cleanup
│   ├── race prevention
│   └── safe active-model transitions
│
├── 7. Error Model
│   ├── FileAccessError
│   ├── UnsupportedFormat
│   ├── InvalidModel
│   ├── InvalidMetadata
│   ├── UnsupportedQuantization
│   ├── RuntimeUnavailable
│   ├── LoadFailed
│   ├── OutOfMemory
│   ├── ImportCancelled
│   ├── StorageError
│   └── InferenceError
│
├── 8. Observability / Action Trace
│   ├── Import
│   ├── Validation
│   ├── Inspection
│   ├── Registration
│   ├── Activation
│   ├── Load
│   ├── Inference
│   └── Trace Payload
│       ├── operationId
│       ├── executionId
│       ├── traceId
│       ├── modelId
│       ├── stage
│       ├── start/end
│       ├── duration واقعی در صورت اندازه‌گیری
│       ├── status
│       ├── errorCode
│       └── metadata غیرحساس
│
├── 9. Capabilities
│   ├── import
│   ├── inspect
│   ├── validate
│   ├── list
│   ├── get
│   ├── activate
│   ├── deactivate
│   ├── unload
│   ├── delete
│   ├── runtimeStatus
│   ├── runtimeInfo
│   └── inference
│
├── 10. Backend Tests
│   ├── valid GGUF import
│   ├── non-GGUF import
│   ├── corrupt file
│   ├── incomplete metadata
│   ├── unsupported quantization
│   ├── valid Q6_K
│   ├── duplicate model
│   ├── import cancellation
│   ├── storage failure
│   ├── successful load
│   ├── failed load
│   ├── unload
│   ├── activation
│   ├── model replacement
│   ├── inactive model deletion
│   ├── active model deletion policy
│   ├── generate
│   ├── stop generation
│   ├── runtime unavailable
│   └── memory/resource failures
│
└── 11. UI Integration Contract
    ├── Model List
    │   ├── list models
    │   └── item state
    │
    ├── Current Model
    │   ├── active model
    │   └── active state
    │
    ├── Import Model
    │   ├── start
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
    │   ├── architecture
    │   ├── compatibility
    │   └── runtime state
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
    ├── Reactive State
    │   ├── models
    │   ├── currentModel
    │   ├── importState
    │   ├── validationState
    │   ├── runtimeState
    │   ├── activeOperation
    │   └── error
    │
    ├── Import Result
    │   ├── success/failure
    │   ├── model descriptor
    │   ├── operation/execution id
    │   ├── validation result
    │   └── error
    │
    ├── Diagnostics
    │   ├── execution id
    │   ├── trace id
    │   ├── error code
    │   ├── error message
    │   └── report navigation target
    │
    └── Runtime Events
        ├── MODEL_IMPORT_STARTED
        ├── MODEL_VALIDATION_STARTED
        ├── MODEL_VALIDATION_COMPLETED
        ├── MODEL_IMPORTED
        ├── MODEL_IMPORT_FAILED
        ├── MODEL_LOAD_STARTED
        ├── MODEL_LOADED
        ├── MODEL_LOAD_FAILED
        ├── MODEL_UNLOADED
        ├── MODEL_ACTIVATED
        └── MODEL_DEACTIVATED
```

## ۲. قراردادهای اجرایی

### استقلال Backend از UI

گره‌های `1` تا `10` باید بدون وابستگی به Composable، Screen یا ViewModel خاص طراحی و کدنویسی شوند. UI فقط مصرف‌کننده قرارداد گره `11` است.

### جریان استاندارد

```text
Android File Picker
      ↓
File URI
      ↓
Model Import
      ↓
Validation
      ↓
GGUF Inspection
      ↓
Registration
      ↓
Ready
      ↓
Activation
      ↓
Runtime Load
      ↓
Inference
```

### مدل هدف اولیه

**Qwen3-1.7B · GGUF · Q6_K**

Backend باید مشخصات واقعی فایل را از Validation و Metadata استخراج کند؛ نام فایل به‌تنهایی معتبر نیست.

### صحت و تست

موفقیت Import، Load و Inference نباید با Mock یا Placeholder جعل شود. Mock فقط برای Boundaryهای ضروری Unit Test مجاز است.

## ۳. قرارداد اتصال به UI

این بخش باید **همزمان با طراحی و کدنویسی Backend** نگهداری شود و هر API، State، Event، Result یا Error جدیدی که UI باید مصرف کند در همین قسمت ثبت شود.

پس از تغییر این قرارداد:

```text
Backend Contract
      ↓
UI Integration Contract
      ↓
UI_TREE.md
      ↓
SCREENS.md
      ↓
LOCAL_MODEL_IMPORT_UI.md
```

تمام ارجاعات UI باید با قرارداد واقعی Backend هماهنگ بمانند.

## ۴. Definition of Done

Backend فقط زمانی کامل است که:

- تمام گره‌های `1` تا `10` با پیاده‌سازی واقعی تکمیل شده باشند.
- تست‌های لازم وجود داشته و نتیجه واقعی آن‌ها قابل مشاهده باشد.
- Trace و Error Contract کامل باشد.
- Q6_K در صورت پشتیبانی Runtime واقعاً Load شود.
- Import و Lifecycle واقعی باشند.
- گره `11` کامل و قابل مصرف توسط UI باشد.
- ارجاعات اسناد UI به این قرارداد برقرار و هماهنگ باشند.
