# درخت استاندارد Backend مدل هوش مصنوعی محلی

**وضعیت:** 🔒 قرارداد اجرایی قبل از پیاده‌سازی UI  
**دامنه:** Core / Runtime / Model Management  
**اصل:** این بخش کاملاً مستقل از UI طراحی و کدنویسی می‌شود.

## ۱. درخت مرجع کامل

```text
Local AI Model Backend
│
├── 1. Architecture
│   ├── Domain
│   ├── Model Management
│   ├── Runtime
│   ├── Storage
│   ├── Operations
│   ├── Errors
│   ├── Observability
│   ├── Resource Management
│   ├── Security
│   ├── Testing
│   └── UI Integration Boundary
│
├── 2. Model Management
│   │
│   ├── Import
│   │   ├── source URI / selected file
│   │   ├── read permission
│   │   ├── source accessibility
│   │   ├── Model ID generation
│   │   ├── managed-storage copy/registration
│   │   ├── integrity verification
│   │   ├── duplicate detection
│   │   ├── cancellation
│   │   ├── cleanup on failure
│   │   └── Import Result
│   │
│   ├── Inspection
│   │   ├── format detection
│   │   ├── GGUF header
│   │   ├── GGUF structure
│   │   ├── metadata extraction
│   │   ├── architecture
│   │   ├── quantization
│   │   ├── tensor/file information
│   │   ├── tokenizer metadata
│   │   └── runtime requirements
│   │
│   ├── Validation
│   │   ├── file validity
│   │   ├── GGUF validity
│   │   ├── metadata validity
│   │   ├── architecture support
│   │   ├── real quantization detection
│   │   ├── runtime compatibility
│   │   └── resource compatibility
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
├── 3. Model Domain
│   ├── ModelDescriptor
│   │   ├── id
│   │   ├── displayName
│   │   ├── source / managed path
│   │   ├── format
│   │   ├── quantization
│   │   ├── architecture
│   │   ├── size
│   │   ├── metadata
│   │   ├── validation status
│   │   └── runtime compatibility
│   │
│   ├── ModelMetadata
│   ├── ModelFormat
│   ├── Quantization
│   ├── Architecture
│   ├── ModelState
│   └── RuntimeCompatibility
│
├── 4. Runtime
│   │
│   ├── Runtime Contract
│   │   ├── load(model)
│   │   ├── unload()
│   │   ├── generate(...)
│   │   ├── stopGeneration()
│   │   └── runtimeInfo()
│   │
│   ├── Runtime Adapter
│   │   └── ModelRuntime / RuntimeAdapter boundary
│   │
│   ├── GGUF Backend
│   │   ├── backend initialization
│   │   ├── GGUF model loading
│   │   ├── metadata access
│   │   ├── tokenizer setup
│   │   ├── context setup
│   │   ├── inference
│   │   ├── generation cancellation
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
├── 5. Model Activation
│   ├── activate
│   ├── deactivate
│   ├── active model
│   ├── previous model during replacement
│   ├── safe transition
│   ├── runtime state
│   └── activation result
│
├── 6. Storage & File Lifecycle
│   ├── selected source
│   ├── managed model file
│   ├── persistent registry
│   ├── integrity information
│   ├── recovery
│   ├── cleanup
│   ├── deletion policy
│   └── temporary URI protection
│
├── 7. Operations
│   ├── Operation ID
│   ├── Execution ID
│   ├── Trace ID
│   ├── operation state
│   ├── cancellation
│   ├── concurrency policy
│   ├── state transitions
│   └── recovery
│
├── 8. Concurrency & Cancellation
│   ├── serialized/policy-controlled operations
│   ├── duplicate import protection
│   ├── duplicate load protection
│   ├── cancellation propagation
│   ├── cancellation cleanup
│   ├── race prevention
│   └── safe active-model transitions
│
├── 9. Error Model
│   ├── typed error
│   ├── error code
│   ├── user-facing message
│   ├── technical cause
│   ├── diagnostics reference
│   ├── FileAccessError
│   ├── UnsupportedFormat
│   ├── InvalidModel
│   ├── InvalidMetadata
│   ├── UnsupportedArchitecture
│   ├── UnsupportedQuantization
│   ├── RuntimeUnavailable
│   ├── LoadFailed
│   ├── OutOfMemory
│   ├── ImportCancelled
│   ├── StorageError
│   └── InferenceError
│
├── 10. Observability / Action Trace
│   ├── Import Trace
│   ├── Validation Trace
│   ├── Inspection Trace
│   ├── Registration Trace
│   ├── Activation Trace
│   ├── Load Trace
│   ├── Inference Trace
│   └── Trace Payload
│       ├── operationId
│       ├── executionId
│       ├── traceId
│       ├── modelId
│       ├── stage
│       ├── start/end
│       ├── real duration when measurable
│       ├── status
│       ├── errorCode
│       └── non-sensitive metadata
│
├── 11. Capabilities
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
├── 12. Resource Management
│   ├── model size
│   ├── memory requirements
│   ├── runtime resources
│   ├── load limits
│   ├── resource reservation
│   ├── cleanup
│   └── OutOfMemory handling
│
├── 13. Security
│   ├── File URI access
│   ├── persistent URI permission where required
│   ├── storage isolation
│   ├── input validation
│   ├── path/URI safety
│   ├── sensitive metadata protection
│   └── no arbitrary external execution
│
├── 14. Backend Tests
│   ├── valid GGUF import
│   ├── non-GGUF import
│   ├── corrupt file
│   ├── incomplete metadata
│   ├── unsupported architecture
│   ├── unsupported quantization
│   ├── valid Q6_K
│   ├── duplicate model
│   ├── import cancellation
│   ├── storage failure
│   ├── successful validation
│   ├── failed validation
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
│   ├── memory/resource failures
│   └── trace/error propagation
│
└── 15. UI Integration Contract
    │
    ├── Commands
    │   ├── import model
    │   ├── inspect
    │   ├── validate
    │   ├── activate
    │   ├── deactivate
    │   ├── unload
    │   └── delete
    │
    ├── Model List
    │   ├── list models
    │   ├── model descriptor
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
    │   ├── model name
    │   ├── file name
    │   ├── format
    │   ├── quantization
    │   ├── size
    │   ├── architecture
    │   ├── validation
    │   ├── compatibility
    │   └── runtime state
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
    ├── Results
    │   ├── ImportResult
    │   │   ├── success/failure
    │   │   ├── model descriptor
    │   │   ├── operation/execution id
    │   │   ├── validation result
    │   │   └── error
    │   └── RuntimeResult
    │       ├── success/failure
    │       ├── operation/execution id
    │       ├── runtime state
    │       └── error
    │
    ├── Events
    │   ├── MODEL_IMPORT_STARTED
    │   ├── MODEL_VALIDATION_STARTED
    │   ├── MODEL_VALIDATION_COMPLETED
    │   ├── MODEL_IMPORTED
    │   ├── MODEL_IMPORT_FAILED
    │   ├── MODEL_LOAD_STARTED
    │   ├── MODEL_LOADED
    │   ├── MODEL_LOAD_FAILED
    │   ├── MODEL_UNLOADED
    │   ├── MODEL_ACTIVATED
    │   └── MODEL_DEACTIVATED
    │
    └── Diagnostics
        ├── executionId
        ├── traceId
        ├── errorCode
        ├── error details
        └── navigation target for same execution/trace
```

## ۲. جریان استاندارد Backend

```text
Android File Picker
      ↓
File URI
      ↓
Import
      ↓
Integrity Check
      ↓
GGUF Inspection
      ↓
Validation
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
      ↓
Trace / Result
```

## ۳. اصل استقلال UI

تمام گره‌های `1` تا `14` باید در Core/Runtime/Model Management مستقل از Composable، Screen یا ViewModel طراحی و کدنویسی شوند.

UI فقط قرارداد گره `15` را مصرف می‌کند.

هیچ Parse، Validation، Persistence یا Runtime Loading نباید داخل UI انجام شود.

## ۴. مدل هدف اولیه

**Qwen3-1.7B · GGUF · Q6_K**

این انتخاب مدل هدف است. Backend باید مشخصات واقعی فایل انتخاب‌شده را از GGUF metadata و validation استخراج کند و نباید صرفاً از نام فایل نتیجه‌گیری کند.

## ۵. قرارداد اتصال به UI

این بخش باید **همزمان با طراحی و کدنویسی Backend** به‌روزرسانی شود. هر API، State، Event، Result، Capability یا Error جدیدی که UI باید مصرف کند باید در گره `15` ثبت شود.

ترتیب مرجع:

```text
Backend Implementation
        ↓
UI Integration Contract
        ↓
UI_TREE.md
        ↓
SCREENS.md
        ↓
LOCAL_MODEL_IMPORT_UI.md
```

پس از هر تغییر Backend که روی UI اثر دارد، ارجاعات اسناد UI باید همان زمان بررسی و اصلاح شوند.

## ۶. Definition of Done

Backend مدل محلی فقط زمانی کامل محسوب می‌شود که:

- Import واقعی انجام شود.
- GGUF واقعاً Inspect و Validate شود.
- Metadata واقعی استخراج شود.
- Model Repository پایدار باشد.
- Lifecycle کامل مدل پیاده شود.
- Runtime واقعی GGUF متصل شود.
- Q6_K در صورت پشتیبانی Backend واقعاً Load شود.
- Inference واقعی کار کند.
- Activation/Deactivation امن باشد.
- Cancellation و Concurrency کنترل شوند.
- Resource و Memory Management وجود داشته باشد.
- Error Contract کامل باشد.
- Action Trace/Diagnostics واقعی ثبت شود.
- تست‌های لازم واقعی باشند.
- هیچ موفقیت Import/Load/Inference با Mock یا Placeholder جعل نشود.
- UI Integration Contract کامل باشد.
- ارجاعات لازم در اسناد UI برقرار باشند.

## ۷. ارجاعات اجباری

- UI Tree: [`../ui/UI_TREE.md`](../ui/UI_TREE.md)
- UI Screens: [`../ui/SCREENS.md`](../ui/SCREENS.md)
- UI Import Model: [`../ui/LOCAL_MODEL_IMPORT_UI.md`](../ui/LOCAL_MODEL_IMPORT_UI.md)

این سند **مرجع واحد Backend مدل محلی** است. قرارداد موازی برای همین قابلیت نباید ساخته شود.
