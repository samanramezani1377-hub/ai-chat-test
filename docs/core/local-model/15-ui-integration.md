# 15 — UI Integration Contract

این تنها مرز رسمی اتصال Backend مدل محلی به UI است. Backend از UI مستقل می‌ماند.

```text
UI Integration
├── Commands
│   ├── import model
│   ├── inspect
│   ├── validate
│   ├── activate
│   ├── deactivate
│   ├── unload
│   └── delete
├── Model List
│   ├── list
│   ├── descriptor
│   └── item state
├── Current Model
│   ├── active model
│   └── active state
├── Import Model State
│   ├── start
│   ├── importing
│   ├── validating
│   ├── ready
│   ├── cancelled
│   └── failed
├── Model Details
│   ├── name
│   ├── file name
│   ├── format
│   ├── quantization
│   ├── size
│   ├── architecture
│   ├── validation
│   ├── compatibility
│   └── runtime state
├── Runtime
│   ├── status
│   ├── loading state
│   ├── runtime info
│   ├── capabilities
│   └── response delivery mode
├── Reactive State
│   ├── models
│   ├── currentModel
│   ├── importState
│   ├── validationState
│   ├── runtimeState
│   ├── activeOperation
│   └── error
├── Results
│   ├── ImportResult
│   └── RuntimeResult
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
└── Diagnostics
    ├── executionId
    ├── traceId
    ├── errorCode
    ├── error details
    └── navigation target for the same execution/trace
```

## Runtime response delivery

نسخه فعلی Android Runtime از `dev.ffmpegkit-maintained:llama-android:0.1.1` استفاده می‌کند. این AAR در نسخه Free خروجی کامل تولید می‌کند و Streaming Token/Flow ارائه نمی‌دهد؛ بنابراین Backend فعلاً یک `onToken` callback با کل متن نهایی ارسال می‌کند. UI نباید این را به‌عنوان Streaming تدریجی نمایش دهد.

قرارداد UI از ابتدا `response delivery mode` را نگه می‌دارد تا با جایگزینی Adapter با یک Runtime دارای Streaming واقعی، UI بدون تغییر معماری بتواند حالت Token Streaming را مصرف کند.

## جریان اتصال

```text
UI Settings / AI Model Area
          ↓
     Import Model
          ↓
   Android File Picker
          ↓
         URI
          ↓
   Backend Model Import
          ↓
 Result + State + IDs + Trace
          ↓
        UI State
```

## قانون Diagnostics

اگر یک Action در Backend Trace داشته باشد، UI باید بتواند با `executionId` یا `traceId` مستقیماً به گزارش همان Execution در بخش عیب‌یابی برسد. UI نباید گزارش مستقل و جداگانه‌ای برای همان Execution بسازد.

## ارجاعات

### Backend

- [Architecture](01-architecture.md)
- [Model Management](02-model-management.md)
- [Model Domain](03-model-domain.md)
- [Runtime](04-runtime.md)
- [Activation](05-activation.md)
- [Storage](06-storage.md)
- [Operations](07-operations.md)
- [Concurrency & Cancellation](08-concurrency.md)
- [Errors](09-errors.md)
- [Observability](10-observability.md)
- [Capabilities](11-capabilities.md)
- [Resources](12-resources.md)
- [Security](13-security.md)
- [Testing](14-testing.md)
- [Definition of Done](16-definition-of-done.md)

### UI

- `docs/ui/UI_TREE.md`
- `docs/ui/SCREENS.md`
- `docs/ui/LOCAL_MODEL_IMPORT_UI.md`

هر تغییر در Contract باید ابتدا این سند را به‌روزرسانی کند و سپس ارجاعات UI بررسی شوند.
