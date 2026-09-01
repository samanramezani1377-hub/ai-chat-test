# 15 — UI Integration Contract

این تنها بخشی است که Backend را به UI متصل می‌کند. خود Backend مستقل از UI باقی می‌ماند.

```text
UI Integration
├── Commands
│   ├── Import Model
│   ├── Inspect
│   ├── Validate
│   ├── Activate
│   ├── Deactivate
│   ├── Unload
│   └── Delete
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
│   └── capabilities
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
    └── navigation target
```

## جریان اتصال

```text
Settings → AI → Model
        ↓
Import Model
        ↓
Android File Picker
        ↓
URI
        ↓
Backend Import
        ↓
Result + State + Execution/Trace
        ↓
UI
```

UI نباید Parser یا Runtime را مستقیماً اجرا کند.

هر تغییر Backend که چیزی به UI اضافه/حذف/تغییر می‌دهد باید همین سند را همزمان به‌روزرسانی کند.

ارجاع UI: `docs/ui/UI_TREE.md`, `docs/ui/SCREENS.md`, `docs/ui/LOCAL_MODEL_IMPORT_UI.md`
