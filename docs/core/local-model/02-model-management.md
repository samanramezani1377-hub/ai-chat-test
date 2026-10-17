# 02 — Model Management

```text
Model Management
├── Import
│   ├── Android SAF URI
│   ├── ZIP package staging
│   ├── Safe ZIP path validation
│   ├── Managed model directory
│   └── Temporary archive cleanup
├── Inspection / Validation
│   ├── genai_config.json
│   ├── At least one ONNX graph
│   ├── Tokenizer assets
│   ├── Model type and context metadata
│   └── Expanded-size and entry-count limits
├── Repository
│   ├── Register model directory and total size
│   ├── Get / List
│   ├── Active Model
│   └── Unregister and recursive cleanup
└── Lifecycle Manager
    ├── Importing
    ├── Ready
    ├── Loading
    ├── Active
    ├── Unloading
    └── Failed
```

در شاخهٔ `onnx-runtime-genai` بستهٔ مدل باید ZIP باشد و پس از استخراج، در ریشهٔ بسته `genai_config.json`، حداقل یک فایل `.onnx` و فایل‌های tokenizer موجود باشند. اگر بسته یک پوشهٔ ریشهٔ مشترک داشته باشد، importer آن پوشه را به‌عنوان ریشهٔ مدل انتخاب می‌کند.

فایل GGUF، فایل ONNX تکی یا ZIP ناقص فعال نمی‌شود. مدل‌های GGUF قدیمی در فهرست باقی می‌مانند تا کاربر بتواند آن‌ها را حذف کند، اما دکمهٔ فعال‌سازی برای آن‌ها غیرفعال است.

## ارجاعات

- [Architecture](01-architecture.md)
- [Model Domain](03-model-domain.md)
- [Storage](06-storage.md)
- [Operations](07-operations.md)
- [Concurrency & Cancellation](08-concurrency.md)
- [Capabilities](11-capabilities.md)
- [Testing](14-testing.md)
- [UI Integration](15-ui-integration.md)
