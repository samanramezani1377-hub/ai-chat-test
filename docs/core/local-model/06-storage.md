# 06 — Storage & File Lifecycle

```text
Storage
├── Selected Source
│   ├── URI
│   └── Access Permission
├── Managed Model File
├── Persistent Registry
├── Integrity Information
├── Recovery
├── Cleanup
├── Deletion Policy
└── Temporary URI Protection
```

مسیر موقت یا URI غیرقابل‌دسترسی نباید به‌عنوان مسیر دائمی مدل ثبت شود.

```text
Selected URI → Validated Source → Managed File → Registry → Runtime
```

## Replacement and deletion guarantees

- Model files are staged in the same filesystem before replacement. The previous file is retained as a backup until the new file is moved into place; a failed move attempts to restore that backup.
- Voice data directories are replaced through a staged directory and a temporary backup. A failed installation restores the preserved previous directory instead of deleting it first.
- Model deletion first renames the model file to a same-directory tombstone, then unregisters the model. If unregistering fails or is cancelled, the original file is restored. The tombstone is deleted only after registry removal succeeds.
- If restoration or final tombstone cleanup fails, the operation returns a storage error and retains the recovery path instead of silently discarding the remaining copy.

## ارجاعات

- [Architecture](01-architecture.md)
- [Model Management](02-model-management.md)
- [Operations](07-operations.md)
- [Concurrency & Cancellation](08-concurrency.md)
- [Security](13-security.md)
- [Testing](14-testing.md)
- [UI Integration](15-ui-integration.md)
