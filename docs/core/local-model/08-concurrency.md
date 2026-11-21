# 08 — Concurrency & Cancellation

```text
Concurrency
├── Operation Serialization / Policy
├── Duplicate Import Protection
├── Duplicate Load Protection
├── Cancellation Propagation
├── Cancellation Cleanup
├── Race Prevention
└── Safe Active Model Transition
```

Import و Load نباید همزمان به State متناقض برسند. Cancellation باید واقعی باشد و بعد از لغو، منابع و فایل‌های موقت پاک‌سازی شوند.

## Runtime lifecycle invariants

LocalModelService owns the boundary between persisted model selection and native runtime residency:

- repository.getActive() is the persisted selection; it is not proof that native weights are loaded.
- activeId is set only after native load() and persistence of the selected model both succeed.
- Generation is allowed only when the persisted active model ID matches activeId.
- runtimeStateUnknown is set before native load/unload transitions. If a transition is interrupted or fails, the service must not load another model over uncertain native state.
- Rollback clears the in-memory active ID, attempts native unload and persisted-selection cleanup in NonCancellable, and keeps runtimeStateUnknown set if native unload fails. The next activation retries unload before attempting another load.

## Locking and cancellation order

Lifecycle changes follow one order:

1. Acquire lifecycleMutex and prevent new generations with stopInProgress.
2. Cancel the active generation job and request native stop without waiting for the generation mutex.
3. Acquire generationMutex, which is the barrier proving the previous generation has left native execution.
4. Acquire runtimeMutex for model load/unload and repository lifecycle transitions.
5. Release locks and clear stopInProgress in finally.

Generation acquires generationMutex before briefly reading the persisted active model under runtimeMutex; it releases runtimeMutex before native generation. Do not reverse this lock order or wait for the generation mutex before requesting native stop.

Cancellation is not an ordinary inference failure. Always rethrow CancellationException after required cleanup. Do not wrap suspending lifecycle cleanup in runCatching, because it can swallow cancellation. Native stop is best-effort; the generation mutex remains the definitive barrier before unload.

## Runtime configuration invalidation

GPU-layer preference changes increment a monotonic runtime-configuration revision. The service records the revision applied by the last successful activation. A same-model activation may return early only when those revisions match. Failed or cancelled native unloads mark the runtime state unknown; the next activation must successfully unload again before loading a model. This avoids losing a settings change that arrives during an in-flight load.

The Android adapter independently reuses an already loaded model only when the model path, context length, and requested GPU-layer mode all match. The requested GPU mode is tracked separately from the actual GPU layers used, because a successful CPU fallback must not cause an unnecessary reload on every activation.

## Tests

LocalModelServiceConcurrencyTest covers serialized lifecycle operations, persisted-but-unloaded model rejection, stop without deadlock, and cancellation during native activation followed by cleanup before the next activation.

## ارجاعات

- [Operations](07-operations.md)
- [Model Management](02-model-management.md)
- [Activation](05-activation.md)
- [Storage](06-storage.md)
- [Runtime](04-runtime.md)
- [Errors](09-errors.md)
- [Resources](12-resources.md)
- [Testing](14-testing.md)
- [UI Integration](15-ui-integration.md)
