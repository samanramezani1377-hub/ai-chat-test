# Documentation → Implementation Traceability

این سند مرجع اتصال Contractهای معماری به محل پیاده‌سازی است.

| Contract | محل پیشنهادی |
|---|---|
| ModelRuntime | `core/runtime` |
| Context | `core/context` |
| Agent | `core/agent` |
| Action Protocol | `core/actions/protocol` |
| Action Registry | `core/actions/registry` |
| Capability Contract | `core/actions/capability` |
| Executor | `core/actions/execution` |
| Verifier | `core/actions/verification` |
| Settings | `core/settings` |
| Persistence | `core/persistence` |
| Observability | `core/observability` |
| ErrorReport | `core/observability/error` |
| Runtime Adapter | `adapters/runtime` |
| Test Adapter | `adapters/test` |
| WooGit Adapter | `adapters/woogit` |
| Chat UI | `ui/chat` |
| Workspace UI | `ui/workspace` |
| Settings UI | `ui/settings` |
| Error Center UI | `ui/errors` |

## Rules

- UI مستقیماً Adapter را مصرف نمی‌کند.
- Core به implementation Runtime/Adapter وابسته نمی‌شود.
- Adapterها Contractهای Core را پیاده‌سازی می‌کنند.
- Error Center فقط API/Contractهای Observability را مصرف می‌کند.
- Workspace Executor نیست.
- این mapping برای جلوگیری از ambiguity است و نباید coupling جدید ایجاد کند.
