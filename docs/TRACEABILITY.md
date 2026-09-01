# Documentation → Implementation Traceability

این سند مرجع اتصال Contractهای معماری به محل پیاده‌سازی است.

## UI source of truth

درخت ساختاری UI در [`ui/UI_TREE.md`](ui/UI_TREE.md) مرجع Canonical است. `SCREENS.md`، `NAVIGATION.md`، `CONTROLS.md` و `STATES.md` جزئیات همان درخت را تکمیل می‌کنند و نباید ساختار مستقل متعارض ایجاد کنند.

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
| Chat UI | `ui/chat` ← مطابق `UI_TREE.md` |
| Workspace UI | `ui/workspace` ← مطابق `UI_TREE.md` |
| Settings UI | `ui/settings` ← مطابق `UI_TREE.md` |
| Error Center UI | `ui/errors` ← مطابق `UI_TREE.md` |

## UI structural mapping

```text
UI_TREE.md
│
├── App Shell
│   ├── Header
│   └── Sidebar
│
├── گفت‌وگو → ui/chat
│
├── فضای کار → ui/workspace
│   ├── خلاصه اجرای جاری
│   ├── Timeline مرکزی
│   ├── Event / Action + Expand لایه‌ای
│   └── تاریخچه عملکرد
│
├── عیب‌یابی → ui/errors
│   ├── خطاها
│   ├── گزارش Execution / Trace
│   ├── عملکرد
│   └── گزارش کار حساس
│
├── تنظیمات → ui/settings
└── درباره برنامه
```

## Rules

- UI مستقیماً Adapter را مصرف نمی‌کند.
- Core به implementation Runtime/Adapter وابسته نمی‌شود.
- Adapterها Contractهای Core را پیاده‌سازی می‌کنند.
- Error Center فقط API/Contractهای Observability را مصرف می‌کند.
- Workspace Executor نیست.
- کنترل‌های Workspace Context-aware و وابسته به Capability/State واقعی هستند.
- جزئیات Contextual داخل Expand همان Event است؛ پنل مستقل Workspace وجود ندارد.
- این mapping برای جلوگیری از ambiguity است و نباید coupling جدید ایجاد کند.
