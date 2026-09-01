# Contract Index

مرجع واحد برای Contractهای اصلی پروژه.

| Contract | سند مرجع |
|---|---|
| ModelRuntime | `ARCHITECTURE.md` |
| Context | `ARCHITECTURE.md` / `PROTOTYPE_SPEC.md` |
| Action Protocol | `ACTION_PROTOCOL.md` |
| Action Registry | `ACTION_REGISTRY.md` |
| Capability | `ACTION_REGISTRY.md` / `WOOGIT_INTEGRATION.md` |
| Execution & Verification | `ACTION_EXECUTION_VERIFICATION.md` |
| Recovery | `RECOVERY_FAILURE.md` |
| Settings | `SETTINGS_CORE.md` |
| Persistence | `PERSISTENCE.md` |
| Observability | `OBSERVABILITY_AND_ERROR_CENTER.md` |
| ErrorReport | `ERROR_REPORT_SCHEMA.md` |
| Workspace | `AI_WORKSPACE.md` / `ui/UI_TREE.md` |
| UI Structure | `ui/UI_TREE.md` |
| UI Navigation | `ui/NAVIGATION.md` |
| UI Screens | `ui/SCREENS.md` |
| UI Controls | `ui/CONTROLS.md` |
| UI States | `ui/STATES.md` |
| UI Design System | `ui/DESIGN_SYSTEM.md` |
| UI Decisions | `ui/questions/README.md` |
| Security | `SECURITY_MODEL.md` |
| WooGit Adapter | `WOOGIT_INTEGRATION.md` |
| CI/Release | `PLATFORM_RELEASE_CI.md` |

## Source of truth

اگر جزئیات یک Contract در چند سند تکرار شود، سند تخصصی همان Contract مرجع دقیق است و `ARCHITECTURE.md` فقط معماری سطح بالا را مشخص می‌کند.

برای ساختار UI، **`ui/UI_TREE.md` مرجع Canonical** است. `ui/SCREENS.md`، `ui/NAVIGATION.md`، `ui/CONTROLS.md`، `ui/STATES.md` و `ui/DESIGN_SYSTEM.md` باید با آن یکسان و هماهنگ باشند. تصمیم‌های UI در `ui/questions/` ثبت می‌شوند و درخت نهایی باید تصمیم‌های قفل‌شده را منعکس کند.
