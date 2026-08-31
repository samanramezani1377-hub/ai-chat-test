# Prototype → WooGit Migration Mapping

## Goal

انتقال قابلیت‌ها به WooGit با حفظ AI Core و Contractهای عمومی و با حداقل تغییر در UI.

| Prototype | Migration target |
|---|---|
| AI Core | حفظ می‌شود |
| Context / Task | حفظ می‌شود |
| Action Contract | حفظ می‌شود |
| Action Registry | حفظ می‌شود |
| Capability Contract | حفظ می‌شود |
| Runtime Adapter | قابل تعویض می‌ماند |
| Test Adapter | برای تست حفظ می‌شود |
| WooGit Adapter | اضافه/فعال می‌شود |
| Verifier | حفظ و با قابلیت‌های WooGit متصل می‌شود |
| Workspace | حفظ می‌شود |
| Central Settings | حفظ می‌شود |
| Central Observability | حفظ می‌شود |
| Error Center | حفظ می‌شود؛ Technical Panel قابل Hide است |
| UI-specific implementation | در صورت نیاز بازطراحی می‌شود |

## Migration sequence

```text
Freeze Contracts
 ↓
Validate Prototype
 ↓
Implement WooGit Adapter
 ↓
Map Capabilities
 ↓
Connect Verifiers
 ↓
Run Integration Tests
 ↓
Validate Recovery / Idempotency
 ↓
Owner Quality Approval
```

## Rules

- WooGit implementation نباید وارد Core شود.
- Actionهای عمومی مانند تغییر قیمت و بازنویسی توضیحات مستقل از WooGit تعریف می‌شوند.
- Approval semantics در مهاجرت تغییر نمی‌کند.
- Logging و ErrorReport Contract حفظ می‌شود.
- Settings Migration باید version-aware باشد.
