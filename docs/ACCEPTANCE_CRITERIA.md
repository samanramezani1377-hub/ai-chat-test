# Acceptance Criteria

این سند معیار آمادگی Prototype برای مهاجرت است. معیارهای کیفیت نهایی و تصمیم نهایی پذیرش توسط مالک پروژه انجام می‌شود.

## Architecture

- [x] AI Core مستقل از UI و Runtime باشد.
- [x] Runtime فقط از طریق Adapter Contract متصل باشد.
- [x] Action Registry، Capability، Executor/Adapter و Verifier مرز روشن داشته باشند.
- [x] WooGit implementation وارد Core نشده باشد.
- [x] Dependency Rules رعایت شده باشند.

## Actions

- [x] Action Contractها پایدار و قابل تست باشند.
- [x] Sensitive Action مسیر Preview/Snapshot → Validate → Final Approval → Execute → Verify را رعایت کند.
- [x] Approval باعث تصمیم‌گیری مجدد AI نشود.
- [x] Success فقط پس از Verification اعلام شود.

## Recovery

- [x] Checkpoint و State قابل بازیابی باشند.
- [x] Retry کورکورانه وجود نداشته باشد.
- [x] Idempotency برای Actionهای لازم رعایت شود.
- [x] وضعیت نامشخص قبل از Execute مجدد Verify شود.

## Observability

- [x] Error/Eventها از Central Logging عبور کنند.
- [x] ErrorReport Contract واحد باشد.
- [x] Persian user error و Raw Machine Error هر دو قابل دسترسی باشند.
- [x] Copy Error Report کار کند.
- [x] Secretها Redact شوند.
- [x] Technical Error Panel بدون حذف Logging قابل Hide باشد.

## Context / Settings / Workspace

- [ ] Recent Messages و سایر مقادیر User-configurable واقعاً قابل تنظیم باشند.
- [ ] Summary و Persistent Task Context از Chat History مستقل باشند.
- [ ] Settings Versioning/Migration کار کند.
- [x] Workspace Executor نباشد و Context را بی‌دلیل کامل وارد Prompt نکند.

## Testing / Release

- [ ] Build و تست‌های پایه سبز باشند.
- [ ] Integration و Contract tests برای مسیرهای اصلی وجود داشته باشند.
- [x] تست‌های سخت‌گیرانه Action/Recovery فعلاً می‌توانند خارج از Hard Gate باشند.
- [ ] قبل از Hard Gate نهایی، مالک کیفیت اپ را شخصاً تأیید کند.

## Final decision

قبولی نهایی Prototype و تصمیم درباره ورود به مرحله Migration با مالک پروژه است. این سند معیار فنی و قابل بررسی فراهم می‌کند و جایگزین تصمیم نهایی مالک نیست.
