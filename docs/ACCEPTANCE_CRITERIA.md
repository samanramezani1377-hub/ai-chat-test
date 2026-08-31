# Acceptance Criteria

این سند معیار آمادگی Prototype برای مهاجرت است. معیارهای کیفیت نهایی و تصمیم نهایی پذیرش توسط مالک پروژه انجام می‌شود.

## Architecture

- [ ] AI Core مستقل از UI و Runtime باشد.
- [ ] Runtime فقط از طریق Adapter Contract متصل باشد.
- [ ] Action Registry، Capability، Executor/Adapter و Verifier مرز روشن داشته باشند.
- [ ] WooGit implementation وارد Core نشده باشد.
- [ ] Dependency Rules رعایت شده باشند.

## Actions

- [ ] Action Contractها پایدار و قابل تست باشند.
- [ ] Sensitive Action مسیر Preview/Snapshot → Validate → Final Approval → Execute → Verify را رعایت کند.
- [ ] Approval باعث تصمیم‌گیری مجدد AI نشود.
- [ ] Success فقط پس از Verification اعلام شود.

## Recovery

- [ ] Checkpoint و State قابل بازیابی باشند.
- [ ] Retry کورکورانه وجود نداشته باشد.
- [ ] Idempotency برای Actionهای لازم رعایت شود.
- [ ] وضعیت نامشخص قبل از Execute مجدد Verify شود.

## Observability

- [ ] Error/Eventها از Central Logging عبور کنند.
- [ ] ErrorReport Contract واحد باشد.
- [ ] Persian user error و Raw Machine Error هر دو قابل دسترسی باشند.
- [ ] Copy Error Report کار کند.
- [ ] Secretها Redact شوند.
- [ ] Technical Error Panel بدون حذف Logging قابل Hide باشد.

## Context / Settings / Workspace

- [ ] Recent Messages و سایر مقادیر User-configurable واقعاً قابل تنظیم باشند.
- [ ] Summary و Persistent Task Context از Chat History مستقل باشند.
- [ ] Settings Versioning/Migration کار کند.
- [ ] Workspace Executor نباشد و Context را بی‌دلیل کامل وارد Prompt نکند.

## Testing / Release

- [ ] Build و تست‌های پایه سبز باشند.
- [ ] Integration و Contract tests برای مسیرهای اصلی وجود داشته باشند.
- [ ] تست‌های سخت‌گیرانه Action/Recovery فعلاً می‌توانند خارج از Hard Gate باشند.
- [ ] قبل از Hard Gate نهایی، مالک کیفیت اپ را شخصاً تأیید کند.

## Final decision

قبولی نهایی Prototype و تصمیم درباره ورود به مرحله Migration با مالک پروژه است. این سند معیار فنی و قابل بررسی فراهم می‌کند و جایگزین تصمیم نهایی مالک نیست.
