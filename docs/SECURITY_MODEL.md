# Security Model

## هدف

تعریف مرزهای امنیتی Actionها و جلوگیری از اجرای ناخواسته یا غیرمجاز عملیات توسط Agent.

## Action classification

هر Action باید این metadata را داشته باشد:

```text
Risk Level: SAFE | LOW | MEDIUM | HIGH | CRITICAL
Read Only: true/false
Mutates State: true/false
Requires Permission: true/false
Requires Confirmation: true/false
Reversible: true/false
```

## Permission vs Confirmation

**Permission** مشخص می‌کند Agent اصولاً اجازه استفاده از Action را دارد یا نه.

**Confirmation** مشخص می‌کند برای Action مجاز، قبل از اجرای واقعی باید کاربر تأیید کند یا نه.

این دو نباید با هم یکی شوند.

## Examples

```text
get_time
SAFE / Read-only / No confirmation

read_file
LOW / Read-only

create_file
MEDIUM / State-changing

delete_file
HIGH / State-changing / Confirmation required

update_price (future WooGit)
HIGH / State-changing / Confirmation according to policy

delete_product (future WooGit)
CRITICAL / State-changing / Confirmation required
```

## Trust boundary

```text
Model Output
   ↓ untrusted
Parser
   ↓
Validator
   ↓
Permission
   ↓
Confirmation
   ↓
Executor
   ↓
Verifier
```

مدل منبع قابل اعتماد برای اعلام اجرای موفق Action نیست.

## Safety rules

- Action ناشناخته اجرا نشود.
- پارامتر نامعتبر اجرا نشود.
- Permission ردشده اجرا نشود.
- Confirmation ردشده اجرا نشود.
- Verification شکست‌خورده به‌عنوان موفقیت گزارش نشود.
- Secrets و credentialها نباید در Prompt/Result/Debug Log بدون نیاز قرار گیرند.
- Actionهای خطرناک باید کمترین سطح دسترسی لازم را داشته باشند.
