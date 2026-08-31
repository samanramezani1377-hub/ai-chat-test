# AI Chat Test — Action Execution & Verification

## هدف

صرف اینکه مدل در پاسخ نهایی بگوید «انجام شد» اثبات نمی‌کند که Action واقعاً اجرا شده است. Prototype باید بتواند اجرای واقعی Action و نتیجه آن را قابل مشاهده و قابل بررسی کند.

## زنجیره اجرای قابل اثبات

```text
User Request
    ↓
AI / Agent
    ↓
Action Request
    ↓
Parser / Validator
    ↓
ActionExecutor
    ↓
Real Execution
    ↓
ActionVerifier
    ↓
Verified Tool Result
    ↓
Agent
    ↓
Final Answer
```

## مسئولیت اجزا

### ActionExecutor

Action را واقعاً اجرا می‌کند و نباید صرفاً موفقیت ساختگی برگرداند.

### ActionVerifier

مستقل از ادعای مدل، نتیجه واقعی اجرا را بررسی می‌کند. Verification باید تا حد امکان با بررسی State یا Result واقعی انجام شود.

### Agent

نتیجه Verified را دریافت می‌کند، در صورت نیاز مرحله بعدی را تصمیم می‌گیرد و Final Answer را بر اساس نتیجه واقعی تولید می‌کند.

### UI / Debug

شواهد اجرای واقعی را در اختیار کاربر قرار می‌دهد تا بتواند ببیند Action واقعاً انجام شده یا خیر.

## Verification بر اساس نوع Action

### create_file

بعد از اجرا باید در صورت امکان بررسی شود:

- فایل واقعاً وجود دارد.
- نام و مسیر فایل صحیح است.
- محتوای فایل با مقدار مورد انتظار تطابق دارد.
- اندازه فایل در صورت قابل‌اتکا بودن ثبت می‌شود.

نمونه:

```text
BEFORE
exists: false

ACTION
create_file
file: test.txt

AFTER
exists: true
content: matches

VERIFIED ✓
```

### read_file

فایل واقعی باید خوانده شود و محتوای واقعی به Agent برگردد. محتوا نباید از متن تولیدی مدل ساخته شود.

### delete_file

در صورت امکان وضعیت قبل و بعد بررسی شود:

```text
BEFORE
exists: true

ACTION
delete_file

AFTER
exists: false

VERIFIED ✓
```

### calculate

نتیجه باید توسط Executor واقعی محاسبه شود و همان نتیجه واقعی به Agent برگردد.

## Before / After State

برای Actionهایی که State را تغییر می‌دهند، در صورت امکان وضعیت قبل و بعد ثبت شود. این الگو برای مهاجرت به WooGit اهمیت ویژه دارد.

مثال آینده برای تغییر قیمت:

```text
PRODUCT #123

BEFORE
Price: 5,000,000

ACTION
update_product_price
+500,000

AFTER
Price: 5,500,000

VERIFIED ✓
```

اگر Verification شکست بخورد، نتیجه باید صریحاً Failed / Verification Failed باشد و Agent نباید موفقیت را اعلام کند.

## Action Test / Debug Panel

Prototype باید یک مسیر مشخص برای تست و مشاهده Action داشته باشد. برای هر اجرای Action حداقل این موارد قابل مشاهده باشند:

1. Action Request
2. پارامترهای ورودی
3. نتیجه واقعی Executor
4. نتیجه Verification
5. Final Answer

در صورت امکان Before/After State نیز نمایش داده شود.

نمونه:

```text
ACTION REQUEST
create_file

PARAMETERS
file_name: test.txt
content: Hello World

EXECUTOR
✓ Action started
✓ Action completed

VERIFICATION
✓ File exists
✓ Content matches

TOOL RESULT
success: true
verified: true

FINAL ANSWER
فایل test.txt با موفقیت ایجاد شد.
```

## اصل ضد موفقیت جعلی

Final Answer هرگز نباید مستقل از Tool Result Verified موفقیت اعلام کند.

اگر Action:

- اجرا نشده باشد؛
- شکست خورده باشد؛ یا
- Verification آن شکست خورده باشد؛

Agent باید همان وضعیت واقعی را گزارش کند.

## استقلال Verification از مدل

Verification نباید صرفاً داخل Prompt مدل تعریف شود و نباید بر اساس جمله‌ای مانند «فکر می‌کنم انجام شد» تصمیم بگیرد. Verification یک مسئولیت نرم‌افزاری مستقل است و باید با بررسی State یا Result واقعی انجام شود.

## کاربرد در WooGit

این معماری مستقیماً برای Actionهای آینده WooGit قابل استفاده است؛ از جمله:

- `update_product`
- `update_product_price`
- تغییر موجودی
- تغییر وضعیت محصول
- `create_product`
- `rewrite_description`
- عملیات سفارش، مشتری، محتوا و Media

مسیر پیشنهادی برای عملیات واقعی:

```text
AI Request
    ↓
Structured Action
    ↓
Validation
    ↓
Preview / Confirmation در صورت نیاز
    ↓
Real Execution
    ↓
Verification
    ↓
Real Result
    ↓
AI Final Answer
```

برای مثال در تغییر قیمت، AI فقط درخواست را تولید می‌کند؛ WooGit Operation آن را اعتبارسنجی و اجرا می‌کند و Verification نتیجه واقعی را بررسی می‌کند.

برای عملیات محتوایی مانند `rewrite_description` بهتر است خروجی ابتدا به‌صورت Preview ارائه شود تا کاربر بتواند آن را قبول، ویرایش یا رد کند.

## اصل کلی

**Executor انجام می‌دهد، Verifier ثابت می‌کند، Agent نتیجه را تفسیر می‌کند و UI شواهد را نشان می‌دهد.**

این تفکیک برای جلوگیری از موفقیت جعلی و برای مهاجرت قابل‌اعتماد AI به WooGit الزامی است.
