# AI Workspace — میز کار هوش مصنوعی

## هدف

Prototype علاوه بر Chat باید برای AI یک **Workspace / میز کار** داشته باشد. هدف این است که AI فقط پاسخ متنی ندهد؛ بلکه بتواند نتایج، فایل‌ها، داده‌ها، Actionها، Previewها و خروجی‌های قابل تعامل را در یک فضای کاری قابل مشاهده در اختیار کاربر قرار دهد.

این Workspace بخشی از UI است، اما **نباید صاحب منطق AI، Agent یا اجرای Action باشد**. Workspace باید State و Result واقعی را از AI Core دریافت و نمایش دهد و تعاملات کاربر را به Core/Operationهای مربوطه منتقل کند.

## موارد استفاده

### 1. مشاهده نتیجه Action

مثلاً بعد از اجرای `create_file`:

```text
Workspace

📄 test.txt
   └── ایجاد شد ✓

⚙ Action
   └── create_file
       ✓ verified
```

### 2. مشاهده داده و نتیجه پردازش

مثلاً کاربر درخواست خلاصه‌کردن یک فایل را می‌دهد:

```text
Workspace

📄 test.txt
   └── خوانده شد ✓

📝 خلاصه
   └── نتیجه واقعی پردازش
```

### 3. نمایش نتایج چندمرحله‌ای Agent

Workspace باید بتواند خروجی‌های میانی و نهایی یک Task را در صورت فعال بودن قابلیت مربوطه نمایش دهد؛ مانند:

- Actionهای اجراشده
- Tool Resultها
- وضعیت Verification
- داده‌های جمع‌آوری‌شده
- خروجی نهایی

### 4. Preview و تأیید کاربر

Workspace باید محل مناسبی برای نمایش Preview عملیات باشد.

مثلاً تغییر قیمت:

```text
┌──────────────────────────────┐
│ تغییر قیمت                   │
│                              │
│ 5,000,000 → 5,500,000        │
│                              │
│ [لغو]       [تأیید و اجرا]   │
└──────────────────────────────┘
```

یا تولید/بازنویسی محتوا:

```text
┌──────────────────────────────┐
│ توضیحات پیشنهادی             │
│                              │
│ متن تولیدشده توسط AI         │
│                              │
│ [قبول] [ویرایش] [رد]         │
└──────────────────────────────┘
```

Workspace نباید بدون تأیید موردنیاز کاربر، Action حساس را اجرا کند.

## قابلیت Interactive بودن

Workspace صرفاً یک پنل نمایشی ثابت نیست. بسته به نوع Result می‌تواند عناصر تعاملی داشته باشد، مانند:

- Accept
- Edit
- Reject
- Confirm
- Cancel
- مشاهده جزئیات
- مشاهده Before / After
- مشاهده Tool Result

اما اجرای واقعی همچنان باید توسط لایه‌های Core/Operation/Executor انجام شود.

## ارتباط Workspace با Chat

Chat و Workspace باید از یک AI Core مشترک استفاده کنند:

```text
                         AI CORE
                            │
                 ┌──────────┴──────────┐
                 ↓                     ↓
               Chat                Workspace
                 ↓                     ↓
              Agent             AI Operations / Results
                 └──────────┬──────────┘
                            ↓
                     Action System
                            ↓
                Executor + Verifier
                            ↓
                       Real Result
                            ↓
                       Workspace
```

Chat می‌تواند یک Task را ایجاد کند و Workspace نتیجه آن Task را نمایش دهد. همچنین یک UI مستقیم می‌تواند بدون عبور از Chat یک AI Operation را درخواست کند و همان Workspace نتیجه را نمایش دهد.

## اصل مهم: Workspace اجراکننده نیست

Workspace نباید مستقیماً APIهای WooCommerce، فایل‌سیستم یا Runtime مدل را کنترل کند.

مسیر صحیح:

```text
User Interaction
      ↓
Workspace UI
      ↓
AI Core / WooGit Operation
      ↓
Validation
      ↓
Executor
      ↓
Verifier
      ↓
Real Result
      ↓
Workspace UI
```

این جداسازی مانع از تکرار منطق و ایجاد مسیرهای غیرقابل‌کنترل برای اجرای Action می‌شود.

## Workspace و Action Verification

برای Actionهایی که در Workspace نمایش داده می‌شوند، نتیجه باید بر اساس Execution و Verification واقعی باشد.

نمونه:

```text
ACTION
create_file

EXECUTOR
✓ completed

VERIFICATION
✓ file exists
✓ content matches

RESULT
success: true
verified: true
```

Workspace نباید صرفاً متن Final Answer مدل را به‌عنوان مدرک موفقیت نمایش دهد.

## Before / After

برای عملیات تغییر‌دهنده State، در صورت امکان Workspace باید وضعیت قبل و بعد را قابل مشاهده کند.

مثلاً در آینده WooGit:

```text
Product #123

BEFORE
Price: 5,000,000

ACTION
update_product_price

AFTER
Price: 5,500,000

VERIFIED ✓
```

این قابلیت برای عملیات محصول، سفارش، محتوا و سایر داده‌های WooGit اهمیت ویژه خواهد داشت.

## نقش Workspace در مهاجرت به WooGit

Workspace یکی از دلایل اصلی جداسازی AI Core از UI است. اگر AI فقط در Chat پیاده‌سازی شود، اضافه‌کردن قابلیت‌هایی مانند دکمه «بازنویسی با AI» یا Preview تغییر قیمت بعداً به بازطراحی Core منجر می‌شود.

معماری هدف:

```text
WooGit UI
│
├── Chat
├── Product Editor
│   ├── Description
│   │   └── [✨ بازنویسی با AI]
│   └── Price
│
└── AI Workspace
        │
        ↓
      AI Core
        │
        ├── Agent
        ├── AI Operations
        ├── Action Protocol
        ├── Executor
        └── Verifier
                ↓
         WooGit Services
                ↓
        WooCommerce API
```

در نتیجه Chat، دکمه‌های AI داخل صفحات و Workspace همگی باید بتوانند از Core و Operationهای مشترک استفاده کنند.

## اصول طراحی

- Workspace باید قابل مشاهده و تعاملی باشد.
- Workspace نباید جایگزین Chat یا AI Core شود.
- Workspace نباید منطق اجرای Action داشته باشد.
- نتایج نمایش‌داده‌شده باید واقعی و قابل ردیابی باشند.
- Verification باید مستقل از ادعای مدل باشد.
- عملیات حساس باید در صورت نیاز Preview و Confirmation داشته باشند.
- Workspace باید برای Resultهای چندمرحله‌ای Agent قابل توسعه باشد.
- طراحی باید از قابلیت‌های آینده WooGit مانند مدیریت محصولات، سفارش‌ها، مشتریان و محتوا پشتیبانی کند.
- اضافه‌کردن یک AI UI جدید نباید نیازمند کپی‌کردن منطق AI باشد.
- Workspace باید تا حد امکان مصرف‌کننده State و Resultهای Core باشد، نه محل نگهداری منطق دامنه.

## محدوده Prototype

در Prototype، Workspace ابتدا باید برای مشاهده و تست Actionها، Tool Resultها، Verification، Preview و خروجی‌های AI طراحی شود. جزئیات ظاهری UI، چیدمان نهایی و امکانات پیشرفته Workspace می‌توانند در مرحله طراحی UI تعیین شوند؛ اصل معماری و جداسازی مسئولیت‌ها از همین سند الزام‌آور است.
