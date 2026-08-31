# Architecture

## Purpose

تعریف معماری مرجع پروژه AI Chat Test و مرزبندی روشن بین UI، AI Core، Agent، Action System، Executor، Adapter، Verifier، Workspace، Storage و Observability.

## High-level

```text
UI
├── Chat
├── Settings
├── Error Center
├── Technical Error Panel (hideable)
└── AI Workspace
        │
        ↓
     AI Core
        ├── Model Management
        ├── Inference
        ├── Context / Conversation
        ├── Agent
        ├── Action Protocol
        ├── Action Registry
        └── Central Observability / Logging
                ↓
          Action System
          ├── Permission
          ├── Validation
          ├── Capability Check
          ├── Adapter / Executor
          └── Verifier
```

## Core/UI boundary

UI نباید منطق مدل، Agent یا Action را دوباره پیاده‌سازی کند. هر UI جدید باید از API/Contractهای Core استفاده کند.

این تصمیم برای Chat، Workspace، Error Center و قابلیت‌های آینده WooGit مانند بازنویسی توضیحات و تغییر قیمت الزام‌آور است.

## Central Observability

Logging و Error Handling باید در هسته مرکزی مدیریت شوند. Componentها Error/Event تولید می‌کنند اما ثبت استاندارد، Correlation، Persistence/Retention و ساخت Error Report توسط Central Observability انجام می‌شود.

یک خطا می‌تواند هم‌زمان یک پیام فارسی و کاربرپسند در همان محل رخداد و یک Raw Machine Error/Trace در Error Center داشته باشد. Error Center باید امکان مشاهده خطاهای ثبت‌شده و Copy یک Error Report کامل برای ارسال به Agent/Developer را فراهم کند.

UI مالک Logging نیست؛ فقط Event/Error را Subscribe یا Query می‌کند.

پنل نمایش کامل خطاهای فنی و Raw Machine Error یک Presentation Feature مستقل و Hideable است. مخفی‌شدن آن نباید Logging Core، Error Store، Error Contract یا Error Report را حذف یا تغییر اساسی دهد؛ کنترل Visibility باید در لایه Presentation/Feature Flag انجام شود.

جزئیات این Contract در `OBSERVABILITY_AND_ERROR_CENTER.md` تعریف شده است.

## Runtime independence — Decision 21

AI Core باید کاملاً مستقل از Model Runtime باشد. `llama.cpp` فقط یکی از Runtime implementationهاست و نباید مستقیماً از داخل Domain/Core صدا زده شود.

```text
AI Core
   ↓
ModelRuntime Contract
   ↓
Runtime Adapter
   ↓
llama.cpp
```

Contract باید عملیات عمومی مانند Load/Unload، Generate/Streaming، Stop Generation، Model/Runtime Info، Context Info و اعمال تنظیمات Inference را در سطحی ارائه کند که Core به implementation خاص وابسته نشود.

تعویض Runtime باید بدون بازطراحی Chat، Agent یا UI امکان‌پذیر باشد. Adapter نباید برای هر Token یک لایه پردازش سنگین یا تبدیل غیرضروری ایجاد کند؛ Streaming باید مستقیماً و با کمترین overhead عملی منتقل شود.

## Conversation / Context — Decision 22

Context ترکیبی و از Source of Truth ساخته می‌شود. اجزای Context عبارت‌اند از:

```text
System Context
      +
Persistent Task Context
      +
Conversation Summary
      +
Recent Messages (user-configurable count)
      +
Workspace Context (query/select as needed)
```

Recent Messages تعداد پیش‌فرض دارد اما مقدار آن یک عدد معماری ثابت نیست و باید توسط کاربر قابل تنظیم باشد. پیام‌های قدیمی‌تر در Summary فشرده می‌شوند. Persistent Task Context هدف، کار فعلی، Intent کاربر و اطلاعات مهم را مستقل از Chat History نگه می‌دارد. Workspace نیز یک منبع Context مستقل است و Core/Agent فقط بخش مرتبط را انتخاب می‌کند تا کل Workspace بی‌دلیل وارد Prompt نشود.

Context هر درخواست از Source of Truth ساخته می‌شود و نباید با اضافه‌کردن بی‌نهایت متن به یک Prompt دائمی رشد کند.

## Action / Adapter / Capability Contract

مسیر مرجع اجرای Action این است:

```text
ActionRegistry
      ↓
Action Contract
      ↓
Capability Check
      ↓
Adapter / Executor
      ↓
Verifier
```

`ActionRegistry` مرجع تعریف و کشف Action است و Executor نیست. Actionها در Categoryهای قابل توسعه سازمان‌دهی می‌شوند و هر Category می‌تواند Executorهای متعدد داشته باشد.

Capability Contract مشخص می‌کند Runtime/Environment فعلی کدام Action یا Capability را پشتیبانی می‌کند. Adapter محیط اجرا را از Core جدا می‌کند. این قرارداد برای Test Adapter، WooGit Adapter و Adapterهای آینده یکسان است.

## State ownership

- Core مالک State منطقی Task و Inference است.
- Action System مالک Lifecycle اجرای Action است.
- Verifier مالک نتیجه Verification است.
- Central Observability مالک Event/Error Contract و Correlation است.
- Workspace فقط State/Result را نمایش می‌دهد و Interaction را به Core برمی‌گرداند.
- UI نباید State دامنه را با یک State مستقل و متناقض کپی کند.

## Future WooGit

```text
WooGit UI
   ↓
AI Core
   ↓
Action Registry
   ↓
Capability Check
   ↓
WooGit Adapter
   ↓
WooGit API / Service
   ↓
Verifier
```

هدف، انتقال Core و Contractهاست، نه کپی کل Prototype.
