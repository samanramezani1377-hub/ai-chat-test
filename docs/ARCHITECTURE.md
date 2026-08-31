# Architecture

## Purpose

تعریف معماری مرجع پروژه AI Chat Test و مرزبندی روشن بین UI، AI Core، Agent، Action System، Executor، Verifier، Workspace، Storage و Observability.

## High-level

```text
UI
├── Chat
├── Settings
├── Error Center / Debug Panel
└── AI Workspace
        │
        ↓
     AI Core
        ├── Model Management
        ├── Inference
        ├── Context / Conversation
        ├── Agent
        ├── Action Protocol
        └── Central Observability / Logging
                ↓
          Action System
          ├── Permission
          ├── Validation
          ├── Executor
          └── Verifier
                ↓
           Real Runtime
```

## Core/UI boundary

UI نباید منطق مدل، Agent یا Action را دوباره پیاده‌سازی کند. هر UI جدید باید از API/Contractهای Core استفاده کند.

این تصمیم برای Chat، Workspace، Error Center و قابلیت‌های آینده WooGit مانند بازنویسی توضیحات و تغییر قیمت الزام‌آور است.

## Central Observability

Logging و Error Handling باید در هسته مرکزی مدیریت شوند. Componentها Error/Event تولید می‌کنند اما ثبت استاندارد، Correlation، Persistence/Retention و ساخت Error Report توسط Central Observability انجام می‌شود.

یک خطا می‌تواند هم‌زمان یک پیام فارسی و کاربرپسند در همان محل رخداد و یک Raw Machine Error/Trace در Error Center داشته باشد. Error Center باید امکان مشاهده خطاهای ثبت‌شده و Copy یک Error Report کامل برای ارسال به Agent/Developer را فراهم کند.

UI مالک Logging نیست؛ فقط Event/Error را Subscribe یا Query می‌کند. جزئیات کامل این Contract در `OBSERVABILITY_AND_ERROR_CENTER.md` تعریف شده است.

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

## Adapters

Runtime مدل، Storage، Android APIs و در آینده WooGit باید پشت Adapter/Port قرار گیرند تا Core به یک implementation خاص وابسته نشود.

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
WooGit Operations
   ↓
Executor
   ↓
WooCommerce / App Services
   ↓
Verifier
```

هدف، انتقال Core و Contractهاست، نه کپی کل Prototype.
