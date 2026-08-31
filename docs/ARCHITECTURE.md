# Architecture

## Purpose

تعریف معماری مرجع پروژه AI Chat Test و مرزبندی روشن بین UI، AI Core، Agent، Action System، Executor، Verifier، Workspace و لایه‌های ذخیره‌سازی/Observability.

## High-level

```text
UI
├── Chat
├── Settings
├── Debug/Test Panel
└── AI Workspace
        │
        ↓
     AI Core
        ├── Model Management
        ├── Inference
        ├── Context / Conversation
        ├── Agent
        ├── Action Protocol
        └── Observability
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

این تصمیم برای Chat، Workspace و قابلیت‌های آینده WooGit مانند بازنویسی توضیحات و تغییر قیمت الزام‌آور است.

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

## Adapters

Runtime مدل، Storage، Android APIs و در آینده WooGit باید پشت Adapter/Port قرار گیرند تا Core به یک implementation خاص وابسته نشود.

## State ownership

- Core مالک State منطقی Task و Inference است.
- Action System مالک Lifecycle اجرای Action است.
- Verifier مالک نتیجه Verification است.
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
