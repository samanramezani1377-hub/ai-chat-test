# Action Registry

## Decision 27 — Registry + Adapter Architecture

معماری Actionها به‌صورت **Registry + Adapter Architecture** باشد.

`ActionRegistry` مرجع مرکزی تعریف و کشف Actionها است، اما خودش Executor نیست. هر Action یک Contract و Metadata مشخص دارد و اجرای واقعی از مسیر Action System و Adapter/Executor مربوط به محیط اجرا انجام می‌شود.

```text
AI Core
  ↓
Action Registry
  ↓
Action Contract
  ↓
Capability Check
  ↓
Adapter / Executor
  ↓
Verifier
```

## دسته‌بندی داخلی Actionها

Actionها داخل خود Registry به‌صورت دسته‌بندی‌شده سازمان‌دهی شوند تا با رشد پروژه مدیریت آن‌ها ساده بماند.

```text
ActionRegistry
├── Product
│   ├── Read
│   ├── Create
│   ├── Update
│   └── Delete
├── Content
│   ├── Read
│   ├── Rewrite
│   └── Update
├── Commerce
│   ├── Price
│   └── Inventory
└── Future Categories
```

این دسته‌بندی بخشی از ساختار منطقی Registry است و نباید باعث وابستگی Core به یک Executor خاص شود.

## Executorهای قابل توسعه

هر دسته می‌تواند در آینده Executorهای بیشتری داشته باشد. اضافه‌کردن Executor جدید نباید نیازمند بازنویسی Action Contract یا AI Core باشد.

```text
Product
├── WooGitProductExecutor
├── TestProductExecutor
└── FutureProductExecutor
```

هر Action Contract می‌تواند توسط Adapter/Executor متناسب با Environment اجرا شود. Capability Check باید قبل از Execution مشخص کند محیط فعلی قابلیت لازم را دارد.

## Action Metadata

هر Action حداقل می‌تواند این Metadata را داشته باشد:

- `id`
- `version`
- `name`
- `description`
- input schema
- output schema
- permission
- risk level
- confirmation policy
- executor/adapter reference
- verifier reference
- capability requirements

Actionهای حساس باید از Confirmation Policy مصوب استفاده کنند و Approval نباید مستقیماً Executor را از UI فعال کند؛ مسیر اجرای واقعی از Action System/Core عبور می‌کند.

## Extensibility Principle

هدف این طراحی این است که در آینده بتوان:

1. Action جدید اضافه کرد.
2. Category جدید اضافه کرد.
3. Executor جدید زیر یک Category اضافه کرد.
4. Adapter محیط اجرا را تغییر داد.
5. Capability جدید اضافه کرد.
6. Verifier متفاوت اضافه کرد.

بدون اینکه UI یا AI Core برای هر تغییر بازنویسی شود.
