# Documentation Readiness

## Final documentation gate

این سند وضعیت آماده‌سازی مستندات برای شروع پیاده‌سازی را ثبت می‌کند.

### Completed

- Core / UI / Runtime boundary
- Action Registry / Contract / Capability / Adapter / Executor / Verifier boundary
- Central Settings Contract + Versioning + Migration
- Central Observability + Error Center
- Canonical ErrorReport Schema
- Hideable Technical Error Panel بدون حذف Logging
- Documentation → Implementation Traceability
- Canonical Contract Index
- Dependency Rules
- End-to-End Data Flow / Lifecycle
- Failure Matrix
- Prototype → WooGit Migration Mapping
- Prototype Acceptance Criteria

### Authority order

```text
Final owner decisions
        ↓
DECISIONS_21_30.md
        ↓
Specialized Contract Documents
        ↓
ARCHITECTURE.md
        ↓
README.md (navigation / overview)
```

اگر بین Overview و یک Contract تخصصی اختلافی دیده شود، Contract تخصصی و تصمیم ثبت‌شده مرجع است و اختلاف باید پیش از implementation برطرف شود.

### Implementation rule

Agent/Developer نباید بر اساس حدس معماری جدید ایجاد کند. ابتدا Contract Index، Architecture، Dependency Rules و Traceability را بخواند و سپس Implementation را مطابق آن‌ها انجام دهد.

### Final owner gate

معیار کیفیت نهایی و تصمیم نهایی برای عبور از Prototype به Migration توسط مالک پروژه انجام می‌شود. تست‌های سخت‌گیرانه Action/Recovery تا آن مرحله الزام Hard Gate ندارند، مگر اینکه مالک پروژه تصمیم دیگری ثبت کند.
