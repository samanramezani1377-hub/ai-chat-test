# AI Chat Test — Performance Metrics

## Question 17 — چه اطلاعات Performance اندازه‌گیری شود؟

تمام Metricهای زیر باید در Prototype اندازه‌گیری شوند و مقدار واقعی داشته باشند. هیچ Metric صوری، تخمینی یا Mock مجاز نیست.

## Metrics

### Inference
- First Token Time
- Input Tokens
- Output Tokens
- Generation Time
- Tokens/sec
- Context Usage

### Model
- Load Time
- Unload Time (فقط در صورت پشتیبانی واقعی Runtime)

### Device
- RAM
- CPU
- GPU/NPU (فقط در صورت دسترسی و اندازه‌گیری واقعی)
- Backend

### Agent
- Total Agent Time
- Step Count
- Action Time برای هر Action
- Retry Count
- Error Count

### Network
- Network Request/Connectionهای واقعی
- Network Usage در حد اطلاعاتی که Android/Runtime واقعاً ارائه می‌کند

### History
- ذخیره نتایج تست‌های قبلی
- مقایسه Performance
- Export گزارش

## نمایش در UI

هر Metric باید یک شناسه/تنظیم مستقل برای Visibility داشته باشد. نمایش یا عدم نمایش هر Metric نباید به Metricهای دیگر وابسته باشد.

نمونه معماری پیشنهادی:

```kotlin
object PerformanceVisibility {
    const val SHOW_FIRST_TOKEN_TIME = true
    const val SHOW_INPUT_TOKENS = true
    const val SHOW_OUTPUT_TOKENS = true
    const val SHOW_GENERATION_TIME = true
    const val SHOW_TOKENS_PER_SECOND = true
    const val SHOW_CONTEXT_USAGE = true
    const val SHOW_MODEL_LOAD_TIME = true
    const val SHOW_MODEL_UNLOAD_TIME = true
    const val SHOW_RAM = true
    const val SHOW_CPU = true
    const val SHOW_GPU_NPU = true
    const val SHOW_BACKEND = true
    const val SHOW_AGENT_TOTAL_TIME = true
    const val SHOW_AGENT_STEP_COUNT = true
    const val SHOW_ACTION_TIME = true
    const val SHOW_RETRY_COUNT = true
    const val SHOW_ERROR_COUNT = true
    const val SHOW_NETWORK_USAGE = true
}
```

اصل مهم: تغییر یک مقدار، فقط Visibility همان Metric را تغییر دهد. مثلاً:

```kotlin
const val SHOW_CPU = false
```

باید فقط نمایش CPU را خاموش کند، بدون اینکه اندازه‌گیری CPU یا سایر Metricها را حذف کند.

## Measurement vs Visibility

اندازه‌گیری و نمایش باید کاملاً از هم جدا باشند:

```text
Runtime
  ↓
Metrics Collector
  ↓
Performance Snapshot
  ├── firstTokenMs
  ├── inputTokens
  ├── outputTokens
  ├── generationTimeMs
  ├── tokensPerSecond
  ├── contextUsage
  ├── modelLoadTimeMs
  ├── ram
  ├── cpu
  ├── backend
  ├── networkUsage
  └── agent metrics
       ↓
Visibility Configuration
       ↓
UI
```

بنابراین خاموش‌کردن نمایش یک Metric نباید باعث شود داده Performance از Collector حذف شود یا Measurement از کار بیفتد.

## تغییر با حداقل دستکاری کد

Visibility باید متمرکز باشد و UI مستقیماً شرط‌های پراکنده مانند `if (showCpu)` در چندین فایل نداشته باشد. هر Metric یک Key مستقل داشته باشد و Renderer/Composable بر اساس همان Key تصمیم بگیرد.

ترجیحاً تنظیمات Visibility در یک محل مرکزی نگهداری شوند تا برای مخفی/نمایان کردن یک Metric فقط **یک مقدار** تغییر کند.

## Runtime Availability

اگر یک Metric توسط Runtime یا دستگاه قابل اندازه‌گیری واقعی نباشد:

```text
Unavailable
```

نمایش داده شود یا بر اساس Visibility همان Metric مخفی شود؛ مقدار ساختگی نباید جایگزین آن شود.

## Network / Offline

تمام Network Usage اپ باید به‌صورت واقعی مانیتور شود. Network Request/Connectionهای واقعی و میزان مصرف شبکه، در حد اطلاعاتی که Android/Runtime واقعاً ارائه می‌کند، ثبت شوند. تست Offline واقعی با قطع اینترنت انجام می‌شود. Firewall، DNS، Fresh Install و سناریوهای پیچیده جزو Requirement نیستند.

## Performance History

نتایج Performance باید قابل ذخیره و مشاهده مجدد باشند و هر Record حداقل مدل، Runtime، Backend، تنظیمات اصلی Inference و Timestamp را داشته باشد تا مقایسه معنادار باشد.

## Export

Performance Report باید قابل Export باشد و داده واقعی Snapshot/History را منتقل کند.

## تصمیم نهایی

تمام Metricهای بالا باید وجود داشته باشند، اما **Visibility هر Metric کاملاً مستقل و جداگانه قابل تنظیم باشد** و تغییر آن با یک تغییر بسیار کوچک در یک تنظیم مرکزی انجام شود. Measurement از Visibility جدا است و خاموش‌کردن نمایش، Measurement را غیرفعال نمی‌کند.
