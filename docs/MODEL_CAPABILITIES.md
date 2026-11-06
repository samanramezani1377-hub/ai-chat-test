# Model Capabilities Contract

## Purpose

AI Chat Test باید برای مدل‌های فعلی و مدل‌هایی که در آینده به‌صورت GGUF یا از طریق Runtimeهای دیگر وارد می‌شوند، **model-agnostic** باقی بماند.

`Qwen3-1.7B Q6_K` مدل مرجع فعلی برای اعتبارسنجی قابلیت‌ها و تنظیمات است، اما هیچ بخش Core، Agent یا UI نباید به نام، اندازه، quantization یا رفتار اختصاصی Qwen3 hard-code شود.

## Reference model vs architecture contract

Qwen3-1.7B Q6_K در نسخه فعلی پروژه یک **reference/baseline model** است. علاوه بر آن، Qwen3.5-2B Q6_K و Qwen3.8-2B Distill Q6_K نیز به‌عنوان پروفایل‌های مرجع GGUF پشتیبانی می‌شوند. Qwen3.8-2B Distill در GGUF از معماری `qwen35` استفاده می‌کند؛ بنابراین نباید برای آن مسیر Runtime جداگانه یا hard-code مدل‌محور ساخته شود. از قابلیت‌های آن برای تست و اعتبارسنجی استفاده می‌شود، از جمله:

- context نسبتاً بزرگ
- thinking / non-thinking در صورت پشتیبانی Runtime
- streaming
- instruction following
- reasoning
- coding
- agent/tool-oriented usage
- sampling settings مانند Temperature، Top-P، Top-K و Min-P

این ویژگی‌ها به‌عنوان قابلیت‌های عمومی سیستم مدل/Runtime دیده می‌شوند، نه قابلیت‌های اختصاصی که Core باید برای Qwen3 بشناسد.

## ModelCapabilities

AI Core باید قابلیت‌های مدل و Runtime فعال را از طریق یک Contract عمومی دریافت کند. شکل مفهومی Contract:

```kotlin
data class ModelCapabilities(
    val contextLength: Int?,
    val supportsThinking: Boolean,
    val supportsToolCalling: Boolean,
    val supportsStreaming: Boolean,
    val supportedSampling: Set<SamplingParameter>,
    val supportsStopSequences: Boolean,
    ...
)
```

فهرست نهایی فیلدها باید بر اساس قابلیت‌های واقعی Runtime و مدل تکمیل شود و نباید بدون پشتیبانی واقعی مقداردهی حدسی شود.

## Capability source

قابلیت‌ها باید در اولویت از اطلاعات واقعی زیر به‌دست آیند:

1. Model metadata / GGUF metadata، در صورت وجود اطلاعات قابل اعتماد.
2. Runtime capabilities، در صورتی که Runtime اطلاعات دقیق‌تری ارائه کند.
3. ترکیب metadata و Runtime با یک policy مشخص برای حل اختلاف.
4. در صورت نبود اطلاعات قابل اعتماد، وضعیت باید `Unknown`/`Unsupported` باشد و سیستم نباید قابلیت را حدس بزند.

## Generic inference settings

`InferenceSettings` باید model-agnostic باقی بماند. وجود یک setting در Contract به معنی پشتیبانی آن توسط همه مدل‌ها نیست.

Runtime باید بتواند مشخص کند کدام sampling parameterها را پشتیبانی می‌کند. برای نمونه:

- Temperature
- Top-P
- Top-K
- Min-P
- Repeat/Penalty settings
- Seed
- Stop sequences
- Context length

`Min-P` بخشی از Contract عمومی باقی می‌ماند؛ پشتیبانی یا عدم پشتیبانی آن باید با Capability مشخص شود. این setting نباید فقط به دلیل تفاوت Runtimeها حذف شود.

## Thinking

Thinking یک capability اختیاری است.

UI و Agent نباید فرض کنند همه مدل‌ها thinking دارند. اگر `supportsThinking=false` یا وضعیت capability نامشخص باشد، کنترل thinking نباید به‌صورت فعال و جعلی نمایش داده شود.

برای مدل‌هایی مانند Qwen3 که thinking را پشتیبانی می‌کنند، Runtime باید Contract لازم برای فعال/غیرفعال‌کردن آن را ارائه کند. استفاده از `/think`، `/no_think` یا مکانیزم دیگری باید implementation detail Runtime/adapter باشد، نه شرط hard-coded در UI یا Agent.

## Tool calling / Actions

پشتیبانی مدل از tool calling با قابلیت اجرای Action در سیستم یکی نیست.

- `supportsToolCalling` بیانگر قابلیت مدل/Runtime برای تولید یا پردازش tool-call به شکل پشتیبانی‌شده است.
- Action Registry، Permission، Validation، Confirmation، Executor و Verifier همچنان مسئولیت Core هستند.
- اگر مدل tool calling native نداشته باشد، Agent می‌تواند در صورت وجود Contract مناسب از Action protocol پروژه استفاده کند.
- هیچ مسیر اجرای Action نباید صرفاً به این دلیل که مدل ادعا کرده tool call ساخته است، بدون Parser/Validator/Security/Verifier اجرا شود.

## Context

`contextLength` باید از مدل/Runtime واقعی گرفته شود.

Core نباید مقدار 32768 یا هر مقدار دیگری را به‌عنوان حد ثابت Qwen3 در خود hard-code کند. مدل آینده ممکن است context کوچک‌تر یا بزرگ‌تر داشته باشد.

Context builder باید با توجه به capability واقعی، تنظیمات کاربر و منابع Context تصمیم بگیرد چه مقدار از:

```text
System Context
+ Persistent Task Context
+ Conversation Summary
+ Recent Messages
+ Workspace Context
```

وارد prompt شود.

## Streaming

Streaming نیز یک capability است. اگر Runtime آن را پشتیبانی کند، UI باید از stream واقعی استفاده کند؛ اگر پشتیبانی نکند، Core باید بتواند با نتیجه non-streaming کار کند بدون اینکه معماری Chat/Agent تغییر کند.

## Agent adaptation

Agent باید capability-aware باشد، نه model-aware.

ممنوع:

```kotlin
if (modelName == "Qwen3-1.7B") { ... }
```

مجاز:

```kotlin
if (capabilities.supportsThinking) { ... }
```

یا تصمیم‌گیری بر اساس Contractهای عمومی و وضعیت واقعی Runtime.

Step budget، Action execution، context management و retry policy نیز باید مستقل از نام مدل طراحی شوند.

## UI adaptation

UI باید بر اساس Capabilityها کنترل‌ها را نمایش یا محدود کند:

- setting پشتیبانی‌شده → قابل تنظیم
- setting پشتیبانی‌نشده → disabled/hidden با وضعیت روشن
- capability نامشخص → از ادعای پشتیبانی خودداری شود
- context limit → نمایش بر اساس مقدار واقعی مدل/Runtime
- thinking → فقط در صورت پشتیبانی واقعی
- streaming → فقط در صورت پشتیبانی واقعی

UI نباید برای یک مدل خاص طراحی شود و سپس مدل‌های دیگر را مجبور به تقلید از همان قابلیت‌ها کند.

## Import and future models

Import یک GGUF جدید نباید نیازمند بازطراحی AI Core باشد.

فرآیند مرجع:

```text
Import Model
    ↓
Read/Validate Metadata
    ↓
Create Model Record
    ↓
Resolve Runtime
    ↓
Resolve ModelCapabilities
    ↓
Expose Generic Model/Inference API
    ↓
UI + Agent adapt to capabilities
```

مدل‌های آینده می‌توانند شامل نسخه‌های بزرگ‌تر Qwen، Gemma، Llama، SmolLM یا مدل‌های دیگر باشند. اضافه‌شدن مدل جدید نباید باعث ایجاد branchهای اختصاصی در Core شود.

## Runtime independence

Model capability و Runtime capability باید از هم قابل تشخیص باشند. یک مدل ممکن است از نظر تئوری قابلیتی داشته باشد اما Runtime فعلی آن را expose نکند.

بنابراین Capability نهایی مورد استفاده UI/Agent باید قابلیت **واقعاً قابل استفاده در Runtime فعال** را نشان دهد، نه صرفاً قابلیت ادعاشده در model card.

```text
Model Metadata
       +
Runtime Capabilities
       ↓
Effective ModelCapabilities
       ↓
AI Core / Agent / UI
```

## Non-negotiable rules

1. Qwen3-1.7B Q6_K مدل مرجع است، نه مدل hard-coded.
2. هیچ `if model == ...` برای کنترل رفتار Core/Agent/UI مجاز نیست.
3. Capabilityها باید از metadata/runtime واقعی استخراج شوند.
4. Capability نامشخص نباید به‌صورت پشتیبانی‌شده فرض شود.
5. Inference settings عمومی هستند؛ پشتیبانی هر setting با Capability مشخص می‌شود.
6. Thinking، Tool Calling، Streaming و Context Length قابلیت‌های اختیاری هستند.
7. Action security مستقل از native tool-calling مدل باقی می‌ماند.
8. Import مدل جدید نباید نیازمند بازطراحی AI Core باشد.
9. Adapter مسئول ترجمه تفاوت‌های Runtime است؛ UI و Core نباید implementation-specific شوند.
10. تست‌های Qwen3 باید به‌عنوان reference tests در کنار تست‌های generic capability contract استفاده شوند.
