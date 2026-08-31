# AI Chat Test

## هدف پروژه

این repository یک **Prototype / Technical Feasibility Test** برای اجرای هوش مصنوعی کاملاً Local روی Android است.

هدف این پروژه ساخت محصول نهایی نیست. هدف این است که قبل از اضافه کردن قابلیت Local AI به پروژه **WooGit**، از نظر فنی بررسی کنیم که اجرای یک مدل زبانی روی گوشی، چت فارسی، Streaming، و یک Agent ساده با Tool/Action تا چه حد عملی و قابل اتکا است.

## تصمیم فعلی مدل

مدل پایه انتخاب‌شده برای Prototype:

- **Qwen3-1.7B**
- فرمت: **GGUF**
- Quantization اولیه: **Q4_K_M**
- Runtime پیشنهادی: **llama.cpp**
- مدل داخل APK قرار نمی‌گیرد.
- مدل به‌صورت فایل جداگانه توسط کاربر Import می‌شود.

### چرا Qwen3-1.7B؟

در تست کیفی اولیه، Qwen3-0.6B برای هدف پروژه ضعیف بود؛ حتی در چت آزاد فارسی کیفیت موردنیاز را ارائه نکرد. در مقابل، Qwen3-1.7B در چت آزاد فارسی قابل قبول‌تر بود و در تست Agent نیز توانست Action مناسب و پارامترهای آن را تولید کند.

نمونه خروجی موفق Agent:

```json
{
  "tool": "create_file",
  "file_name": "test.txt",
  "content": "Hello World"
}
```

پس فعلاً Qwen3-1.7B به‌عنوان **Baseline Model** انتخاب شده است.

> این انتخاب برای Prototype است و پس از تست واقعی روی Android می‌تواند تغییر کند. در آینده باید امکان Import مدل‌های بزرگ‌تر، مانند Qwen3-4B، بدون بازطراحی اساسی اپ وجود داشته باشد.

## معماری هدف

معماری عمداً ساده نگه داشته می‌شود:

```text
Android UI
   │
   ├── ModelManager
   │      └── Import / Validate / Load / Unload
   │
   ├── AIEngine
   │      └── llama.cpp / GGUF
   │
   └── AgentManager
          ├── AgentParser
          └── ActionExecutor
```

## قابلیت‌های Prototype

### 1. Import Model

کاربر بتواند فایل `.gguf` را از طریق Android Storage Access Framework انتخاب و وارد برنامه کند.

مدل باید خارج از APK نگهداری شود.

از مسیرهای hardcoded مانند `/Android/data/...` استفاده نشود؛ مسیر و دسترسی فایل باید با APIهای استاندارد Android مدیریت شود.

### 2. Model Management

حداقل عملیات:

- Import model
- Validate model
- Load model
- Unload model
- نمایش مدل‌های Import‌شده
- نمایش وضعیت مدل

نام مدل نباید در تمام کد hardcode شود؛ مدل باید قابل تعویض باشد.

### 3. Local Chat

اپ باید بتواند پس از Load مدل:

- پیام کاربر را دریافت کند.
- Inference را کاملاً روی دستگاه اجرا کند.
- خروجی را به‌صورت Streaming نمایش دهد.
- Generation را متوقف کند.

هیچ API یا سرویس Cloud برای Inference استفاده نشود.

### 4. Agent / Action Test

مدل باید بتواند در صورت نیاز یک Action ساختاریافته درخواست کند.

چرخه هدف:

```text
User Request
     ↓
LLM
     ↓
Action / Tool Request
     ↓
ActionExecutor
     ↓
Tool Result
     ↓
LLM
     ↓
Final Answer
```

برای Prototype فقط چند Action ساده کافی است؛ مثلاً:

- `calculate`
- `get_time`
- `create_file`

مدل نباید مستقیماً APIهای Android را صدا بزند. فقط درخواست ساختاریافته تولید می‌کند و اپ تصمیم می‌گیرد چه Actionی اجرا شود.

### 5. Agent Safety Rules

- Action ناشناخته اجرا نشود.
- مدل نباید قبل از اجرای واقعی Action ادعای `success` کند.
- نتیجه Tool فقط پس از اجرای واقعی Action به مدل برگردانده شود.
- Agent Loop محدود باشد؛ برای Prototype حداکثر 3 مرحله کافی است.
- Tool result و final answer از هم تفکیک شوند.

## تست‌های پایه

### Chat فارسی

```text
سلام، خودت را معرفی کن و بگو چه کارهایی می‌توانی انجام بدهی.
```

### محاسبه

```text
125 × 37 چند می‌شود؟
```

### Action

```text
یک فایل متنی با نام test.txt بساز و داخل آن دقیقاً عبارت Hello World را قرار بده.
```

### Tool Result

پس از اجرای واقعی Action، نتیجه‌ای مانند زیر به مدل داده شود:

```json
{
  "success": true,
  "file_name": "test.txt"
}
```

سپس مدل باید یک پاسخ نهایی کوتاه تولید کند.

### WooCommerce-style Action

```text
یک محصول جدید ایجاد کن با عنوان «تشک طبی فنری»، قیمت ۵ میلیون تومان، موجودی ۱۲ عدد و وضعیت پیش‌نویس.
```

هدف این تست بررسی تبدیل دستور طبیعی به درخواست ساختاریافته است؛ فعلاً هیچ اتصال واقعی به WooCommerce لازم نیست.

## معیارهای موفقیت

Prototype زمانی موفق تلقی می‌شود که روی یک Android واقعی بتوانیم این زنجیره را اجرا کنیم:

```text
Import Qwen3-1.7B GGUF
        ↓
Load locally
        ↓
Chat
        ↓
Streaming inference
        ↓
Agent action request
        ↓
Tool execution
        ↓
Tool result
        ↓
Final answer
```

و در حالت Offline نیز کار کند.

## Performance Metrics

در تست واقعی باید حداقل این موارد اندازه‌گیری شوند:

- Model load time
- Generation speed (tokens/sec)
- Total generated tokens
- Context length
- Memory usage در صورت امکان
- زمان پاسخ اولیه / first token در صورت امکان
- پایداری هنگام Generation طولانی

نمونه گزارش:

```text
Model: Qwen3-1.7B Q4_K_M
Device: <device>
Load time: <value>
First token: <value>
Generation: <value> tok/s
Tokens: <value>
Memory: <value>
Offline: PASS/FAIL
Agent: PASS/FAIL
```

## محدودیت‌های فعلی

فعلاً این موارد خارج از محدوده Prototype هستند:

- Cloud AI
- API خارجی
- Multi-agent
- RAG
- Vector database
- Voice
- Image generation
- حساب کاربری
- Marketplace مدل
- سیستم Plugin پیچیده
- Autonomous background agent
- تعداد زیاد Tool

## مسیر توسعه

### Phase 1 — Local Inference

1. Android project setup
2. llama.cpp integration
3. GGUF model import
4. Model validation
5. Model loading
6. Local inference
7. Streaming
8. Stop generation

### Phase 2 — Agent

1. Action schema
2. Agent parser
3. Action executor
4. Tool result
5. محدود کردن Agent loop
6. نمایش وضعیت Agent در UI

### Phase 3 — Performance

1. تست روی گوشی واقعی
2. اندازه‌گیری tok/s
3. اندازه‌گیری Load Time
4. بررسی RAM
5. بررسی Context
6. بررسی پایداری

### Phase 4 — تصمیم برای WooGit

پس از تکمیل تست، بر اساس کیفیت، سرعت، RAM، حجم مدل و پایداری تصمیم می‌گیریم که آیا قابلیت Local AI Agent ارزش انتقال به **WooGit** را دارد یا خیر.

## اصل مهم توسعه

این پروژه یک آزمایش فنی است. از پیچیده‌سازی غیرضروری خودداری شود.

هر تغییر باید:

1. با هدف Prototype مرتبط باشد.
2. Build پروژه را خراب نکند.
3. قابلیت‌های قبلی را بدون دلیل حذف نکند.
4. نتیجه واقعی تست را گزارش کند.
5. از Mock یا گزارش جعلی عملکرد استفاده نکند.

## وضعیت فعلی

**Baseline Model:** Qwen3-1.7B  
**Format:** GGUF  
**Initial Quantization:** Q4_K_M  
**Runtime:** llama.cpp  
**Purpose:** Android Local AI + Agent Feasibility Test  
**Next milestone:** اجرای واقعی Qwen3-1.7B روی Android و اندازه‌گیری عملکرد
