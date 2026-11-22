# Android GPU Inference Engine

این شاخه فقط شامل موتور استنتاج GPU برای Android، مستندات آن، تست‌های مستقل و یک اپلیکیشن آزمایشی بسیار ساده است.

## ساختار

- `engine/`: کتابخانه Android و هسته C++ موتور.
- `engine/src/main/cpp/gpu_engine/`: GGUF، tokenizer، quantization، graph، memory planner، scheduler، inference session و backendها.
- `test-app/`: اپ آزمایشی؛ فقط واردکردن فایل GGUF، تلاش برای فعال‌سازی با GPU و نمایش/کپی گزارش.
- `docs/gpu-engine/`: وضعیت، معماری و نقشه راه موتور.
- `docs/USAGE_FA.md`: راهنمای کامل ساخت و استفاده.
- `scripts/`: تست قراردادهای C++ و بررسی استقلال موتور.

## اصول پروژه

- اجرای GPU از طریق موتور اختصاصی انجام می‌شود.
- Vulkan غیرفعال است.
- backendهای فعلی: Dawn/WebGPU با مسیر OpenGL ES و backend مستقل Universal GLES.
- ABI فعلی Android: فقط `arm64-v8a`.
- موتور برای مدل GGUF طراحی شده است.
- هیچ مسیر CPU یا llama.cpp در این شاخه وجود ندارد.
- مدل، معماری، tokenizer، quantization و kernel تا زمان عبور از preflight و اعتبارسنجی مربوطه مجاز به اجرا نیستند؛ رفتار موتور در موارد پشتیبانی‌نشده fail-closed است.

## وضعیت پشتیبانی

وجود decoder یا kernel در کد به‌تنهایی به معنی پشتیبانی کامل نیست. پشتیبانی هر ترکیب مدل، tokenizer، quantization و backend باید با اعتبارسنجی عددی و اجرای واقعی تأیید شود. در حال حاضر نباید این پروژه را موتور عمومی تأییدشده برای تمام GPUها یا تمام مدل‌ها تلقی کرد. اجرای end-to-end روی سخت‌افزار واقعی هنوز باید جداگانه گزارش و تأیید شود.

## ساخت و تست

پیش‌نیازها: JDK 17، Gradle 8.13، Android SDK Platform 36، NDK 27.2.12479018، Python 3 و Go 1.24 برای تولید کد وابستگی Dawn.

تست مستقل هسته:

```bash
bash scripts/test_gpu_engine_contracts.sh
bash scripts/verify_gpu_engine_independence.sh
```

ساخت APK آزمایشی:

```bash
gradle :test-app:assembleDebug
```

APK در مسیر زیر ساخته می‌شود:

```
test-app/build/outputs/apk/debug/test-app-debug.apk
```

راهنمای کامل: [docs/USAGE_FA.md](docs/USAGE_FA.md)
