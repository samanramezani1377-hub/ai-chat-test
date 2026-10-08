package com.samanramezani.aichattest.ui.errors

internal data class PersianError(
    val code: String,
    val title: String,
    val message: String,
    val action: String,
    val technical: String,
)

internal object ErrorCenter {
    fun resolve(raw: String?, operation: String? = null): PersianError {
        val technical = raw?.trim().takeUnless { it.isNullOrBlank() } ?: "بدون جزئیات فنی"
        val text = technical.lowercase()

        return when {
            "model_not_active" in text ->
                PersianError("MODEL-000", "مدل فعالی وجود ندارد",
                    "برای ارسال پیام هنوز هیچ مدل محلی فعال نشده است.",
                    "از بخش تنظیمات یک مدل GGUF را فعال کن و سپس دوباره پیام بفرست.", technical)

            text == "input_empty" || "پیام خالی" in text ->
                PersianError("INPUT-001", "پیام خالی است",
                    "برای ارسال، متن پیام را وارد کن.",
                    "یک پیام بنویس و دوباره ارسال کن.", technical)

            "input_too_large" in text ->
                PersianError("INPUT-002", "پیام بیش از حد بزرگ است",
                    "حجم متن ورودی برای این اجرا بیش از حد مجاز است.",
                    "متن را کوتاه‌تر کن یا آن را در چند پیام بفرست.", technical)

            "OPENCL_GPU_ONLY_REJECTED_NO_OPENCL_GPU_DEVICE".lowercase() in text ||
                "platform ids not available" in text ||
                "no opencl gpu device" in text ->
                PersianError("OPENCL-001", "GPU ‏OpenCL پیدا نشد",
                    "برنامه نتوانست یک دستگاه GPU قابل استفاده از درایور OpenCL پیدا کند.",
                    "درایور OpenCL دستگاه را بررسی کن و برنامه را دوباره اجرا کن. اگر مشکل باقی ماند، گزارش کامل را ارسال کن.",
                    technical)

            "OPENCL_GPU_ONLY_MODEL_RESIDENCY_REJECTED".lowercase() in text ->
                PersianError("OPENCL-002", "وزن‌های مدل روی GPU قرار نگرفتند",
                    "مدل بارگذاری شد، اما بخشی از وزن‌ها روی دستگاه GPU مورد انتظار قرار نگرفت.",
                    "این خطا به مسیر GPU مربوط است و نباید با فعال‌کردن حالت CPU دور زده شود. گزارش کامل را بررسی کن.",
                    technical)

            "unable to load" in text || "MODEL_LOAD_FAILED" in text || "model_load_returned_failed" in text ->
                PersianError("MODEL-001", "بارگذاری مدل ناموفق بود",
                    "فایل مدل قابل شناسایی است، اما Runtime نتوانست آن را برای اجرا آماده کند.",
                    "فایل GGUF و سازگاری معماری مدل با Runtime را بررسی کن؛ سپس دوباره فعال‌سازی را امتحان کن.",
                    technical)

            "gguf_preflight_failed" in text || "file access" in text || "cannot read" in text ||
                "no such file" in text || "permission denied" in text ->
                PersianError("MODEL-002", "دسترسی به فایل مدل مشکل دارد",
                    "فایل مدل وجود ندارد، خواندن آن ممکن نیست یا دسترسی لازم به فایل برقرار نیست.",
                    "مدل را دوباره وارد کن و مطمئن شو فایل کامل و قابل خواندن است.",
                    technical)

            "unsupported model architecture" in text || "unsupported architecture" in text ->
                PersianError("MODEL-003", "معماری مدل پشتیبانی نمی‌شود",
                    "این Runtime معماری این مدل را پشتیبانی نمی‌کند.",
                    "مدلی با معماری پشتیبانی‌شده انتخاب کن یا Runtime مناسب آن معماری را اضافه کن.",
                    technical)

            "out of memory" in text || "oom" in text || "allocation failed" in text ||
                "failed to allocate" in text ->
                PersianError("MEM-001", "حافظه کافی برای اجرای مدل وجود ندارد",
                    "در زمان بارگذاری یا اجرا، حافظه موردنیاز مدل تأمین نشده است.",
                    "مدل سبک‌تر یا زمینه کوتاه‌تر را امتحان کن و برنامه‌های سنگین دیگر را ببند.",
                    technical)

            "context" in text && ("too large" in text || "exceed" in text || "overflow" in text) ->
                PersianError("INPUT-003", "حجم ورودی بیش از ظرفیت زمینه است",
                    "متن و سابقه گفت‌وگو از ظرفیت زمینه مدل بیشتر شده است.",
                    "متن کوتاه‌تری بفرست یا تعداد پیام‌های اخیر/ظرفیت زمینه را کاهش بده.",
                    technical)

            "agent session" in text ->
                PersianError("AGENT-001", "اجرای عامل در دسترس نیست",
                    "جلسه اجرای عامل ایجاد یا بازیابی نشد.",
                    "مدل فعال را بررسی کن و دوباره عملیات را اجرا کن.",
                    technical)

            "approval" in text && ("failed" in text || "error" in text) ->
                PersianError("AGENT-002", "ادامه عملیات تأییدشده ناموفق بود",
                    "عملیات تأیید شد، اما اجرای مرحله بعد کامل نشد.",
                    "دوباره تلاش کن؛ اگر تکرار شد گزارش کامل را بررسی کن.",
                    technical)

            else -> PersianError(
                "APP-001",
                if (operation.isNullOrBlank()) "اجرای عملیات ناموفق بود" else "اجرای $operation ناموفق بود",
                "برنامه با یک خطای ثبت‌شده روبه‌رو شد، اما نوع دقیق آن از متن خطا قابل تشخیص نیست.",
                "گزارش کامل را کپی کن تا جزئیات فنی و رویدادهای قبل از خطا بررسی شوند.",
                technical,
            )
        }
    }
}
