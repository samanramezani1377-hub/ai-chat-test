package com.woogit.aicore.agent

import com.woogit.aicore.domain.ActionArgumentType
import com.woogit.aicore.domain.ActionRegistry
import com.woogit.aicore.domain.RiskLevel

/**
 * Builds the system instructions from the actions actually registered at runtime.
 * The prompt is informational only; ActionLifecycle remains the authority for validation,
 * permissions, approval, execution, and verification.
 */
object ActionToolPrompt {
    fun build(registry: ActionRegistry): String = buildString {
        appendLine("تو یک دستیار محلی هستی و فقط می‌توانی از ابزارهایی استفاده کنی که در فهرست واقعی زیر آمده‌اند.")
        appendLine("اگر برای پاسخ به اجرای ابزار نیاز داری، فقط یک شیء JSON مطابق قرارداد ActionRequest v1 برگردان؛ هیچ متن، Markdown یا code fence بیرون JSON ننویس.")
        appendLine("قالب دقیق فراخوانی:")
        appendLine("""{"version":1,"actionId":"call-1","action":"calculate","arguments":{"expression":"2+2"}}""")
        appendLine("actionId شناسه یکتای این فراخوانی است؛ action باید دقیقاً شناسه یکی از ابزارهای فهرست‌شده باشد؛ arguments باید یک شیء JSON با پارامترهای همان ابزار باشد.")
        appendLine("در هر پاسخ حداکثر یک ابزار را فراخوانی کن. پس از دریافت نتیجه واقعی ابزار، در صورت نیاز مرحله بعد را انتخاب کن.")
        appendLine("اگر به ابزار نیاز نیست، پاسخ عادی و قابل‌فهم به کاربر بده. هرگز ادعا نکن ابزاری اجرا شده مگر نتیجه آن را در پیام tool_response دریافت کرده باشی.")
        appendLine("مسیر فایل‌ها باید نسبی به Workspace برنامه باشد؛ به مسیرهای بیرون Workspace دسترسی نداری. ابزار شبکه یا جست‌وجوی وب در فهرست زیر وجود ندارد مگر صریحاً نمایش داده شده باشد.")
        appendLine("عملیات حساس مانند حذف فایل فقط از مسیر تأیید صریح کاربر قابل اجراست؛ تأیید را دور نزن.")
        appendLine()
        appendLine("ابزارهای فعال در همین اجرای برنامه:")
        val actions = registry.all().sortedBy { it.id }
        if (actions.isEmpty()) {
            appendLine("(هیچ ابزاری در Registry ثبت نشده است؛ هیچ ActionRequest تولید نکن.)")
        } else {
            actions.forEach { action ->
                appendLine("- ${action.id}: ${action.description}")
                appendLine("  risk=${action.risk.name}; permission=${action.permission}")
                if (action.schema.arguments.isEmpty()) {
                    appendLine("  arguments: {}")
                } else {
                    appendLine("  arguments:")
                    action.schema.arguments.forEach { arg ->
                        val type = when (arg.type) {
                            ActionArgumentType.STRING -> "string"
                            ActionArgumentType.INTEGER -> "integer"
                            ActionArgumentType.NUMBER -> "number"
                            ActionArgumentType.BOOLEAN -> "boolean"
                            ActionArgumentType.OBJECT -> "object"
                            ActionArgumentType.ARRAY -> "array"
                        }
                        val required = if (arg.required) "required" else "optional"
                        val max = arg.maxLength?.let { ", maxLength=$it" }.orEmpty()
                        appendLine("    - ${arg.name}: $type, $required$max")
                    }
                }
                if (action.risk == RiskLevel.SENSITIVE) appendLine("  requires explicit user approval before execution.")
            }
        }
    }.trim()
}
