package com.samanramezani.aichattest.screen

/** Pure language-level routing for explicit questions about the currently visible screen. */
internal object ScreenQuestionClassifier {
    fun isScreenQuestion(raw: String): Boolean {
        val q = raw.lowercase().replace('ي', 'ی').replace('ك', 'ک').replace('‌', ' ')
        val directPhrases = listOf(
            "صفحه", "روی گوشی", "روی نمایشگر", "اینجا چی", "اینجا چه", "این قسمت",
            "این دکمه", "این برنامه", "این ارور", "این خطا", "چی نوشته", "چه نوشته",
            "این چیه", "این چیست", "این کجاست", "از روی صفحه", "متن تصویر", "اسکرین",
            "عکس صفحه", "نوشته روی", "ببین چی", "می بینی", "میبینی", "صفحه نمایش", "این پیام",
            "این نوشته", "این گزینه", "این آیکن", "این آیکون",
            "this screen", "on the screen", "what is on", "what's on", "what does this say",
            "read this", "this button", "this app", "screen says", "screen error",
            "what is this", "what's this", "how do i fix this", "what should i click",
            "what should i select", "what does this mean"
        )
        if (directPhrases.any { q.contains(it) }) return true

        val deictic = listOf("اینو", "این رو", "این را", "این", "اینجا", "اونجا", "اون", "همین")
        val asksForInterpretation = listOf(
            "چیه", "چیست", "چطور", "چگونه", "چرا", "معنی", "توضیح", "بخون",
            "بخوان", "نوشته", "میگه", "می گوید", "حل کنم", "بزنم", "انتخاب کنم",
            "کار کنم", "چیکار", "چه کار", "مشکل", "خطا", "ارور", "درسته", "درست است"
        )
        return q.length <= 180 &&
            deictic.any { q.contains(it) } &&
            asksForInterpretation.any { q.contains(it) }
    }
}
