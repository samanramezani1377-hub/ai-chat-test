package com.samanramezani.aichattest.voice

import android.content.Context
import android.content.Intent
import android.media.AudioManager
import android.net.Uri
import android.provider.Settings
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

internal enum class DinaIntent {
    CHAT, GET_TIME, GET_DATE, INCREASE_VOLUME, DECREASE_VOLUME, SET_VOLUME, MUTE, UNMUTE,
    INCREASE_BRIGHTNESS, DECREASE_BRIGHTNESS, SET_BRIGHTNESS
}
internal data class DinaCommand(val intent: DinaIntent, val value: Int? = null, val valueIsPercent: Boolean = false)

internal object DinaCommandClassifier {
    private fun normalize(input: String): String = input
        .replace('ي', 'ی').replace('ى', 'ی').replace('ك', 'ک')
        .replace('۰', '0').replace('۱', '1').replace('۲', '2').replace('۳', '3')
        .replace('۴', '4').replace('۵', '5').replace('۶', '6').replace('۷', '7')
        .replace('۸', '8').replace('۹', '9')
        .replace('٠', '0').replace('١', '1').replace('٢', '2').replace('٣', '3')
        .replace('٤', '4').replace('٥', '5').replace('٦', '6').replace('٧', '7')
        .replace('٨', '8').replace('٩', '9')
        .lowercase(Locale.ROOT).replace(Regex("[\\u0640\\s]+"), " ").trim()

    fun classify(raw: String): DinaCommand {
        val text = normalize(raw)
        if (text.isBlank()) return DinaCommand(DinaIntent.CHAT)

        if (Regex("(ساعت چنده|ساعت چند است|ساعت الان|الان ساعت|ساعت رو بگو|چه ساعتیه?|زمان فعلی|time\\b)").containsMatchIn(text)) {
            return DinaCommand(DinaIntent.GET_TIME)
        }
        if (Regex("(امروز چندمه|تاریخ امروز|امروز چه تاریخ|تاریخ چنده|تاریخ امروز رو بگو)").containsMatchIn(text)) {
            return DinaCommand(DinaIntent.GET_DATE)
        }

        val volume = Regex("(صدا|ولوم|volume|بلندی صدا)").containsMatchIn(text)
        val brightness = Regex("(نور صفحه|روشنایی صفحه|نور نمایشگر|روشنایی|brightness)").containsMatchIn(text)
        val mute = Regex("(قطع صدا|صدا رو قطع|بی.?صدا|ساکت کن|mute|silence)").containsMatchIn(text)
        val unmute = Regex("(وصل صدا|صدا رو وصل|از بی.?صدا|از سکوت خارج|unmute|unsilence)").containsMatchIn(text)
        val decrease = Regex("(کم کن|کمتر کن|پایین بیار|کاهش بده|بیار پایین|کم شه|کمتر شه|کمش کن|decrease|turn down)").containsMatchIn(text)
        val increase = Regex("(زیاد کن|بیشتر کن|بالا ببر|افزایش بده|ببر بالا|زیاد شه|بیشتر شه|زیادش کن|increase|turn up)").containsMatchIn(text)
        val targetMatch = Regex("(?:روی|رو|به)\\s*(\\d{1,3})\\s*(درصد|درجه)?").find(text)
            ?: Regex("(?:بذار|بزار|بگذار|تنظیم(?:ش)? کن|قرار بده|set)\\s+(?:[^0-9]{0,24})?(\\d{1,3})\\s*(درصد|درجه)?").find(text)
        val target = targetMatch?.groupValues?.getOrNull(1)?.toIntOrNull()
        val explicitPercent = targetMatch?.groupValues?.getOrNull(2) == "درصد" ||
            Regex("\\d{1,3}\\s*درصد").containsMatchIn(text)
        val hasSetCue = Regex("(بذار|بزار|بگذار|تنظیم|قرار بده|روی\\s*\\d|رو\\s*\\d|به\\s*\\d|set\\s)").containsMatchIn(text)

        if (volume) {
            if (unmute) return DinaCommand(DinaIntent.UNMUTE)
            if (mute) return DinaCommand(DinaIntent.MUTE)
            if (hasSetCue && target != null && target in 0..100) {
                return DinaCommand(DinaIntent.SET_VOLUME, target, explicitPercent)
            }
            if (decrease) return DinaCommand(DinaIntent.DECREASE_VOLUME)
            if (increase) return DinaCommand(DinaIntent.INCREASE_VOLUME)
        }
        if (brightness) {
            if (hasSetCue && target != null && target in 0..100) {
                return DinaCommand(DinaIntent.SET_BRIGHTNESS, target, true)
            }
            if (decrease) return DinaCommand(DinaIntent.DECREASE_BRIGHTNESS)
            if (increase) return DinaCommand(DinaIntent.INCREASE_BRIGHTNESS)
        }
        return DinaCommand(DinaIntent.CHAT)
    }
}

/** Local deterministic executor. Classification and device mutation are deliberately separate. */
internal class DinaDeviceCommandExecutor(private val context: Context) {
    fun execute(command: DinaCommand): String? = when (command.intent) {
        DinaIntent.CHAT -> null
        DinaIntent.GET_TIME -> {
            val now = SimpleDateFormat("HH:mm", Locale.US).format(Date()).split(":")
            "الان ساعت ${persianNumber(now[0].toInt())} و ${persianNumber(now[1].toInt())} دقیقه است."
        }
        DinaIntent.GET_DATE -> "تاریخ امروز ${SimpleDateFormat("d MMMM yyyy", Locale("fa", "IR")).format(Date())} است."
        DinaIntent.INCREASE_VOLUME -> changeVolume(2)
        DinaIntent.DECREASE_VOLUME -> changeVolume(-2)
        DinaIntent.SET_VOLUME -> setVolume(requireNotNull(command.value), command.valueIsPercent)
        DinaIntent.MUTE -> muteAudio(true)
        DinaIntent.UNMUTE -> muteAudio(false)
        DinaIntent.INCREASE_BRIGHTNESS -> changeBrightness(0.10f)
        DinaIntent.DECREASE_BRIGHTNESS -> changeBrightness(-0.10f)
        DinaIntent.SET_BRIGHTNESS -> setBrightness(requireNotNull(command.value))
    }

    private fun changeVolume(delta: Int): String {
        val audio = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
        val max = audio.getStreamMaxVolume(AudioManager.STREAM_MUSIC)
        val current = audio.getStreamVolume(AudioManager.STREAM_MUSIC)
        val target = (current + delta).coerceIn(0, max)
        audio.setStreamVolume(AudioManager.STREAM_MUSIC, target, 0)
        return "صدای رسانه از ${persianNumber(current)} به ${persianNumber(target)} از ${persianNumber(max)} تغییر کرد."
    }

    private fun muteAudio(mute: Boolean): String {
        val audio = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
        audio.adjustStreamVolume(
            AudioManager.STREAM_MUSIC,
            if (mute) AudioManager.ADJUST_MUTE else AudioManager.ADJUST_UNMUTE,
            0,
        )
        return if (mute) "صدای رسانه قطع شد." else "صدای رسانه از حالت بی‌صدا خارج شد."
    }

    private fun setVolume(requested: Int, isPercent: Boolean): String {
        require(requested in 0..100) { "VOICE-CMD-VOLUME-001: مقدار صدا باید بین صفر تا صد باشد." }
        val audio = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
        val max = audio.getStreamMaxVolume(AudioManager.STREAM_MUSIC)
        val target = if (isPercent || requested > max) (requested * max / 100.0).toInt().coerceIn(0, max) else requested
        audio.setStreamVolume(AudioManager.STREAM_MUSIC, target, 0)
        return "صدای رسانه روی ${persianNumber(target)} از ${persianNumber(max)} تنظیم شد."
    }

    private fun changeBrightness(delta: Float): String {
        if (!Settings.System.canWrite(context)) {
            openBrightnessPermission()
            return "برای تغییر نور صفحه، مجوز تغییر تنظیمات سیستم را در صفحه‌ای که باز شد فعال کن و دوباره بگو."
        }
        val current = Settings.System.getInt(context.contentResolver, Settings.System.SCREEN_BRIGHTNESS, 128)
        val target = (current + (delta * 255).toInt()).coerceIn(0, 255)
        Settings.System.putInt(context.contentResolver, Settings.System.SCREEN_BRIGHTNESS, target)
        return "روشنایی صفحه ${persianNumber((target * 100 / 255.0).toInt())} درصد شد."
    }

    private fun setBrightness(requested: Int): String {
        require(requested in 0..100) { "VOICE-CMD-BRIGHTNESS-001: نور صفحه باید بین صفر تا صد درصد باشد." }
        if (!Settings.System.canWrite(context)) {
            openBrightnessPermission()
            return "برای تنظیم نور صفحه، مجوز تغییر تنظیمات سیستم را در صفحه‌ای که باز شد فعال کن و دوباره بگو."
        }
        val target = (requested * 255 / 100.0).toInt().coerceIn(0, 255)
        Settings.System.putInt(context.contentResolver, Settings.System.SCREEN_BRIGHTNESS, target)
        return "روشنایی صفحه روی ${persianNumber(requested)} درصد تنظیم شد."
    }

    private fun openBrightnessPermission() {
        context.startActivity(Intent(Settings.ACTION_MANAGE_WRITE_SETTINGS, Uri.parse("package:${context.packageName}"))
            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
    }

    private fun persianNumber(number: Int): String {
        val words = arrayOf("صفر", "یک", "دو", "سه", "چهار", "پنج", "شش", "هفت", "هشت", "نه",
            "ده", "یازده", "دوازده", "سیزده", "چهارده", "پانزده", "شانزده", "هفده", "هجده", "نوزده")
        val tens = arrayOf("", "", "بیست", "سی", "چهل", "پنجاه", "شصت", "هفتاد", "هشتاد", "نود")
        if (number < 20) return words[number.coerceAtLeast(0)]
        if (number < 100) return tens[number / 10] + if (number % 10 == 0) "" else " و ${words[number % 10]}"
        if (number < 1000) {
            val hundreds = arrayOf("", "صد", "دویست", "سیصد", "چهارصد", "پانصد", "ششصد", "هفتصد", "هشتصد", "نهصد")
            val remainder = number % 100
            return if (remainder == 0) hundreds[number / 100] else "${hundreds[number / 100]} و ${persianNumber(remainder)}"
        }
        return number.toString()
    }
}
