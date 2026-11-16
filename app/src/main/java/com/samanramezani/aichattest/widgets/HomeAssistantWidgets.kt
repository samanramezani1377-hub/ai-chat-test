package com.samanramezani.aichattest.widgets

import android.app.PendingIntent
import android.appwidget.AppWidgetManager
import android.appwidget.AppWidgetProvider
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.widget.RemoteViews
import com.samanramezani.aichattest.R
import com.samanramezani.aichattest.widgets.ui.WidgetComposeActivity

internal object HomeWidgetState {
    private const val PREFS = "home_assistant_widgets"
    private const val CHAT_STATUS = "chat_status"
    private const val CHAT_REPLY = "chat_reply"
    private const val VOICE_STATUS = "voice_status"

    fun chatStatus(context: Context): String =
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getString(CHAT_STATUS, "آماده") ?: "آماده"

    fun chatReply(context: Context): String =
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .getString(CHAT_REPLY, "پیامت را از همین ویجت بفرست و پاسخ را همین‌جا بگیر.")
            ?: "پیامت را از همین ویجت بفرست و پاسخ را همین‌جا بگیر."

    fun voiceStatus(context: Context): String =
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .getString(VOICE_STATUS, "برای یک نوبت صحبت لمس کن") ?: "برای یک نوبت صحبت لمس کن"

    fun updateChat(context: Context, status: String, reply: String? = null) {
        val editor = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().putString(CHAT_STATUS, status)
        if (reply != null) editor.putString(CHAT_REPLY, reply.take(1800))
        editor.apply()
        updateChatContent(context)
    }

    fun updateVoice(context: Context, status: String) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().putString(VOICE_STATUS, status).apply()
        updateVoiceContent(context)
    }

    private fun updateChatContent(context: Context) {
        val manager = AppWidgetManager.getInstance(context)
        val ids = manager.getAppWidgetIds(ComponentName(context, ChatAssistantWidgetProvider::class.java))
        if (ids.isEmpty()) return
        val partial = RemoteViews(context.packageName, R.layout.widget_chat_assistant).apply {
            setTextViewText(R.id.widget_chat_status, chatStatus(context))
            setTextViewText(R.id.widget_chat_reply, chatReply(context))
        }
        manager.partiallyUpdateAppWidget(ids, partial)
    }

    private fun updateVoiceContent(context: Context) {
        val manager = AppWidgetManager.getInstance(context)
        val ids = manager.getAppWidgetIds(ComponentName(context, VoiceAssistantWidgetProvider::class.java))
        if (ids.isEmpty()) return
        val partial = RemoteViews(context.packageName, R.layout.widget_voice_assistant).apply {
            setTextViewText(R.id.widget_voice_status, voiceStatus(context))
        }
        manager.partiallyUpdateAppWidget(ids, partial)
    }

    fun renderChat(context: Context) {
        val manager = AppWidgetManager.getInstance(context)
        val ids = manager.getAppWidgetIds(ComponentName(context, ChatAssistantWidgetProvider::class.java))
        if (ids.isEmpty()) return
        val views = RemoteViews(context.packageName, R.layout.widget_chat_assistant).apply {
            setTextViewText(R.id.widget_chat_status, chatStatus(context))
            setTextViewText(R.id.widget_chat_reply, chatReply(context))
            val intent = Intent(context, WidgetComposeActivity::class.java)
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP)
            val pending = PendingIntent.getActivity(
                context, 7301, intent,
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
            )
            setOnClickPendingIntent(R.id.widget_chat_compose, pending)
        }
        manager.updateAppWidget(ids, views)
    }

    fun renderVoice(context: Context) {
        val manager = AppWidgetManager.getInstance(context)
        val ids = manager.getAppWidgetIds(ComponentName(context, VoiceAssistantWidgetProvider::class.java))
        if (ids.isEmpty()) return
        val views = RemoteViews(context.packageName, R.layout.widget_voice_assistant).apply {
            setTextViewText(R.id.widget_voice_status, voiceStatus(context))
            val intent = Intent(context, HomeWidgetAssistantService::class.java)
                .setAction(HomeWidgetAssistantService.ACTION_VOICE_TURN)
            val pending = PendingIntent.getForegroundService(
                context, 7302, intent,
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
            )
            setOnClickPendingIntent(R.id.widget_voice_start, pending)
        }
        manager.updateAppWidget(ids, views)
    }
}

class ChatAssistantWidgetProvider : AppWidgetProvider() {
    override fun onUpdate(context: Context, manager: AppWidgetManager, appWidgetIds: IntArray) {
        HomeWidgetState.renderChat(context)
    }
}

class VoiceAssistantWidgetProvider : AppWidgetProvider() {
    override fun onUpdate(context: Context, manager: AppWidgetManager, appWidgetIds: IntArray) {
        HomeWidgetState.renderVoice(context)
    }
}
