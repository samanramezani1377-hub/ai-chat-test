package com.samanramezani.aichattest

import android.app.Activity
import android.os.Bundle
import android.view.Gravity
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView

class MainActivity : Activity() {
    private lateinit var container: AppContainer

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        container = (application as? AIChatApplication)?.container ?: AppContainer()
        showWorkspace()
    }

    private fun showWorkspace() {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(40, 48, 40, 32)
        }

        val title = TextView(this).apply {
            text = "AI Chat Test"
            textSize = 26f
        }
        root.addView(title)

        val status = TextView(this).apply {
            text = "Core متصل است • Runtime هنوز متصل نشده"
            textSize = 16f
            setPadding(0, 16, 0, 24)
        }
        root.addView(status)

        root.addView(sectionButton("Chat") { showPlaceholder("Chat", "صفحه Chat در مرحله بعد به ModelRuntime متصل می‌شود.") })
        root.addView(sectionButton("Workspace") { showPlaceholder("Workspace", "میز کار آماده اتصال به Task و Action Core است.") })
        root.addView(sectionButton("Errors") { showPlaceholder("Error Center", "مرکز خطا از Central Observability تغذیه خواهد شد.") })
        root.addView(sectionButton("Settings") { showPlaceholder("Settings", "تنظیمات از Central Settings Core مصرف خواهد شد.") })

        setContentView(root)
    }

    private fun sectionButton(label: String, action: () -> Unit) = Button(this).apply {
        text = label
        setOnClickListener { action() }
    }

    private fun showPlaceholder(title: String, message: String) {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER
            setPadding(40, 40, 40, 40)
        }
        root.addView(TextView(this).apply { text = title; textSize = 24f })
        root.addView(TextView(this).apply { text = message; textSize = 16f; setPadding(0, 20, 0, 20) })
        root.addView(Button(this).apply { text = "بازگشت"; setOnClickListener { showWorkspace() } })
        setContentView(root)
    }
}
