package com.samanramezani.aichattest

import android.app.Activity
import android.os.Bundle
import android.widget.TextView

class MainActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(
            TextView(this).apply {
                text = "AI Chat Test"
                textSize = 24f
                setPadding(48, 48, 48, 48)
            }
        )
    }
}
