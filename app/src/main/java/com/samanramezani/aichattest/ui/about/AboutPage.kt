package com.samanramezani.aichattest.ui.about

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

@Composable
internal fun AboutPage() {
    SimplePage("درباره برنامه") {
        Text("نسخه و مشخصات برنامه", style = MaterialTheme.typography.titleLarge)
        Text("رابط کاربری فارسی و RTL بر اساس قرارداد UI v1.0.0.", color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text("مدیریت مدل محلی از مسیر تنظیمات → هوش مصنوعی → مدل انجام می‌شود.", color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@Composable
private fun SimplePage(title: String, content: @Composable ColumnScope.() -> Unit) {
    LazyColumn(Modifier.fillMaxSize().padding(20.dp), contentPadding = PaddingValues(bottom = 32.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        item { Text(title, style = MaterialTheme.typography.headlineMedium) }
        item { Column(verticalArrangement = Arrangement.spacedBy(10.dp), content = content) }
    }
}
