package com.samanramezani.aichattest.ui.diagnostics

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.widget.Toast
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.samanramezani.aichattest.CrashLogStore
import com.samanramezani.aichattest.ui.components.SimplePage

@Composable
internal fun CrashLogPage() {
    val context = LocalContext.current
    val report by CrashLogStore.report.collectAsState()

    SimplePage("لاگ کرش") {
        Column(
            Modifier.fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.secondaryContainer),
            ) {
                Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("ثبت و بررسی توقف غیرعادی برنامه", style = MaterialTheme.typography.titleLarge)
                    Text(
                        "خطاهای Java/Kotlin و کرش‌های Native پس از راه‌اندازی مجدد اینجا ثبت می‌شوند. گزارش فقط اطلاعات مهم، نتایج موفق کلیدی، مرحلهٔ توقف و backtrace سیستم (در صورت دسترسی) را نگه می‌دارد؛ لاگ‌های تکراری و جزئیات غیرضروری حذف می‌شوند.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSecondaryContainer,
                    )
                }
            }

            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(
                    onClick = {
                        val value = report
                        if (value.isNullOrBlank()) {
                            Toast.makeText(context, "هنوز گزارشی از کرش ثبت نشده است.", Toast.LENGTH_SHORT).show()
                        } else {
                            copyCrashReport(context, value)
                        }
                    },
                    modifier = Modifier.weight(1f),
                    enabled = !report.isNullOrBlank(),
                ) {
                    androidx.compose.material3.Icon(Icons.Default.ContentCopy, contentDescription = null)
                    Spacer(Modifier.width(6.dp))
                    Text("کپی گزارش کامل")
                }
                OutlinedButton(onClick = { CrashLogStore.refresh(context) }) {
                    androidx.compose.material3.Icon(Icons.Default.Refresh, contentDescription = null)
                    Text("تازه‌سازی")
                }
            }

            if (report.isNullOrBlank()) {
                Card(Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        Text("هنوز کرشی ثبت نشده است.", style = MaterialTheme.typography.titleMedium)
                        Text(
                            "اگر برنامه به‌صورت غیرمنتظره بسته شود، گزارش در اجرای بعدی به این بخش اضافه می‌شود. برای کرش Native لازم است برنامه را دوباره باز کنی تا رکورد سیگنال از فایل بومی به گزارش قابل‌کپی منتقل شود.",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
            } else {
                Card(Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(14.dp)) {
                        Text("گزارش‌های ذخیره‌شده", style = MaterialTheme.typography.titleMedium)
                        Spacer(Modifier.height(8.dp))
                        Text(
                            report.orEmpty(),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurface,
                        )
                    }
                }
                Text(
                    "توجه: برای کرش Native، نوع سیگنال و آخرین مرحله اجرا قابل ثبت است؛ علت دقیق خط کد فقط وقتی قطعی می‌شود که Stack Trace/Tombstone و Build ID متناظر در دسترس باشد. گزارش بین شواهد قطعی و احتمال‌ها تفکیک می‌کند.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

private fun copyCrashReport(context: Context, report: String) {
    val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as? ClipboardManager ?: return
    clipboard.setPrimaryClip(ClipData.newPlainText("AI Chat Test Crash Log", report))
    Toast.makeText(context, "گزارش کامل کرش کپی شد.", Toast.LENGTH_SHORT).show()
}
