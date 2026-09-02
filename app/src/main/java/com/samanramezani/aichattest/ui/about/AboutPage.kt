package com.samanramezani.aichattest.ui.about

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp

@Composable
internal fun AboutPage() {
    SimplePage("درباره برنامه") {
        SectionCard("AI Chat Test") {
            Text(
                "یک محیط آزمایشی برای اجرای هوش مصنوعی به‌صورت محلی روی دستگاه، همراه با گفت‌وگو، اجرای عملیات و ابزارهای عیب‌یابی.",
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            InfoRow("نوع اجرا", "محلی و روی دستگاه")
            InfoRow("رابط", "فارسی و راست‌به‌چپ (RTL)")
            InfoRow("نسخه قرارداد UI", "v1.0.0")
        }

        SectionCard("قابلیت‌ها") {
            Feature("گفت‌وگوی متنی با مدل محلی")
            Feature("نمایش زنده توکن‌های پاسخ")
            Feature("مدیریت و وارد کردن مدل")
            Feature("Agent و اجرای Action با چرخه تأیید")
            Feature("Workspace برای مشاهده Execution و Timeline")
            Feature("Diagnostics برای بررسی Runtime و عملکرد")
            Feature("اجرای inference در پس‌زمینه با Foreground Service")
        }

        SectionCard("مدیریت مدل") {
            Text(
                "مدل‌های محلی از بخش تنظیمات مدیریت می‌شوند. وارد کردن، فعال‌سازی، غیرفعال‌سازی و حذف مدل از همان بخش انجام می‌شود.",
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }

        SectionCard("امنیت اجرای عملیات") {
            Text(
                "عملیات Agent مستقیماً و بدون کنترل اجرا نمی‌شوند. چرخه اجرای Action شامل آماده‌سازی، اعتبارسنجی، تأیید در صورت نیاز، اجرا، بررسی نتیجه و ثبت وضعیت است.",
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Text(
                "Executionهای قطع‌شده می‌توانند به وضعیت Unknown منتقل شوند تا بدون تأیید و بررسی دوباره، بی‌صدا تکرار نشوند.",
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }

        SectionCard("Workspace و عیب‌یابی") {
            Text(
                "Workspace برای مشاهده وضعیت اجرای جاری و Timeline واقعی Actionهاست. صفحه Diagnostics نیز اطلاعات Runtime، زمان اولین توکن، زمان تولید، توکن‌ها و خطاهای واقعی موجود را نمایش می‌دهد.",
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }

        SectionCard("حریم خصوصی") {
            Text(
                "این برنامه برای inference محلی طراحی شده است. داده‌ای را به‌عنوان قابلیت شبکه‌ای برنامه فرض نمی‌کنیم؛ مسیر واقعی پردازش به Runtime و اجزای فعال برنامه وابسته است.",
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }

        SectionCard("راهنمای سریع") {
            InfoRow("شروع", "مدل را وارد و فعال کنید، سپس وارد گفت‌وگو شوید.")
            InfoRow("Workspace", "برای مشاهده اجرای Actionها و Timeline")
            InfoRow("Diagnostics", "برای بررسی عملکرد و خطاهای Runtime")
            InfoRow("Settings", "برای مدل‌ها و تنظیمات inference")
        }

        Text(
            "اطلاعات این صفحه باید با قابلیت‌های واقعی برنامه همگام بماند؛ مقدار یا قابلیت ساختگی در آن نمایش داده نمی‌شود.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(horizontal = 4.dp),
        )
    }
}

@Composable
private fun SectionCard(title: String, content: @Composable ColumnScope.() -> Unit) {
    Surface(
        modifier = Modifier
            .fillMaxWidth()
            .semantics { contentDescription = title },
        shape = RoundedCornerShape(18.dp),
        color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.45f),
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
            content = {
                Text(title, style = MaterialTheme.typography.titleMedium)
                content()
            },
        )
    }
}

@Composable
private fun InfoRow(label: String, value: String) {
    Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
        Text(label, style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text(value, style = MaterialTheme.typography.bodyMedium)
    }
}

@Composable
private fun Feature(text: String) {
    Text("• $text", style = MaterialTheme.typography.bodyMedium)
}

@Composable
private fun SimplePage(title: String, content: @Composable ColumnScope.() -> Unit) {
    LazyColumn(
        Modifier.fillMaxSize().padding(20.dp),
        contentPadding = PaddingValues(bottom = 32.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        item { Text(title, style = MaterialTheme.typography.headlineMedium) }
        item { Column(verticalArrangement = Arrangement.spacedBy(12.dp), content = content) }
    }
}
