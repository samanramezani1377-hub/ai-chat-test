package com.samanramezani.aichattest.ui.chat

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

@Composable
internal fun AssistantMarkdown(text: String) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        val fence = String(charArrayOf(96, 96, 96))
        val lines = text.replace("\r\n", "\n").split('\n')
        var index = 0
        while (index < lines.size) {
            val line = lines[index]
            when {
                line.startsWith(fence) -> {
                    val language = line.removePrefix(fence).trim()
                    val code = buildString {
                        index++
                        while (index < lines.size && !lines[index].startsWith(fence)) {
                            append(lines[index])
                            if (index < lines.lastIndex) append('\n')
                            index++
                        }
                    }.trimEnd()
                    CodeBlock(code, language)
                }
                line.trim().isEmpty() -> Spacer(Modifier.height(2.dp))
                line.matches(Regex("^#{1,6}\\s+.*")) -> Text(line.substringAfter('#').trimStart(), style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.SemiBold)
                line.matches(Regex("^[-*+]\\s+.*")) -> Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("•", fontWeight = FontWeight.Bold)
                    InlineMarkdown(line.drop(1).trimStart(), Modifier.weight(1f))
                }
                line.matches(Regex("^\\d+[.)]\\s+.*")) -> {
                    val number = line.takeWhile { it.isDigit() }
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text(number + ".", fontWeight = FontWeight.SemiBold)
                        InlineMarkdown(line.dropWhile { it.isDigit() }.drop(1).trimStart(), Modifier.weight(1f))
                    }
                }
                line.startsWith("> ") -> QuoteBlock(line.removePrefix("> "))
                else -> InlineMarkdown(line)
            }
            index++
        }
    }
}

@Composable
private fun InlineMarkdown(text: String, modifier: Modifier = Modifier) {
    Text(parseInlineMarkdown(text), modifier = modifier, style = MaterialTheme.typography.bodyLarge)
}

private fun parseInlineMarkdown(text: String): AnnotatedString = buildAnnotatedString {
    var i = 0
    while (i < text.length) {
        when {
            text.startsWith("**", i) || text.startsWith("__", i) -> {
                val marker = text.substring(i, i + 2)
                val end = text.indexOf(marker, i + 2)
                if (end > i + 2) {
                    withStyle(SpanStyle(fontWeight = FontWeight.Bold)) { append(text.substring(i + 2, end)) }
                    i = end + 2
                } else { append(text[i]); i++ }
            }
            text[i].code == 96 -> {
                val end = text.indexOfFirst { it.code == 96 && text.indexOf(it) >= i + 1 }
                if (end > i + 1) { append(text.substring(i + 1, end)); i = end + 1 }
                else { append(text[i]); i++ }
            }
            text.startsWith("~~", i) -> {
                val end = text.indexOf("~~", i + 2)
                if (end > i + 2) {
                    withStyle(SpanStyle(textDecoration = TextDecoration.LineThrough)) { append(text.substring(i + 2, end)) }
                    i = end + 2
                } else { append(text[i]); i++ }
            }
            text[i] == '*' || text[i] == '_' -> {
                val marker = text[i]
                val end = text.indexOf(marker, i + 1)
                if (end > i + 1) {
                    withStyle(SpanStyle(fontStyle = FontStyle.Italic)) { append(text.substring(i + 1, end)) }
                    i = end + 1
                } else { append(text[i]); i++ }
            }
            else -> { append(text[i]); i++ }
        }
    }
}

@Composable
private fun CodeBlock(code: String, language: String) {
    Column(Modifier.fillMaxWidth().background(MaterialTheme.colorScheme.surfaceVariant, RoundedCornerShape(14.dp)).padding(12.dp)) {
        if (language.isNotEmpty()) Text(language, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.primary)
        Text(code, style = MaterialTheme.typography.bodyMedium.copy(fontFamily = FontFamily.Monospace, fontSize = 13.sp))
    }
}

@Composable
private fun QuoteBlock(text: String) {
    Row(Modifier.fillMaxWidth().background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = .55f), RoundedCornerShape(10.dp)).padding(10.dp)) {
        Box(Modifier.width(3.dp).heightIn(min = 22.dp).background(MaterialTheme.colorScheme.primary, RoundedCornerShape(2.dp)))
        Spacer(Modifier.width(9.dp))
        InlineMarkdown(text)
    }
}
