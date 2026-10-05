package com.samanramezani.aichattest.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

object UiTokens {
    val pagePadding = 18.dp
    val compactPadding = 14.dp
    val sectionGap = 14.dp
    val itemGap = 10.dp
    val minimumTouchTarget = 48.dp
    val contentMaxWidth = 820.dp
    val sidebarMaxWidth = 560.dp
    val surfaceRadius = 24.dp
    val controlRadius = 18.dp
    val headerRadius = 24.dp
    val dialogRadius = 28.dp
    val composerRadius = 28.dp
    val headerMinHeight = 64.dp
    val compactControlHeight = 48.dp
    val surfaceElevation = 0.dp
    val overlayElevation = 8.dp
}

@Composable
fun UiSection(title: String, modifier: Modifier = Modifier, content: @Composable ColumnScope.() -> Unit) {
    Column(modifier = modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(UiTokens.itemGap)) {
        Text(title, style = MaterialTheme.typography.titleLarge)
        content()
    }
}

@Composable
fun UiSurface(modifier: Modifier = Modifier, content: @Composable ColumnScope.() -> Unit) {
    Surface(
        modifier = modifier.fillMaxWidth(),
        shape = RoundedCornerShape(UiTokens.surfaceRadius),
        color = MaterialTheme.colorScheme.surface,
        tonalElevation = UiTokens.surfaceElevation,
    ) { Column(content = content) }
}

@Composable
fun UiStatusRow(label: String, value: String, modifier: Modifier = Modifier) {
    Row(
        modifier = modifier.fillMaxWidth().heightIn(min = UiTokens.minimumTouchTarget).padding(horizontal = UiTokens.compactPadding),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween,
    ) {
        Text(label, style = MaterialTheme.typography.bodyMedium)
        Text(value, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}
