package com.samanramezani.aichattest.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

/** Central, lightweight UI contracts shared by documented screens. */
object UiTokens {
    // Layout
    val pagePadding = 20.dp
    val compactPadding = 12.dp
    val sectionGap = 12.dp
    val itemGap = 8.dp
    val minimumTouchTarget = 48.dp
    val contentMaxWidth = 760.dp
    val sidebarMaxWidth = 520.dp

    // Shape
    val surfaceRadius = 20.dp
    val controlRadius = 16.dp
    val headerRadius = 22.dp
    val dialogRadius = 28.dp
    val composerRadius = 24.dp

    // Vertical rhythm
    val headerMinHeight = 68.dp
    val compactControlHeight = 48.dp

    // Elevation is intentionally shallow: the UI contract forbids heavy depth.
    val surfaceElevation = 1.dp
    val overlayElevation = 6.dp
}

@Composable
fun UiSection(
    title: String,
    modifier: Modifier = Modifier,
    content: @Composable () -> Unit,
) {
    Column(
        modifier = modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(UiTokens.itemGap),
    ) {
        Text(title, style = MaterialTheme.typography.titleLarge)
        content()
    }
}

@Composable
fun UiSurface(
    modifier: Modifier = Modifier,
    content: @Composable () -> Unit,
) {
    Surface(
        modifier = modifier.fillMaxWidth(),
        shape = RoundedCornerShape(UiTokens.surfaceRadius),
        color = MaterialTheme.colorScheme.surface,
        tonalElevation = UiTokens.surfaceElevation,
    ) {
        content()
    }
}

@Composable
fun UiStatusRow(
    label: String,
    value: String,
    modifier: Modifier = Modifier,
) {
    Row(
        modifier = modifier
            .fillMaxWidth()
            .heightIn(min = UiTokens.minimumTouchTarget)
            .padding(horizontal = UiTokens.compactPadding),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween,
    ) {
        Text(label, style = MaterialTheme.typography.bodyMedium)
        Text(
            value,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}
