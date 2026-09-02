package com.samanramezani.aichattest

import androidx.compose.runtime.Composable
import com.woogit.aicore.domain.ModelDescriptor

internal fun Boolean?.orFalse(): Boolean = this == true

/** Compatibility bridge for the pre-refactor MainScreen call shape. */
@Composable
internal fun ApprovalDialog(
    execution: ExecutionState?,
    onApprove: () -> Unit,
    onReject: () -> Unit,
) {
    if (execution?.approvalRequired != true) return
    androidx.compose.material3.AlertDialog(
        onDismissRequest = {},
        title = { androidx.compose.material3.Text("تأیید عملیات") },
        text = {
            androidx.compose.foundation.layout.Column {
                androidx.compose.material3.Text(execution.action)
                androidx.compose.material3.Text("این عملیات قبل از اجرا به تأیید شما نیاز دارد.")
            }
        },
        confirmButton = {
            androidx.compose.material3.TextButton(onClick = onApprove) { androidx.compose.material3.Text("تأیید و اجرا") }
        },
        dismissButton = {
            androidx.compose.material3.TextButton(onClick = onReject) { androidx.compose.material3.Text("رد") }
        },
    )
}
