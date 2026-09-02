package com.samanramezani.aichattest

import androidx.compose.foundation.layout.Column
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import com.samanramezani.aichattest.ui.state.ExecutionState

internal fun Boolean?.orFalse(): Boolean = this == true

@Composable
internal fun ApprovalDialog(
    execution: ExecutionState?,
    onApprove: () -> Unit,
    onReject: () -> Unit,
) {
    if (execution?.approvalRequired != true) return
    AlertDialog(
        onDismissRequest = {},
        title = { Text("تأیید عملیات") },
        text = {
            Column {
                Text(execution.action)
                Text("این عملیات قبل از اجرا به تأیید شما نیاز دارد.")
            }
        },
        confirmButton = { TextButton(onClick = onApprove) { Text("تأیید و اجرا") } },
        dismissButton = { TextButton(onClick = onReject) { Text("رد") } },
    )
}
