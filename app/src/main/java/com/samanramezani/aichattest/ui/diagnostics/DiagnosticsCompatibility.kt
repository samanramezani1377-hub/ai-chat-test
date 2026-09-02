package com.samanramezani.aichattest.ui.diagnostics

import androidx.compose.runtime.Composable
import com.samanramezani.aichattest.AppContainer
import com.samanramezani.aichattest.ui.state.ExecutionState
import com.woogit.aicore.domain.ModelDescriptor

/** Compatibility overload for the pre-refactor MainScreen signature. */
@Composable
internal fun DiagnosticsPage(
    container: AppContainer?,
    execution: ExecutionState?,
    activeModel: ModelDescriptor?,
) {
    DiagnosticsPage(error = null, execution = execution)
}
