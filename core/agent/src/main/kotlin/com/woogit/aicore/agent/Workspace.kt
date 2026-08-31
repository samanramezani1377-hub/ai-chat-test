package com.woogit.aicore.agent

import java.time.Instant

/** Persistent workbench state; it is data owned by Core, not UI state. */
data class WorkspaceState(
    val id: String,
    val goal: String? = null,
    val activeTaskId: String? = null,
    val status: Status = Status.IDLE,
    val updatedAt: Instant = Instant.now()
) {
    enum class Status { IDLE, WORKING, WAITING_APPROVAL, FAILED, COMPLETED }
}

interface WorkspaceStore {
    suspend fun get(id: String): WorkspaceState?
    suspend fun save(state: WorkspaceState)
}

class InMemoryWorkspaceStore : WorkspaceStore {
    private val values = mutableMapOf<String, WorkspaceState>()

    @Synchronized
    override suspend fun get(id: String): WorkspaceState? = values[id]

    @Synchronized
    override suspend fun save(state: WorkspaceState) { values[state.id] = state }
}
