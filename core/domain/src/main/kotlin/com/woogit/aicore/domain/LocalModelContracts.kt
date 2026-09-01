package com.woogit.aicore.domain

import java.nio.file.Path

interface ModelImporter {
    suspend fun import(source: Path): ModelResult<ModelDescriptor>
}

interface ModelInspector {
    suspend fun inspect(path: Path): ModelResult<ModelInspection>
}

interface ModelValidator {
    suspend fun validate(inspection: ModelInspection): ModelResult<ValidationStatus>
}

interface ModelRepository {
    suspend fun register(model: ModelDescriptor): ModelResult<Unit>
    suspend fun get(id: String): ModelResult<ModelDescriptor?>
    suspend fun list(): ModelResult<List<ModelDescriptor>>
    suspend fun getActive(): ModelResult<ModelDescriptor?>
    suspend fun setActive(id: String?): ModelResult<Unit>
    suspend fun unregister(id: String): ModelResult<Unit>
}

interface ModelLifecycleManager {
    suspend fun activate(id: String): ModelResult<ModelDescriptor>
    suspend fun deactivate(): ModelResult<Unit>
    suspend fun unload(): ModelResult<Unit>
}
