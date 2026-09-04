package com.samanramezani.aichattest

import android.content.ContentResolver
import android.net.Uri
import com.woogit.aicore.domain.ApiProviderConfigStore
import com.woogit.aicore.domain.ChatMessage
import com.woogit.aicore.domain.GenerationRequest
import com.woogit.aicore.domain.GenerationResult
import com.woogit.aicore.domain.ModelDescriptor
import com.woogit.aicore.domain.ModelError
import com.woogit.aicore.domain.ModelFormat
import com.woogit.aicore.domain.ModelMetadata
import com.woogit.aicore.domain.ModelResult
import com.woogit.aicore.domain.ModelState
import com.woogit.aicore.domain.Quantization
import com.woogit.aicore.domain.RuntimeCompatibility
import com.woogit.aicore.domain.ValidationStatus
import com.woogit.aicore.runtime.RemoteApiRuntimeAdapter
import com.woogit.aicore.runtime.RuntimeAdapter
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.file.Path
import java.nio.file.Paths

/** Compatibility façade kept for the existing UI while inference is now 100% remote/API based. */
class AndroidModelManager(
    @Suppress("UNUSED_PARAMETER") private val contentResolver: ContentResolver,
    @Suppress("UNUSED_PARAMETER") private val modelDirectory: Path,
    runtime: RuntimeAdapter,
) {
    private val remoteRuntime = runtime
    @Volatile private var active = true

    private fun descriptor(): ModelDescriptor {
        val config = ApiProviderConfigStore.current
        return ModelDescriptor(
            id = "remote:${config.providerId}:${config.model}",
            displayName = "${config.providerId} · ${config.model}",
            path = Paths.get(System.getProperty("java.io.tmpdir") ?: ".").resolve("remote-api-model"),
            format = ModelFormat.UNKNOWN,
            quantization = Quantization.UNKNOWN,
            sizeBytes = 0L,
            metadata = ModelMetadata(name = config.model, contextLength = null),
            validation = ValidationStatus.VALID,
            runtimeCompatibility = RuntimeCompatibility(true, "Remote API runtime"),
            state = if (active) ModelState.ACTIVE else ModelState.READY,
        )
    }

    suspend fun import(@Suppress("UNUSED_PARAMETER") uri: Uri): ModelResult<ModelDescriptor> =
        ModelResult.Failure(ModelError.UnsupportedFormat("Local GGUF models are disabled; configure a remote API provider instead."))

    suspend fun models(): ModelResult<List<ModelDescriptor>> = ModelResult.Success(if (active) listOf(descriptor()) else emptyList())
    suspend fun activeModel(): ModelResult<ModelDescriptor?> = ModelResult.Success(if (active) descriptor() else null)
    suspend fun restoreActive(): ModelResult<ModelDescriptor?> = ModelResult.Success(if (active) descriptor() else null)

    suspend fun activate(@Suppress("UNUSED_PARAMETER") id: String): ModelResult<ModelDescriptor> = withContext(Dispatchers.IO) {
        try {
            remoteRuntime.load(descriptor())
            active = true
            ModelResult.Success(descriptor())
        } catch (t: Throwable) {
            ModelResult.Failure(ModelError.LoadFailed("Remote API is not ready: ${t.message}", t))
        }
    }

    suspend fun deactivate(): ModelResult<Unit> {
        remoteRuntime.unload()
        active = false
        return ModelResult.Success(Unit)
    }

    suspend fun unload(): ModelResult<Unit> = deactivate()

    suspend fun generate(messages: List<ChatMessage>, settings: com.woogit.aicore.domain.InferenceSettings, onToken: suspend (String) -> Unit = {}): ModelResult<GenerationResult> =
        try { ModelResult.Success(remoteRuntime.generate(GenerationRequest(messages, settings), onToken)) }
        catch (t: Throwable) { ModelResult.Failure(ModelError.Inference("Remote API generation failed: ${t.message}", t)) }

    suspend fun stopGeneration() = remoteRuntime.stopGeneration()
    suspend fun delete(@Suppress("UNUSED_PARAMETER") id: String): ModelResult<Unit> = ModelResult.Success(Unit)
}
