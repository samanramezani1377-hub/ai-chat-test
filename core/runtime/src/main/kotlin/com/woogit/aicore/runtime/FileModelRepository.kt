package com.woogit.aicore.runtime

import com.woogit.aicore.domain.ModelDescriptor
import com.woogit.aicore.domain.ModelError
import com.woogit.aicore.domain.ModelRepository
import com.woogit.aicore.domain.ModelResult
import java.nio.file.Files
import java.nio.file.Path
import java.util.Properties

/** Small durable registry. Model files remain outside the registry; only metadata is persisted. */
class FileModelRepository(
    private val registryFile: Path
) : ModelRepository {
    private val lock = Any()

    override suspend fun register(model: ModelDescriptor): ModelResult<Unit> = synchronized(lock) {
        runCatching {
            Files.createDirectories(registryFile.parent ?: registryFile.toAbsolutePath().parent)
            val properties = load()
            val prefix = "model.${model.id}."
            properties.setProperty(prefix + "displayName", model.displayName)
            properties.setProperty(prefix + "path", model.path.toString())
            properties.setProperty(prefix + "format", model.format.name)
            properties.setProperty(prefix + "quantization", model.quantization.name)
            properties.setProperty(prefix + "sizeBytes", model.sizeBytes.toString())
            properties.setProperty(prefix + "architecture", model.metadata.architecture.orEmpty())
            properties.setProperty(prefix + "name", model.metadata.name.orEmpty())
            properties.setProperty(prefix + "contextLength", model.metadata.contextLength?.toString().orEmpty())
            properties.setProperty(prefix + "embeddingLength", model.metadata.embeddingLength?.toString().orEmpty())
            properties.setProperty(prefix + "blockCount", model.metadata.blockCount?.toString().orEmpty())
            properties.setProperty("activeModelId", properties.getProperty("activeModelId", ""))
            save(properties)
            ModelResult.Success(Unit)
        }.getOrElse { ModelResult.Failure(ModelError.Storage("Unable to persist model registry", it)) }
    }

    override suspend fun get(id: String): ModelResult<ModelDescriptor?> = synchronized(lock) {
        runCatching { loadModel(load(), id) }
            .fold({ ModelResult.Success(it) }, { ModelResult.Failure(ModelError.Storage("Unable to read model registry", it)) })
    }

    override suspend fun list(): ModelResult<List<ModelDescriptor>> = synchronized(lock) {
        runCatching {
            val properties = load()
            properties.stringPropertyNames()
                .asSequence()
                .filter { it.startsWith("model.") && it.endsWith(".displayName") }
                .map { it.removePrefix("model.").removeSuffix(".displayName") }
                .mapNotNull { loadModel(properties, it) }
                .toList()
        }.fold({ ModelResult.Success(it) }, { ModelResult.Failure(ModelError.Storage("Unable to list model registry", it)) })
    }

    override suspend fun getActive(): ModelResult<ModelDescriptor?> = synchronized(lock) {
        runCatching {
            loadModel(load(), load().getProperty("activeModelId").orEmpty())
        }.fold({ ModelResult.Success(it) }, { ModelResult.Failure(ModelError.Storage("Unable to read active model", it)) })
    }

    override suspend fun setActive(id: String?): ModelResult<Unit> = synchronized(lock) {
        runCatching {
            val properties = load()
            if (!id.isNullOrBlank() && loadModel(properties, id) == null) {
                return@synchronized ModelResult.Failure(ModelError.InvalidModel("Cannot activate an unregistered model"))
            }
            properties.setProperty("activeModelId", id.orEmpty())
            save(properties)
            ModelResult.Success(Unit)
        }.getOrElse { ModelResult.Failure(ModelError.Storage("Unable to persist active model", it)) }
    }

    override suspend fun unregister(id: String): ModelResult<Unit> = synchronized(lock) {
        runCatching {
            val properties = load()
            val prefix = "model.$id."
            properties.stringPropertyNames().filter { it.startsWith(prefix) }.forEach(properties::remove)
            if (properties.getProperty("activeModelId") == id) properties.setProperty("activeModelId", "")
            save(properties)
            ModelResult.Success(Unit)
        }.getOrElse { ModelResult.Failure(ModelError.Storage("Unable to remove model from registry", it)) }
    }

    private fun load(): Properties = Properties().also { properties ->
        if (Files.isRegularFile(registryFile)) Files.newInputStream(registryFile).use(properties::load)
    }

    private fun save(properties: Properties) {
        Files.newOutputStream(registryFile).use { properties.store(it, "AI Chat local model registry") }
    }

    private fun loadModel(properties: Properties, id: String): ModelDescriptor? {
        if (id.isBlank() || !properties.containsKey("model.$id.displayName")) return null
        val prefix = "model.$id."
        return ModelDescriptor(
            id = id,
            displayName = properties.getProperty(prefix + "displayName"),
            path = Path.of(properties.getProperty(prefix + "path")),
            format = enumValue(properties.getProperty(prefix + "format"), com.woogit.aicore.domain.ModelFormat.UNKNOWN),
            quantization = enumValue(properties.getProperty(prefix + "quantization"), com.woogit.aicore.domain.Quantization.UNKNOWN),
            sizeBytes = properties.getProperty(prefix + "sizeBytes", "0").toLong(),
            metadata = com.woogit.aicore.domain.ModelMetadata(
                architecture = properties.getProperty(prefix + "architecture").orEmpty().ifBlank { null },
                name = properties.getProperty(prefix + "name").orEmpty().ifBlank { null },
                contextLength = properties.getProperty(prefix + "contextLength").toLongOrNull(),
                embeddingLength = properties.getProperty(prefix + "embeddingLength").toLongOrNull(),
                blockCount = properties.getProperty(prefix + "blockCount").toLongOrNull()
            ),
            validation = com.woogit.aicore.domain.ValidationStatus.VALID,
            runtimeCompatibility = com.woogit.aicore.domain.RuntimeCompatibility(false, "Runtime compatibility must be established by the configured runtime"),
            state = if (properties.getProperty("activeModelId") == id) com.woogit.aicore.domain.ModelState.ACTIVE else com.woogit.aicore.domain.ModelState.READY
        )
    }

    private inline fun <reified T : Enum<T>> enumValue(value: String?, fallback: T): T =
        value?.let { runCatching { enumValueOf<T>(it) }.getOrNull() } ?: fallback
}
