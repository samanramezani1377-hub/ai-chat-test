package com.woogit.aicore.runtime

import com.woogit.aicore.domain.ModelDescriptor
import com.woogit.aicore.domain.ModelError
import com.woogit.aicore.domain.ModelFormat
import com.woogit.aicore.domain.ModelImporter
import com.woogit.aicore.domain.ModelResult
import com.woogit.aicore.domain.ModelState
import com.woogit.aicore.domain.RuntimeCompatibility
import com.woogit.aicore.domain.ValidationStatus
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption
import java.util.UUID

/** Imports a user-selected model into managed storage using the real file and GGUF inspector. */
class FileModelImporter(
    private val storageDirectory: Path,
    private val inspector: GgufInspector = GgufInspector()
) : ModelImporter {
    override suspend fun import(source: Path): ModelResult<ModelDescriptor> {
        if (!Files.isRegularFile(source) || !Files.isReadable(source)) {
            return ModelResult.Failure(ModelError.FileAccess("Selected model file cannot be read"))
        }
        return try {
            Files.createDirectories(storageDirectory)
            val inspectionResult = inspector.inspect(source)
            val inspection = when (inspectionResult) {
                is ModelResult.Success -> inspectionResult.value
                is ModelResult.Failure -> return inspectionResult
            }
            if (inspection.format != ModelFormat.GGUF) {
                return ModelResult.Failure(ModelError.UnsupportedFormat("Only GGUF models are supported"))
            }

            val id = UUID.randomUUID().toString()
            val target = storageDirectory.resolve("$id.gguf")
            Files.copy(source, target, StandardCopyOption.COPY_ATTRIBUTES)
            try {
                val storedSize = Files.size(target)
                if (storedSize != inspection.sizeBytes) {
                    return ModelResult.Failure(ModelError.Storage("Imported model size changed during copy"))
                }
                ModelResult.Success(
                    ModelDescriptor(
                        id = id,
                        displayName = inspection.metadata.name ?: source.fileName.toString(),
                        path = target,
                        format = inspection.format,
                        quantization = inspection.quantization,
                        sizeBytes = storedSize,
                        metadata = inspection.metadata,
                        validation = ValidationStatus.VALID,
                        runtimeCompatibility = RuntimeCompatibility(
                            supported = false,
                            reason = "Runtime compatibility is established when a concrete runtime is configured"
                        ),
                        state = ModelState.READY
                    )
                )
            } catch (t: Throwable) {
                Files.deleteIfExists(target)
                throw t
            }
        } catch (t: Throwable) {
            ModelResult.Failure(ModelError.Storage("Unable to import model", t))
        }
    }
}
