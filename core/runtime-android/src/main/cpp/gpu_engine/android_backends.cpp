#include "android_backends.h"
#include "execution_engine.h"
#include "model_registry.h"
#include "operators.h"
#include "quantization.h"
#include "tokenizer.h"

namespace ai_chat::gpu_engine {

Status register_android_gpu_backends(GpuEngine& engine) {
    Status status = engine.register_backend(create_dawn_backend());
    if (!status.ok()) return status;
    return engine.register_backend(create_universal_gles_backend());
}

Status initialize_android_gpu_engine(GpuEngine& engine) {
    Status status = register_android_gpu_backends(engine);
    if (!status.ok()) return status;
    return engine.initialize_with_fallback({BackendKind::DawnWebGpu, BackendKind::OpenGles});
}



Status register_android_gpu_runtime_components(OperatorRegistry& operators,
                                               QuantizationRegistry& quantizations,
                                               ArchitectureRegistry& architectures,
                                               TokenizerRegistry& tokenizers) {
    Status status = register_builtin_operator_candidates(operators);
    if (!status.ok()) return status;
    status = register_builtin_gles_operator_candidates(operators);
    if (!status.ok()) return status;
    status = register_builtin_quantized_operator_candidates(operators);
    if (!status.ok()) return status;
    status = register_builtin_quantization_formats(quantizations);
    if (!status.ok()) return status;
    status = register_builtin_architecture_adapters(architectures);
    if (!status.ok()) return status;
    return register_builtin_tokenizer_adapters(tokenizers);
}

Status initialize_android_gpu_runtime(GpuEngine& engine, OperatorRegistry& operators,
                                      QuantizationRegistry& quantizations,
                                      ArchitectureRegistry& architectures,
                                      TokenizerRegistry& tokenizers) {
    Status status = register_android_gpu_backends(engine);
    if (!status.ok()) return status;
    status = register_android_gpu_runtime_components(operators, quantizations, architectures, tokenizers);
    if (!status.ok()) return status;
    return engine.initialize_with_fallback({BackendKind::DawnWebGpu, BackendKind::OpenGles});
}

} // namespace ai_chat::gpu_engine
