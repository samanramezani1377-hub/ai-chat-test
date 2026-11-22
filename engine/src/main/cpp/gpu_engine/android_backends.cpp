#include "android_backends.h"
#include "execution_engine.h"
#include "model_registry.h"
#include "model_lowerings.h"
#include "operators.h"
#include "quantization.h"
#include "tokenizer.h"

#include <utility>
#include <vector>
#include "model_lowerings.h"

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
                                               TokenizerRegistry& tokenizers,
                                               ModelLoweringRegistry& lowerings) {
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
    status = register_builtin_tokenizer_adapters(tokenizers);
    if (!status.ok()) return status;
    status = register_builtin_model_lowerings(BackendKind::DawnWebGpu, lowerings);
    if (!status.ok()) return status;
    return register_builtin_model_lowerings(BackendKind::OpenGles, lowerings);
}

Status initialize_android_gpu_runtime(GpuEngine& engine, OperatorRegistry& operators,
                                      QuantizationRegistry& quantizations,
                                      ArchitectureRegistry& architectures,
                                      TokenizerRegistry& tokenizers,
                                      ModelLoweringRegistry& lowerings) {
    Status status = register_android_gpu_backends(engine);
    if (!status.ok()) return status;
    status = register_android_gpu_runtime_components(operators, quantizations, architectures, tokenizers, lowerings);
    if (!status.ok()) return status;

    // Backend selection is not complete at device discovery. A backend must
    // compile, dispatch and numerically validate every baseline F32 operator
    // before it can be selected for the current dense-model execution path.
    // This lets Universal GLES take over when Dawn exists but lacks a working
    // operator implementation on the actual device.
    std::vector<std::string> failures;
    for (BackendKind kind : {BackendKind::DawnWebGpu, BackendKind::OpenGles}) {
        status = engine.initialize(kind);
        if (!status.ok()) {
            failures.emplace_back((kind == BackendKind::DawnWebGpu ? "DawnWebGPU: " : "UniversalGLES: ") +
                                  status.message);
            continue;
        }

        KernelValidationReport validation;
        status = validate_builtin_operator_kernels(*engine.active_backend(), operators, validation);
        if (status.ok() && validation.complete) return Status::success();

        std::string reason = status.ok()
            ? "baseline operator validation incomplete"
            : status.message;
        for (const auto& item : validation.unavailable) reason += " | unavailable=" + item;
        for (const auto& item : validation.rejected) reason += " | rejected=" + item;
        failures.emplace_back((kind == BackendKind::DawnWebGpu ? "DawnWebGPU: " : "UniversalGLES: ") +
                              reason);
        engine.shutdown();
    }

    std::string message = "No GPU backend passed baseline operator validation";
    for (const auto& failure : failures) message += " || " + failure;
    return {StatusCode::Unsupported, std::move(message)};
}

} // namespace ai_chat::gpu_engine
