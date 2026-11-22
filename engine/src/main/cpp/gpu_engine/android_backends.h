#pragma once

#include "gpu_engine.h"

namespace ai_chat::gpu_engine {

class OperatorRegistry;
class QuantizationRegistry;
class ArchitectureRegistry;
class TokenizerRegistry;
class ModelLoweringRegistry;

// Registers Dawn first (primary) and Universal GLES second (fallback backend).
// Call once before engine initialization; this only registers backends and does
// not claim model/operator readiness.
Status register_android_gpu_backends(GpuEngine& engine);
Status initialize_android_gpu_engine(GpuEngine& engine);
Status register_android_gpu_runtime_components(OperatorRegistry& operators,
                                               QuantizationRegistry& quantizations,
                                               ArchitectureRegistry& architectures,
                                               TokenizerRegistry& tokenizers,
                                               ModelLoweringRegistry& lowerings);
Status initialize_android_gpu_runtime(GpuEngine& engine, OperatorRegistry& operators,
                                      QuantizationRegistry& quantizations,
                                      ArchitectureRegistry& architectures,
                                      TokenizerRegistry& tokenizers,
                                      ModelLoweringRegistry& lowerings);

} // namespace ai_chat::gpu_engine
