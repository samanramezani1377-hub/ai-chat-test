#pragma once

#include "model_graph.h"

namespace ai_chat::gpu_engine {

// Registers structural lowerings for F32 and packed quantized-weight kernels.
// A lowering does not promote a kernel: OperatorRegistry still requires a
// backend-specific numerical validation before graph dispatch is authorized.
Status register_builtin_model_lowerings(BackendKind backend,
                                        ModelLoweringRegistry& registry);

} // namespace ai_chat::gpu_engine
