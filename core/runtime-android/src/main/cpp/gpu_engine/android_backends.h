#pragma once

#include "gpu_engine.h"

namespace ai_chat::gpu_engine {

// Registers Dawn first (primary) and Universal GLES second (fallback backend).
// Call once before GpuEngine::initialize; this only registers backends and does
// not claim model/operator readiness.
Status register_android_gpu_backends(GpuEngine& engine);
Status initialize_android_gpu_engine(GpuEngine& engine);

} // namespace ai_chat::gpu_engine
