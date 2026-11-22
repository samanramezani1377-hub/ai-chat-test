#pragma once

#include "gpu_engine.h"
#include "../universal_gpu_adapter.h"

namespace ai_chat::gpu_engine {

// Converts only facts proven by the existing Universal GPU Adapter.
// This is a diagnostics/capability bridge, not an inference backend.
Status probe_universal_adapter(DeviceInfo& out);

DeviceInfo device_info_from_universal_adapter(
    const ai_chat::gpu::UniversalGpuCapabilities& capabilities);

} // namespace ai_chat::gpu_engine
