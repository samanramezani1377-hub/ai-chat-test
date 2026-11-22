#pragma once

#include "gpu_engine.h"

#include <string>
#include <vector>

namespace ai_chat::gpu_engine {

enum class RuntimeSelection : uint8_t {
    CpuLlamaCpp = 0,
    DedicatedGpuEngine,
    Unavailable,
};

struct RuntimeReadiness {
    bool cpu_fallback_available = true;
    bool gpu_device_ready = false;
    bool tokenizer_validated = false;
    bool architecture_validated = false;
    bool quantization_validated = false;
    bool operator_coverage_validated = false;
    bool model_graph_validated = false;
    bool memory_preflight_passed = false;
};

struct RuntimeSelectionReport {
    RuntimeSelection selected = RuntimeSelection::CpuLlamaCpp;
    bool gpu_selected = false;
    bool cpu_fallback_available = true;
    std::vector<std::string> blockers;
};

// Pure fail-closed policy. It does not load a model or modify the existing CPU runtime.
RuntimeSelectionReport select_runtime(const RuntimeReadiness& readiness);

} // namespace ai_chat::gpu_engine
