#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ai_chat::gpu_engine {

struct GpuMemoryPlanInput {
    uint64_t total_weight_bytes = 0;
    uint64_t streaming_chunk_bytes = 0;
    uint64_t kv_cache_bytes = 0;
    uint64_t scratch_bytes = 0;
    uint64_t safety_reserve_bytes = 0;
    uint64_t device_budget_bytes = 0;
    bool stream_weights = true;
};

struct GpuMemoryPlanReport {
    uint64_t resident_weight_bytes = 0;
    uint64_t kv_cache_bytes = 0;
    uint64_t scratch_bytes = 0;
    uint64_t required_bytes = 0;
    uint64_t usable_budget_bytes = 0;
    bool fits = false;
    std::vector<std::string> blockers;
};

// Uses an explicit application-provided GPU memory budget. It never guesses
// total physical VRAM from WebGPU, which does not expose that value.
GpuMemoryPlanReport plan_gpu_memory(const GpuMemoryPlanInput& input);

} // namespace ai_chat::gpu_engine
