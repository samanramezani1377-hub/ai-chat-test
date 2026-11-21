#include "memory_planner.h"

#include <limits>

namespace ai_chat::gpu_engine {
namespace {

bool checked_add(uint64_t a, uint64_t b, uint64_t& out) {
    if (b > std::numeric_limits<uint64_t>::max() - a) return false;
    out = a + b;
    return true;
}

} // namespace

GpuMemoryPlanReport plan_gpu_memory(const GpuMemoryPlanInput& input) {
    GpuMemoryPlanReport report;
    report.kv_cache_bytes = input.kv_cache_bytes;
    report.scratch_bytes = input.scratch_bytes;

    if (input.device_budget_bytes == 0) {
        report.blockers.emplace_back("GPU_MEMORY_BUDGET_UNKNOWN");
        return report;
    }
    if (input.stream_weights && input.total_weight_bytes != 0 &&
        (input.streaming_chunk_bytes == 0 || input.streaming_chunk_bytes > input.total_weight_bytes)) {
        report.blockers.emplace_back("INVALID_WEIGHT_STREAMING_CHUNK");
        return report;
    }
    report.resident_weight_bytes = input.stream_weights
        ? input.streaming_chunk_bytes : input.total_weight_bytes;

    if (input.safety_reserve_bytes >= input.device_budget_bytes) {
        report.blockers.emplace_back("GPU_MEMORY_RESERVE_EXCEEDS_BUDGET");
        return report;
    }
    report.usable_budget_bytes = input.device_budget_bytes - input.safety_reserve_bytes;

    uint64_t required = 0;
    if (!checked_add(report.resident_weight_bytes, report.kv_cache_bytes, required) ||
        !checked_add(required, report.scratch_bytes, required)) {
        report.blockers.emplace_back("GPU_MEMORY_ESTIMATE_OVERFLOW");
        return report;
    }
    report.required_bytes = required;
    if (required > report.usable_budget_bytes) {
        report.blockers.emplace_back("GPU_MEMORY_BUDGET_INSUFFICIENT");
        return report;
    }
    report.fits = true;
    return report;
}

} // namespace ai_chat::gpu_engine
