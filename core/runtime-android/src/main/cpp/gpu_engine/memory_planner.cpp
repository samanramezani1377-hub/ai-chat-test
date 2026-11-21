#include "memory_planner.h"

#include <limits>
#include <algorithm>
#include <limits>

namespace ai_chat::gpu_engine {
namespace {

bool checked_add(uint64_t a, uint64_t b, uint64_t& out) {
    if (b > std::numeric_limits<uint64_t>::max() - a) return false;
    out = a + b;
    return true;
}

} // namespace


namespace {
bool checked_multiply(uint64_t a, uint64_t b, uint64_t& out) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) return false;
    out = a * b;
    return true;
}
}

ModelGpuMemoryEstimateReport estimate_model_gpu_memory(
        const ModelExecutionPlan& model, const ModelGpuMemoryEstimateInput& input) {
    ModelGpuMemoryEstimateReport report;
    if (!model.structurally_validated || model.layer_count == 0 ||
        model.embedding_length == 0 || model.attention_heads == 0 ||
        model.kv_heads == 0 || model.head_dimension == 0 ||
        model.feed_forward_length == 0 || input.token_count == 0 ||
        input.context_length < input.token_count || input.bytes_per_element == 0) {
        report.blockers.emplace_back("MODEL_MEMORY_ESTIMATE_INPUT_INVALID");
        return report;
    }
    uint64_t kv = 1;
    const uint64_t kv_factors[] = {2, model.layer_count, input.context_length,
        model.kv_heads, model.head_dimension, input.bytes_per_element};
    for (uint64_t factor : kv_factors) {
        if (!checked_multiply(kv, factor, kv)) {
            report.blockers.emplace_back("MODEL_KV_CACHE_SIZE_OVERFLOW");
            return report;
        }
    }
    report.estimated_kv_cache_bytes = kv;
    uint64_t hidden = 0, ffn = 0, attention_scores = 0, scratch = 0;
    if (!checked_multiply(input.token_count, model.embedding_length, hidden) ||
        !checked_multiply(hidden, input.bytes_per_element, hidden) ||
        !checked_multiply(input.token_count, model.feed_forward_length, ffn) ||
        !checked_multiply(ffn, 3, ffn) ||
        !checked_multiply(ffn, input.bytes_per_element, ffn) ||
        !checked_multiply(input.token_count, model.attention_heads, attention_scores) ||
        !checked_multiply(attention_scores, input.context_length, attention_scores) ||
        !checked_multiply(attention_scores, input.bytes_per_element, attention_scores) ||
        !checked_add(hidden, ffn, scratch) || !checked_add(scratch, attention_scores, scratch)) {
        report.blockers.emplace_back("MODEL_SCRATCH_SIZE_OVERFLOW");
        return report;
    }
    report.estimated_scratch_bytes = scratch;
    GpuMemoryPlanInput plan_input;
    plan_input.total_weight_bytes = input.total_weight_bytes;
    plan_input.streaming_chunk_bytes = input.streaming_chunk_bytes;
    plan_input.kv_cache_bytes = report.estimated_kv_cache_bytes;
    plan_input.scratch_bytes = report.estimated_scratch_bytes;
    plan_input.safety_reserve_bytes = input.safety_reserve_bytes;
    plan_input.device_budget_bytes = input.device_budget_bytes;
    plan_input.stream_weights = input.stream_weights;
    report.plan = plan_gpu_memory(plan_input);
    report.blockers = report.plan.blockers;
    return report;
}

GpuMemoryPlanReport plan_gpu_memory(const GpuMemoryPlanInput& input) {
    GpuMemoryPlanReport report;
    report.kv_cache_bytes = input.kv_cache_bytes;
    report.scratch_bytes = input.scratch_bytes;

    if (input.device_budget_bytes == 0) {
        report.blockers.emplace_back("GPU_MEMORY_BUDGET_UNKNOWN");
        return report;
    }
    if (input.stream_weights && input.total_weight_bytes != 0 &&
        input.streaming_chunk_bytes == 0) {
        report.blockers.emplace_back("INVALID_WEIGHT_STREAMING_CHUNK");
        return report;
    }
    report.resident_weight_bytes = input.stream_weights
        ? std::min(input.total_weight_bytes, input.streaming_chunk_bytes)
        : input.total_weight_bytes;

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
