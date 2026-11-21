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


Status allocate_model_activation_tensors(const ModelExecutionPlan& model, MemoryManager& memory,
                                         uint64_t token_count, ModelExecutionContext& context) {
    if (!model.structurally_validated || token_count == 0)
        return {StatusCode::InvalidArgument, "Activation allocation requires a validated model plan and non-zero token count"};

    std::unordered_map<std::string, size_t> last_use;
    for (size_t index = 0; index < model.operations.size(); ++index) {
        const auto& operation = model.operations[index];
        if (operation.side_effect || operation.output.empty()) continue;
        for (const auto& input : operation.inputs) last_use[input] = index;
    }

    struct Slot {
        std::shared_ptr<Buffer> buffer;
        uint64_t size_bytes = 0;
        size_t available_after = 0;
    };
    std::vector<Slot> slots;
    std::vector<MemoryLease> leases;
    std::unordered_map<std::string, Tensor> pending_tensors;
    std::unordered_map<std::string, size_t> producers;
    const std::unordered_map<std::string, bool> retained(
        model.retained_outputs.begin(), model.retained_outputs.end());

    for (size_t index = 0; index < model.operations.size(); ++index) {
        const auto& operation = model.operations[index];
        if (operation.side_effect || operation.output.empty()) continue;
        if (context.tensors.find(operation.output) != context.tensors.end())
            continue;
        if (operation.output_shape.empty())
            return {StatusCode::InvalidArgument, "Model activation output has no declared shape: " + operation.output};
        if (producers.find(operation.output) != producers.end())
            return {StatusCode::InvalidArgument, "Model plan writes the same activation name more than once: " + operation.output};
        producers.emplace(operation.output, index);

        TensorDesc descriptor;
        descriptor.shape = operation.output_shape;
        for (uint64_t& dimension : descriptor.shape) {
            if (dimension == 0) dimension = token_count;
            if (dimension == 0)
                return {StatusCode::InvalidArgument, "Model activation shape contains a zero dimension"};
        }
        descriptor.data_type = DataType::F32;
        descriptor.layout = TensorLayout::Contiguous;
        descriptor.memory = MemoryDomain::Device;
        descriptor.alignment = 4;
        uint64_t required_bytes = 0;
        Status status = Tensor::required_bytes(descriptor, required_bytes);
        if (!status.ok()) return status;

        Slot* selected = nullptr;
        for (auto& slot : slots) {
            if (slot.available_after < index && slot.size_bytes >= required_bytes) {
                selected = &slot;
                break;
            }
        }
        if (!selected) {
            BufferDesc buffer_desc;
            buffer_desc.size_bytes = required_bytes;
            buffer_desc.alignment = descriptor.alignment;
            buffer_desc.memory = MemoryDomain::Device;
            buffer_desc.persistent = false;
            buffer_desc.label = "activation-" + operation.output;
            MemoryLease lease;
            status = memory.allocate(buffer_desc, lease);
            if (!status.ok()) return status;
            if (!lease || !lease.buffer() || lease.buffer()->size_bytes() < required_bytes)
                return {StatusCode::BackendFailure, "Memory manager returned an undersized activation lease"};
            leases.push_back(std::move(lease));
            slots.push_back({leases.back().buffer(), required_bytes, index});
            selected = &slots.back();
        }

        Tensor tensor;
        status = Tensor::create(std::move(descriptor), selected->buffer, 0, tensor);
        if (!status.ok()) return status;
        pending_tensors.emplace(operation.output, std::move(tensor));
        const auto last = last_use.find(operation.output);
        selected->available_after = retained.find(operation.output) != retained.end()
            ? std::numeric_limits<size_t>::max()
            : (last == last_use.end() ? index : last->second);
    }

    for (auto& entry : pending_tensors) {
        if (!context.tensors.emplace(entry.first, std::move(entry.second)).second)
            return {StatusCode::InvalidArgument, "Activation tensor binding was concurrently created"};
    }
    context.activation_leases.insert(context.activation_leases.end(),
                                     std::make_move_iterator(leases.begin()),
                                     std::make_move_iterator(leases.end()));
    return Status::success();
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
