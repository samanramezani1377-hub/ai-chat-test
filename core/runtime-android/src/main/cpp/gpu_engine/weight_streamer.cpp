#include "weight_streamer.h"

#include <algorithm>
#include <limits>
#include <unordered_set>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

namespace ai_chat::gpu_engine {
namespace {

bool round_up(uint64_t value, uint64_t alignment, uint64_t& out) {
    if (alignment == 0) return false;
    const uint64_t remainder = value % alignment;
    if (remainder == 0) { out = value; return true; }
    const uint64_t add = alignment - remainder;
    if (value > std::numeric_limits<uint64_t>::max() - add) return false;
    out = value + add;
    return true;
}

uint64_t gcd(uint64_t a, uint64_t b) {
    while (b != 0) {
        const uint64_t remainder = a % b;
        a = b;
        b = remainder;
    }
    return a;
}

bool lcm_checked(uint64_t a, uint64_t b, uint64_t& out) {
    if (a == 0 || b == 0) return false;
    const uint64_t reduced = a / gcd(a, b);
    if (reduced > std::numeric_limits<uint64_t>::max() / b) return false;
    out = reduced * b;
    return true;
}

} // namespace


Status load_model_execution_weights(const GgufReader& reader,
                                    const ModelExecutionPlan& plan,
                                    Backend& backend, MemoryManager& memory,
                                    const DeviceInfo& device,
                                    ModelExecutionContext& context) {
    if (!backend.healthy())
        return {StatusCode::DeviceLost, "Cannot load model weights to an unhealthy backend"};
    std::unordered_set<std::string> required;
    for (const auto& operation : plan.operations) {
        for (const auto& name : operation.weight_tensors) required.insert(name);
    }

    std::unordered_map<std::string, Tensor> pending_tensors;
    std::unordered_map<std::string, QuantizedWeightBinding> pending_quantized;
    std::vector<MemoryLease> pending_leases;
    for (const auto& name : required) {
        if (context.tensors.find(name) != context.tensors.end() ||
            context.quantized_weights.find(name) != context.quantized_weights.end())
            continue;
        const GgufTensorInfo* tensor = reader.find_tensor(name);
        if (!tensor || !tensor->byte_size_known || tensor->byte_size == 0)
            return {StatusCode::Unsupported, "Model weight tensor is missing or has unknown storage size: " + name};

        const uint64_t backend_limit = device.limits.max_storage_buffer_binding_size != 0
            ? std::min(device.limits.max_storage_buffer_binding_size,
                       device.limits.max_buffer_size == 0 ? device.limits.max_storage_buffer_binding_size
                                                         : device.limits.max_buffer_size)
            : device.limits.max_buffer_size;
        if (backend_limit == 0 || tensor->byte_size > backend_limit)
            return {StatusCode::Unsupported, "Model weight exceeds the single-binding streaming limit: " + name};

        WeightStreamingPolicy policy;
        policy.preferred_chunk_bytes = backend_limit;
        policy.maximum_chunk_bytes = backend_limit;
        policy.target_memory = MemoryDomain::Device;
        LoadedWeightTensor loaded;
        Status status = GgufWeightStreamer::load_tensor(reader, *tensor, backend, memory,
                                                        device, policy, loaded);
        if (!status.ok()) return status;
        if (loaded.resident_chunks.size() != 1 ||
            loaded.resident_chunks[0].logical_size_bytes != tensor->byte_size)
            return {StatusCode::Unsupported, "Model weight requires multi-chunk execution, which is not integrated yet: " + name};

        MemoryLease& lease = loaded.resident_chunks[0].memory;
        if (!lease || !lease.buffer())
            return {StatusCode::BackendFailure, "Weight streamer returned an empty device allocation: " + name};
        const auto buffer = lease.buffer();
        if (tensor->type.name == "F32" && tensor->type.type_id == 0) {
            TensorDesc descriptor;
            descriptor.shape = tensor->dimensions;
            descriptor.data_type = DataType::F32;
            descriptor.layout = TensorLayout::Contiguous;
            descriptor.memory = MemoryDomain::Device;
            descriptor.alignment = std::max<uint64_t>(4, device.limits.min_storage_buffer_offset_alignment);
            Tensor weight;
            status = Tensor::create(std::move(descriptor), buffer, 0, weight);
            if (!status.ok()) return status;
            pending_tensors.emplace(name, std::move(weight));
        } else if (tensor->type.name == "Q8_0" || tensor->type.name == "Q4_0" || tensor->type.name == "Q4_1") {
            QuantizedWeightBinding binding;
            binding.buffer = buffer;
            binding.byte_size = tensor->byte_size;
            binding.file_type_id = tensor->type_id;
            binding.dimensions = tensor->dimensions;
            pending_quantized.emplace(name, std::move(binding));
        } else {
            return {StatusCode::Unsupported, "GPU model weight format has no integrated execution binding yet: " +
                                             tensor->type.name + ":" + name};
        }
        pending_leases.push_back(std::move(lease));
    }

    for (const auto& entry : pending_tensors) {
        if (context.tensors.find(entry.first) != context.tensors.end())
            return {StatusCode::InvalidArgument, "Model weight Tensor binding already exists: " + entry.first};
    }
    for (const auto& entry : pending_quantized) {
        if (context.quantized_weights.find(entry.first) != context.quantized_weights.end())
            return {StatusCode::InvalidArgument, "Quantized model weight binding already exists: " + entry.first};
    }
    for (auto& entry : pending_tensors) context.tensors.emplace(entry.first, std::move(entry.second));
    for (auto& entry : pending_quantized) context.quantized_weights.emplace(entry.first, std::move(entry.second));
    context.weight_leases.insert(context.weight_leases.end(),
                                 std::make_move_iterator(pending_leases.begin()),
                                 std::make_move_iterator(pending_leases.end()));
    return Status::success();
}

Status GgufWeightStreamer::load_tensor(const GgufReader& reader,
                                       const GgufTensorInfo& tensor,
                                       Backend& backend,
                                       MemoryManager& memory,
                                       const DeviceInfo& device,
                                       const WeightStreamingPolicy& policy,
                                       LoadedWeightTensor& out) {
    out = LoadedWeightTensor{};
    const GgufTensorInfo* canonical = reader.find_tensor(tensor.name);
    if (!canonical || !canonical->byte_size_known ||
        canonical->absolute_offset != tensor.absolute_offset ||
        canonical->byte_size != tensor.byte_size) {
        return {StatusCode::Unsupported, "Weight tensor is foreign to the GGUF reader or has unknown storage size"};
    }
    out.name = canonical->name;
    out.dimensions = canonical->dimensions;
    out.type = canonical->type;
    out.total_size_bytes = canonical->byte_size;
    Status status = stream_tensor(reader, tensor, backend, memory, device, policy,
        [&out](WeightChunk&& chunk) {
            out.resident_chunks.push_back(std::move(chunk));
            return Status::success();
        });
    if (!status.ok()) out = LoadedWeightTensor{};
    return status;
}

Status GgufWeightStreamer::stream_tensor(
        const GgufReader& reader, const GgufTensorInfo& tensor,
        Backend& backend, MemoryManager& memory, const DeviceInfo& device,
        const WeightStreamingPolicy& policy,
        const std::function<Status(WeightChunk&&)>& consume) {
    if (!consume) return {StatusCode::InvalidArgument, "Weight streamer requires a chunk consumer"};
    const GgufTensorInfo* canonical = reader.find_tensor(tensor.name);
    if (!canonical || !canonical->byte_size_known ||
        canonical->absolute_offset != tensor.absolute_offset ||
        canonical->byte_size != tensor.byte_size) {
        return {StatusCode::Unsupported, "Weight tensor is foreign to the GGUF reader or has unknown storage size"};
    }
    if (!backend.healthy()) return {StatusCode::DeviceLost, "Cannot stream weights to an unhealthy GPU backend"};
    if (policy.target_memory == MemoryDomain::Host) {
        return {StatusCode::Unsupported, "This streamer currently targets backend-managed device/shared memory"};
    }

    uint64_t backend_limit = device.limits.max_storage_buffer_binding_size;
    if (backend_limit == 0 || (device.limits.max_buffer_size != 0 && device.limits.max_buffer_size < backend_limit)) {
        backend_limit = device.limits.max_buffer_size;
    }
    if (backend_limit == 0) return {StatusCode::Unsupported, "Backend did not report a usable storage-buffer limit"};
    const uint64_t maximum = policy.maximum_chunk_bytes == 0
        ? backend_limit : std::min(policy.maximum_chunk_bytes, backend_limit);
    const uint64_t preferred = policy.preferred_chunk_bytes == 0 ? maximum : policy.preferred_chunk_bytes;
    uint64_t chunk_limit = std::min(preferred, maximum);
    const uint64_t block_bytes = canonical->type.bytes_per_block;
    if (!canonical->type.storage_size_known || block_bytes == 0) {
        return {StatusCode::Unsupported, "GGUF tensor storage block layout is unknown"};
    }
    uint64_t alignment = 0;
    if (!lcm_checked(4, block_bytes, alignment)) {
        return {StatusCode::InvalidArgument, "Weight block alignment overflows"};
    }
    chunk_limit -= chunk_limit % alignment;
    if (chunk_limit == 0) return {StatusCode::Unsupported, "Device storage limit is smaller than one aligned tensor block"};

    uint64_t offset = 0;
    while (offset < canonical->byte_size) {
        const uint64_t remaining = canonical->byte_size - offset;
        const uint64_t logical_size = std::min(remaining, chunk_limit);
        if (logical_size == 0 || (logical_size % block_bytes) != 0) {
            return {StatusCode::InvalidArgument, "Weight chunk splits a quantization storage block"};
        }
        uint64_t allocation_size = 0;
        if (!round_up(logical_size, 4, allocation_size)) {
            return {StatusCode::InvalidArgument, "Weight chunk allocation size overflows"};
        }

        BufferDesc allocation;
        allocation.size_bytes = allocation_size;
        allocation.alignment = std::max<uint64_t>(4, device.limits.min_storage_buffer_offset_alignment);
        allocation.memory = policy.target_memory;
        allocation.persistent = true;
        allocation.label = "weight:" + canonical->name;

        WeightChunk chunk;
        chunk.tensor_offset_bytes = offset;
        chunk.logical_size_bytes = logical_size;
        chunk.allocated_size_bytes = allocation_size;
        Status status = memory.allocate(allocation, chunk.memory);
        if (!status.ok()) return status;

        std::vector<uint8_t> staging(static_cast<size_t>(allocation_size), 0);
        status = reader.read_tensor_chunk(*canonical, offset, staging.data(), logical_size);
        if (!status.ok()) return status;
        status = backend.upload(chunk.memory.buffer(), 0, staging.data(), allocation_size);
        if (!status.ok()) return status;
        status = consume(std::move(chunk));
        if (!status.ok()) return status;
        offset += logical_size;
    }
    return Status::success();
}

} // namespace ai_chat::gpu_engine
