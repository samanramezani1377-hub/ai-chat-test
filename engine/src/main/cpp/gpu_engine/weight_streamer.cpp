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
    std::vector<uint32_t> pending_token_ids;
    bool remap_embedding_token_ids = false;
    const bool single_operation = plan.operations.size() == 1;
    const std::string operation_name = single_operation ? plan.operations.front().operator_name : std::string();

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
        if (backend_limit == 0)
            return {StatusCode::Unsupported, "Backend did not report a usable model-weight storage limit"};

        const bool q6k_matmul = single_operation && operation_name == "MatMul_Q6_K" &&
                                tensor->type.name == "Q6_K" && tensor->type_id == 14;
        const bool q6k_embedding = single_operation && operation_name == "Embedding_Q6_K" &&
                                   tensor->type.name == "Q6_K" && tensor->type_id == 14;

        if (q6k_matmul) {
            if (tensor->dimensions.size() != 2 || tensor->dimensions[0] == 0 ||
                tensor->dimensions[0] % 256 != 0 || tensor->dimensions[1] == 0)
                return {StatusCode::Unsupported, "Q6_K MatMul tensor dimensions are not channel-chunkable: " + name};
            const uint64_t row_bytes = (tensor->dimensions[0] / 256) * 210;
            if (row_bytes == 0 || row_bytes > backend_limit)
                return {StatusCode::Unsupported, "Q6_K MatMul one output-channel row exceeds the backend binding limit: " + name};
            // Keep only metadata here. The execution path reads, uploads, dispatches,
            // and releases one complete output-channel group at a time.
            QuantizedWeightBinding binding;
            binding.byte_size = tensor->byte_size;
            binding.file_type_id = tensor->type_id;
            binding.dimensions = tensor->dimensions;
            pending_quantized.emplace(name, std::move(binding));
            continue;
        }

        if (q6k_embedding) {
            if (tensor->dimensions.size() != 2 || tensor->dimensions[0] == 0 ||
                tensor->dimensions[0] % 256 != 0 || tensor->dimensions[1] == 0 ||
                context.token_ids.empty())
                return {StatusCode::Unsupported, "Q6_K Embedding metadata or token batch is invalid: " + name};
            const uint64_t row_bytes = (tensor->dimensions[0] / 256) * 210;
            if (row_bytes == 0 || row_bytes > backend_limit ||
                context.token_ids.size() > UINT32_MAX ||
                context.token_ids.size() > UINT64_MAX / row_bytes)
                return {StatusCode::Unsupported, "Q6_K Embedding gathered rows exceed safe storage limits: " + name};
            const uint64_t logical_bytes = context.token_ids.size() * row_bytes;
            if (logical_bytes > backend_limit || logical_bytes > SIZE_MAX - 3)
                return {StatusCode::Unsupported, "Q6_K Embedding token-row gather exceeds the backend binding limit"};
            uint64_t allocated_bytes = 0;
            if (!round_up(logical_bytes, 4, allocated_bytes))
                return {StatusCode::InvalidArgument, "Q6_K Embedding gathered allocation size overflows"};

            BufferDesc allocation;
            allocation.size_bytes = allocated_bytes;
            allocation.alignment = std::max<uint64_t>(4, device.limits.min_storage_buffer_offset_alignment);
            allocation.memory = MemoryDomain::Device;
            allocation.persistent = false;
            allocation.label = "q6-k-embedding-gather:" + name;
            MemoryLease lease;
            Status status = memory.allocate(allocation, lease);
            if (!status.ok()) return status;
            if (!lease || !lease.buffer())
                return {StatusCode::BackendFailure, "Q6_K Embedding gather allocation returned an empty buffer"};

            std::vector<uint8_t> staging(static_cast<size_t>(allocated_bytes), 0);
            pending_token_ids.resize(context.token_ids.size());
            for (size_t index = 0; index < context.token_ids.size(); ++index) {
                const uint32_t token_id = context.token_ids[index];
                if (token_id >= tensor->dimensions[1])
                    return {StatusCode::InvalidArgument, "Q6_K Embedding token ID is outside the GGUF vocabulary"};
                status = reader.read_tensor_chunk(*tensor, static_cast<uint64_t>(token_id) * row_bytes,
                    staging.data() + index * row_bytes, row_bytes);
                if (!status.ok()) return status;
                pending_token_ids[index] = static_cast<uint32_t>(index);
            }
            status = backend.upload(lease.buffer(), 0, staging.data(), allocated_bytes);
            if (!status.ok()) return status;

            QuantizedWeightBinding binding;
            binding.buffer = lease.buffer();
            binding.byte_size = logical_bytes;
            binding.file_type_id = tensor->type_id;
            binding.dimensions = {tensor->dimensions[0], static_cast<uint64_t>(context.token_ids.size())};
            pending_quantized.emplace(name, std::move(binding));
            pending_leases.push_back(std::move(lease));
            remap_embedding_token_ids = true;
            continue;
        }

        if (tensor->byte_size > backend_limit)
            return {StatusCode::Unsupported, "Model weight exceeds the single-binding limit and has no chunk-aware operator lowering: " +
                name + " | tensor_bytes=" + std::to_string(tensor->byte_size) +
                " | max_storage_binding_bytes=" + std::to_string(backend_limit)};

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
            return {StatusCode::Unsupported, "Model weight requires an operator-specific chunk lowering: " + name};

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
        } else if (tensor->type.name == "Q8_0" || tensor->type.name == "Q4_0" || tensor->type.name == "Q4_1" || tensor->type.name == "Q5_0" || tensor->type.name == "Q5_1" || tensor->type.name == "F16" || tensor->type.name == "BF16" || tensor->type.name == "Q6_K" || tensor->type.name == "Q4_K" || tensor->type.name == "Q5_K" || tensor->type.name == "Q2_K" || tensor->type.name == "Q3_K") {
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
    if (remap_embedding_token_ids) context.token_ids = std::move(pending_token_ids);
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
