#include "weight_streamer.h"

#include <algorithm>
#include <limits>
#include <string>
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
    if (!backend.healthy()) return {StatusCode::DeviceLost, "Cannot stream weights to an unhealthy GPU backend"};
    if (policy.target_memory == MemoryDomain::Host) {
        return {StatusCode::Unsupported, "This streamer currently targets backend-managed device/shared memory"};
    }

    uint64_t maximum = policy.maximum_chunk_bytes;
    if (maximum == 0) {
        maximum = device.limits.max_storage_buffer_binding_size;
        if (maximum == 0 || (device.limits.max_buffer_size != 0 && device.limits.max_buffer_size < maximum)) {
            maximum = device.limits.max_buffer_size;
        }
    }
    if (maximum == 0) return {StatusCode::Unsupported, "Backend did not report a usable storage-buffer limit"};
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

    out.name = canonical->name;
    out.dimensions = canonical->dimensions;
    out.type = canonical->type;
    out.total_size_bytes = canonical->byte_size;
    uint64_t offset = 0;
    while (offset < canonical->byte_size) {
        const uint64_t remaining = canonical->byte_size - offset;
        const uint64_t logical_size = std::min(remaining, chunk_limit);
        if (logical_size == 0 || (logical_size % block_bytes) != 0) {
            out = LoadedWeightTensor{};
            return {StatusCode::InvalidArgument, "Weight chunk splits a quantization storage block"};
        }
        uint64_t allocation_size = 0;
        if (!round_up(logical_size, 4, allocation_size)) {
            out = LoadedWeightTensor{};
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
        if (!status.ok()) {
            out = LoadedWeightTensor{};
            return status;
        }
        std::vector<uint8_t> staging(static_cast<size_t>(allocation_size), 0);
        status = reader.read_tensor_chunk(*canonical, offset, staging.data(), logical_size);
        if (!status.ok()) {
            out = LoadedWeightTensor{};
            return status;
        }
        status = backend.upload(chunk.memory.buffer(), 0, staging.data(), allocation_size);
        if (!status.ok()) {
            out = LoadedWeightTensor{};
            return status;
        }
        out.resident_chunks.push_back(std::move(chunk));
        offset += logical_size;
    }
    return Status::success();
}

} // namespace ai_chat::gpu_engine
