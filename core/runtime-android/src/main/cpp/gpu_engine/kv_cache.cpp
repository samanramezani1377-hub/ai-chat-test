#include "kv_cache.h"

#include <limits>
#include <utility>

namespace ai_chat::gpu_engine {
namespace {

uint64_t data_type_size(DataType type) {
    switch (type) {
        case DataType::F32: return 4;
        case DataType::F16:
        case DataType::BF16: return 2;
        default: return 0;
    }
}

bool multiply_checked(uint64_t a, uint64_t b, uint64_t& out) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) return false;
    out = a * b;
    return true;
}

} // namespace

KvCache::KvCache(Backend& backend, KvCacheConfig config,
                 uint64_t bytes_per_token, uint64_t total_bytes)
    : backend_(backend), config_(config),
      bytes_per_token_(bytes_per_token), total_bytes_(total_bytes) {}

Status KvCache::estimate_bytes(const KvCacheConfig& config, uint64_t& out_bytes) {
    out_bytes = 0;
    const uint64_t item_size = data_type_size(config.data_type);
    if (config.layers == 0 || config.kv_heads == 0 || config.head_dimension == 0 ||
        config.maximum_sequence_length == 0 || item_size == 0) {
        return {StatusCode::InvalidArgument, "KV cache dimensions or data type are invalid"};
    }
    uint64_t bytes = 0;
    if (!multiply_checked(config.kv_heads, config.head_dimension, bytes) ||
        !multiply_checked(bytes, item_size, bytes) ||
        !multiply_checked(bytes, config.maximum_sequence_length, bytes) ||
        !multiply_checked(bytes, config.layers, bytes) ||
        !multiply_checked(bytes, 2, out_bytes)) {
        return {StatusCode::InvalidArgument, "KV cache memory estimate overflows uint64"};
    }
    return Status::success();
}

Status KvCache::create(Backend& backend, MemoryManager& memory,
                       KvCacheConfig config, std::unique_ptr<KvCache>& out) {
    out.reset();
    uint64_t total_bytes = 0;
    Status status = estimate_bytes(config, total_bytes);
    if (!status.ok()) return status;
    uint64_t bytes_per_token = 0;
    if (!multiply_checked(config.kv_heads, config.head_dimension, bytes_per_token) ||
        !multiply_checked(bytes_per_token, data_type_size(config.data_type), bytes_per_token)) {
        return {StatusCode::InvalidArgument, "KV cache token stride overflows"};
    }
    if (!backend.healthy()) return {StatusCode::DeviceLost, "Cannot allocate KV cache on an unhealthy backend"};

    auto cache = std::unique_ptr<KvCache>(
        new KvCache(backend, config, bytes_per_token, total_bytes));
    cache->layers_.resize(config.layers);
    uint64_t per_layer_bytes = 0;
    if (!multiply_checked(bytes_per_token, config.maximum_sequence_length, per_layer_bytes)) {
        return {StatusCode::InvalidArgument, "KV cache layer size overflows"};
    }

    for (uint32_t layer = 0; layer < config.layers; ++layer) {
        BufferDesc desc;
        desc.size_bytes = per_layer_bytes;
        desc.alignment = 4;
        desc.memory = MemoryDomain::Device;
        desc.persistent = true;
        desc.label = "kv-cache-key-layer-" + std::to_string(layer);
        status = memory.allocate(desc, cache->layers_[layer].keys);
        if (!status.ok()) return status;
        desc.label = "kv-cache-value-layer-" + std::to_string(layer);
        status = memory.allocate(desc, cache->layers_[layer].values);
        if (!status.ok()) return status;
    }
    out = std::move(cache);
    return Status::success();
}

Status KvCache::append(uint32_t layer, uint32_t start_position, uint32_t token_count,
                       const void* keys, const void* values) {
    if (layer >= layers_.size() || token_count == 0 || !keys || !values)
        return {StatusCode::InvalidArgument, "KV cache append arguments are invalid"};
    if (!backend_.healthy()) return {StatusCode::DeviceLost, "KV cache backend is unhealthy"};
    Layer& state = layers_[layer];
    if (start_position != state.valid_tokens ||
        token_count > config_.maximum_sequence_length - start_position) {
        return {StatusCode::InvalidArgument, "KV cache append must be sequential and remain within capacity"};
    }
    uint64_t byte_offset = 0, byte_count = 0;
    if (!multiply_checked(start_position, bytes_per_token_, byte_offset) ||
        !multiply_checked(token_count, bytes_per_token_, byte_count)) {
        return {StatusCode::InvalidArgument, "KV cache append range overflows"};
    }
    Status status = backend_.upload(state.keys.buffer(), byte_offset, keys, byte_count);
    if (!status.ok()) return status;
    status = backend_.upload(state.values.buffer(), byte_offset, values, byte_count);
    if (!status.ok()) return status;
    state.valid_tokens += token_count;
    return Status::success();
}

Status KvCache::reset_layer(uint32_t layer) {
    if (layer >= layers_.size()) return {StatusCode::InvalidArgument, "KV cache layer index is invalid"};
    layers_[layer].valid_tokens = 0;
    return Status::success();
}

uint32_t KvCache::valid_tokens(uint32_t layer) const noexcept {
    return layer < layers_.size() ? layers_[layer].valid_tokens : 0;
}

std::shared_ptr<Buffer> KvCache::key_buffer(uint32_t layer) const noexcept {
    return layer < layers_.size() ? layers_[layer].keys.buffer() : std::shared_ptr<Buffer>{};
}

std::shared_ptr<Buffer> KvCache::value_buffer(uint32_t layer) const noexcept {
    return layer < layers_.size() ? layers_[layer].values.buffer() : std::shared_ptr<Buffer>{};
}

uint64_t KvCache::bytes_per_token() const noexcept { return bytes_per_token_; }
uint64_t KvCache::total_bytes() const noexcept { return total_bytes_; }

} // namespace ai_chat::gpu_engine
