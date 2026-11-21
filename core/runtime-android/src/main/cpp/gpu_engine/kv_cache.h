#pragma once

#include "execution_engine.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace ai_chat::gpu_engine {

struct KvCacheConfig {
    uint32_t layers = 0;
    uint32_t kv_heads = 0;
    uint32_t head_dimension = 0;
    uint32_t maximum_sequence_length = 0;
    DataType data_type = DataType::F32;
};

class KvCache {
public:
    static Status estimate_bytes(const KvCacheConfig& config, uint64_t& out_bytes);
    static Status create(Backend& backend, MemoryManager& memory,
                         KvCacheConfig config, std::unique_ptr<KvCache>& out);

    Status append(uint32_t layer, uint32_t start_position, uint32_t token_count,
                  const void* keys, const void* values);
    // Copies device-resident K/V tensors into the cache without a host readback.
    Status append_device(uint32_t layer, uint32_t start_position, uint32_t token_count,
                         const std::shared_ptr<Buffer>& keys, uint64_t keys_offset,
                         const std::shared_ptr<Buffer>& values, uint64_t values_offset);
    Status reset_layer(uint32_t layer);
    uint32_t valid_tokens(uint32_t layer) const noexcept;
    std::shared_ptr<Buffer> key_buffer(uint32_t layer) const noexcept;
    std::shared_ptr<Buffer> value_buffer(uint32_t layer) const noexcept;
    Status key_tensor(uint32_t layer, Tensor& out) const;
    Status value_tensor(uint32_t layer, Tensor& out) const;
    uint64_t bytes_per_token() const noexcept;
    uint64_t total_bytes() const noexcept;

private:
    struct Layer {
        MemoryLease keys;
        MemoryLease values;
        uint32_t valid_tokens = 0;
    };

    KvCache(Backend& backend, KvCacheConfig config,
            uint64_t bytes_per_token, uint64_t total_bytes);
    Backend& backend_;
    KvCacheConfig config_;
    uint64_t bytes_per_token_ = 0;
    uint64_t total_bytes_ = 0;
    std::vector<Layer> layers_;
};

} // namespace ai_chat::gpu_engine
