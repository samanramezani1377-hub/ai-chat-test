#pragma once

#include "execution_engine.h"
#include "gguf_reader.h"
#include "model_graph.h"

#include <functional>
#include <list>
#include <mutex>
#include <memory>
#include <unordered_map>
#include <string>
#include <vector>

namespace ai_chat::gpu_engine {


struct QuantizedWeightChunkCacheStats {
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;
    uint64_t resident_bytes = 0;
    uint64_t resident_chunks = 0;
};

// GPU-resident, bounded LRU cache for immutable GGUF weight slices.
// Leases are shared with in-flight dispatches, so eviction never invalidates
// a buffer that is still referenced by a queued command batch.
class QuantizedWeightChunkCache {
public:
    explicit QuantizedWeightChunkCache(uint64_t capacity_bytes = 0);
    bool find(const std::string& key, MemoryLease& out);
    void insert(const std::string& key, uint64_t bytes, const MemoryLease& lease);
    void set_capacity(uint64_t capacity_bytes);
    // Drops all idle cached chunks without disabling future admissions.
    void evict_all();
    void clear();
    QuantizedWeightChunkCacheStats stats() const;

private:
    struct Entry {
        std::string key;
        uint64_t bytes = 0;
        MemoryLease lease;
    };
    mutable std::mutex mutex_;
    uint64_t capacity_bytes_ = 0;
    uint64_t resident_bytes_ = 0;
    uint64_t hits_ = 0;
    uint64_t misses_ = 0;
    uint64_t evictions_ = 0;
    std::list<Entry> lru_;
    std::unordered_map<std::string, std::list<Entry>::iterator> index_;
};

struct WeightChunk {
    uint64_t tensor_offset_bytes = 0;
    uint64_t logical_size_bytes = 0;
    uint64_t allocated_size_bytes = 0;
    MemoryLease memory;
};

struct LoadedWeightTensor {
    std::string name;
    std::vector<uint64_t> dimensions;
    GgufTensorTypeInfo type;
    uint64_t total_size_bytes = 0;
    std::vector<WeightChunk> resident_chunks;
};

struct WeightStreamingPolicy {
    uint64_t preferred_chunk_bytes = 0; // zero selects an adaptive backend/memory-aware size
    uint64_t maximum_chunk_bytes = 0; // zero means use backend-reported limits
    uint64_t required_chunk_alignment_bytes = 0; // operation-level row/group boundary
    MemoryDomain target_memory = MemoryDomain::Device;
};

Status load_model_execution_weights(const GgufReader& reader,
                                    const ModelExecutionPlan& plan,
                                    Backend& backend, MemoryManager& memory,
                                    const DeviceInfo& device,
                                    ModelExecutionContext& context);

class GgufWeightStreamer {
public:
    // Streams one validated chunk at a time. The callback may retain the chunk's
    // MemoryLease; otherwise its GPU allocation is released before the next chunk.
    static Status stream_tensor(const GgufReader& reader,
                                const GgufTensorInfo& tensor,
                                Backend& backend,
                                MemoryManager& memory,
                                const DeviceInfo& device,
                                const WeightStreamingPolicy& policy,
                                const std::function<Status(WeightChunk&&)>& consume);
    static Status load_tensor(const GgufReader& reader,
                              const GgufTensorInfo& tensor,
                              Backend& backend,
                              MemoryManager& memory,
                              const DeviceInfo& device,
                              const WeightStreamingPolicy& policy,
                              LoadedWeightTensor& out);
};

} // namespace ai_chat::gpu_engine
