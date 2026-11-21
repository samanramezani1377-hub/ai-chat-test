#pragma once

#include "execution_engine.h"
#include "gguf_reader.h"

#include <functional>
#include <string>
#include <vector>

namespace ai_chat::gpu_engine {

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
    uint64_t preferred_chunk_bytes = 4 * 1024 * 1024;
    uint64_t maximum_chunk_bytes = 0; // zero means use backend-reported limits
    MemoryDomain target_memory = MemoryDomain::Device;
};

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
