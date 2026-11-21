#pragma once

#include "gpu_engine.h"

#include <cstdint>
#include <random>
#include <vector>

namespace ai_chat::gpu_engine {

struct SamplingConfig {
    float temperature = 1.0f;
    uint32_t top_k = 0;
    float top_p = 1.0f;
    float repetition_penalty = 1.0f;
    uint64_t seed = 0;
};

class Sampler {
public:
    explicit Sampler(SamplingConfig config = {});
    Status set_config(SamplingConfig config);
    Status sample(const std::vector<float>& logits,
                  const std::vector<int32_t>& history,
                  int32_t& out_token);

private:
    SamplingConfig config_;
    std::mt19937_64 random_;
};

class StopController {
public:
    StopController(int32_t eos_token_id = -1,
                   std::vector<std::vector<int32_t>> stop_sequences = {});
    void reset();
    bool push(int32_t token);
    const std::vector<int32_t>& generated_tokens() const noexcept;

private:
    int32_t eos_token_id_ = -1;
    std::vector<std::vector<int32_t>> stop_sequences_;
    std::vector<int32_t> generated_;
    size_t maximum_stop_length_ = 0;
};

} // namespace ai_chat::gpu_engine
