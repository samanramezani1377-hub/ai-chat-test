#pragma once

#include "android_backends.h"
#include "kv_cache.h"
#include "memory_planner.h"
#include "model_registry.h"
#include "quantization.h"
#include "sampler.h"
#include "tokenizer.h"
#include "weight_streamer.h"

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace ai_chat::gpu_engine {

// A deliberately fail-closed GPU-only inference session. This standalone
// Android engine does not provide or silently select a CPU inference fallback.
class GpuInferenceSession {
public:
    using TextCallback = std::function<void(const std::string&)>;

    GpuInferenceSession() = default;
    ~GpuInferenceSession();

    GpuInferenceSession(const GpuInferenceSession&) = delete;
    GpuInferenceSession& operator=(const GpuInferenceSession&) = delete;

    Status load_model(const std::string& path, uint32_t context_length,
                      uint64_t memory_budget_bytes);
    Status update_memory_budget(uint64_t allocation_cap_bytes);
    Status generate(const std::string& prompt, uint32_t max_new_tokens,
                    SamplingConfig sampling, const TextCallback& on_text);
    Status count_tokens(const std::string& text, uint32_t& out_count) const;
    void cancel() noexcept;
    void reset_stop() noexcept;
    void unload() noexcept;
    bool loaded() const noexcept;
    bool uses_qwen35_tokenizer() const noexcept;
    bool uses_nfc_tokenizer() const noexcept;
    std::string diagnostics() const;

private:
    Status read_last_token_logits(std::vector<float>& logits);
    Status run_model_step(uint64_t token_count);
    void unload_locked() noexcept;
    std::string backend_name() const;

    mutable std::mutex mutex_;
    std::atomic<bool> cancelled_{false};
    std::unique_ptr<GpuEngine> engine_;
    std::unique_ptr<OperatorRegistry> operators_;
    std::unique_ptr<QuantizationRegistry> quantizations_;
    std::unique_ptr<ArchitectureRegistry> architectures_;
    std::unique_ptr<TokenizerRegistry> tokenizers_;
    std::unique_ptr<ModelLoweringRegistry> lowerings_;
    std::unique_ptr<GgufModelLoader> loader_;
    std::unique_ptr<MemoryManager> memory_;
    std::unique_ptr<KvCache> kv_cache_;
    ModelExecutionPlan plan_;
    ModelExecutionContext context_;
    ModelGpuMemoryEstimateReport memory_estimate_;
    uint32_t context_length_ = 0;
    uint64_t memory_budget_bytes_ = 0;
    uint64_t total_weight_bytes_ = 0;
    uint64_t largest_weight_tensor_bytes_ = 0;
    uint64_t peak_operation_weight_bytes_ = 0;
    uint64_t kv_cache_bytes_ = 0;
    uint64_t last_tokenization_ms_ = 0;
    uint64_t last_prefill_ms_ = 0;
    uint64_t last_prefill_readback_ms_ = 0;
    uint64_t last_ttft_ms_ = 0;
    uint64_t last_decode_ms_ = 0;
    bool last_tokenization_timing_valid_ = false;
    bool last_prefill_timing_valid_ = false;
    bool last_prefill_readback_timing_valid_ = false;
    bool last_ttft_timing_valid_ = false;
    bool last_decode_timing_valid_ = false;
    std::string last_generation_stage_ = "not_started";
    std::string last_model_graph_validation_ = "not_run";
    std::string q6_backend_validation_ = "not_run";
    uint64_t last_prompt_tokens_ = 0;
    uint64_t last_output_tokens_ = 0;
    double last_decode_tokens_per_second_ = 0.0;
    bool last_generation_completed_ = false;

    std::string last_attempt_model_path_;
    std::string last_attempt_architecture_;
    std::string last_attempt_backend_;
    std::string last_attempt_gpu_vendor_;
    std::string last_attempt_gpu_device_;
    std::string last_attempt_gpu_architecture_;
    std::string last_attempt_gpu_driver_;
    std::string last_attempt_gpu_api_;
    std::vector<std::string> last_attempt_gpu_features_;
    uint64_t last_attempt_memory_budget_bytes_ = 0;
    uint64_t last_attempt_weight_bytes_ = 0;
    uint64_t last_attempt_largest_weight_tensor_bytes_ = 0;
    uint64_t last_attempt_peak_operation_weight_bytes_ = 0;
    uint64_t last_attempt_kv_cache_bytes_ = 0;
    uint64_t last_attempt_gpu_max_buffer_bytes_ = 0;
    uint64_t last_attempt_gpu_max_storage_binding_bytes_ = 0;
    uint32_t last_attempt_context_length_ = 0;
    uint32_t last_attempt_gpu_max_workgroup_invocations_ = 0;
    bool last_attempt_gpu_compute_available_ = false;
    bool last_attempt_gpu_self_test_passed_ = false;

    uint64_t last_failed_memory_budget_bytes_ = 0;
    uint64_t last_failed_memory_required_bytes_ = 0;
    uint64_t last_failed_memory_usable_bytes_ = 0;
    uint64_t last_failed_memory_kv_bytes_ = 0;
    uint64_t last_failed_memory_scratch_bytes_ = 0;
    uint64_t last_failed_tensor_bytes_ = 0;
    uint64_t last_failed_storage_binding_limit_bytes_ = 0;
    std::string model_path_;
    std::string backend_fallback_reason_;
    std::string last_error_;
    bool loaded_ = false;
};

} // namespace ai_chat::gpu_engine
