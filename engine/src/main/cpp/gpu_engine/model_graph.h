#pragma once

#include "gguf_reader.h"
#include "execution_engine.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace ai_chat::gpu_engine {

class OperatorRegistry;
class ComputeGraph;
class KvCache;
class QuantizedWeightChunkCache;
using ModelTensorBindings = std::unordered_map<std::string, Tensor>;

struct QuantizedWeightChunkBinding {
    uint64_t tensor_offset_bytes = 0;
    uint64_t byte_size = 0;
    std::shared_ptr<Buffer> buffer;
};

struct QuantizedWeightBinding {
    std::shared_ptr<Buffer> buffer;
    uint64_t byte_size = 0;
    uint32_t file_type_id = 0;
    std::vector<uint64_t> dimensions;
    std::vector<QuantizedWeightChunkBinding> chunks;
};

struct ModelExecutionContext {
    ModelTensorBindings tensors;
    std::unordered_map<std::string, QuantizedWeightBinding> quantized_weights;
    std::vector<MemoryLease> weight_leases;
    Backend* backend = nullptr;
    std::vector<uint32_t> token_ids;
    KvCache* kv_cache = nullptr;
    uint64_t past_length = 0;
    uint64_t sequence_length = 0;
    uint64_t context_length = 0;
    bool prefill = false;
    // Expensive GPU readbacks and CPU-reference checks are opt-in diagnostics, never a decode/prefill default.
    bool enable_runtime_quantization_parity_validation = false;
    // Bounded session cache for quantized MatMul weights reused across token steps.
    uint64_t persistent_weight_cache_limit_bytes = 0;
    uint64_t persistent_weight_cache_hits = 0;
    uint64_t persistent_weight_cache_misses = 0;
    std::shared_ptr<QuantizedWeightChunkCache> weight_chunk_cache;
    std::vector<MemoryLease> activation_leases;
    // Recurrent and convolution state survives prefill/decode graph rebuilds.
    std::vector<MemoryLease> state_leases;
};

struct ModelPlanOperation {
    std::string id;
    std::string operator_name;
    std::vector<std::string> inputs;
    std::vector<std::string> weight_tensors;
    std::string output;
    // A leading zero denotes the dynamic token/sequence dimension.
    std::vector<uint64_t> output_shape;
    std::unordered_map<std::string, double> float_attributes;
    std::unordered_map<std::string, uint64_t> integer_attributes;
    std::unordered_map<std::string, bool> bool_attributes;
    std::vector<std::string> control_inputs;
    bool side_effect = false;
    // Typed semantic parameters keep architecture-specific details in model data,
    // not in backend-specific graph code.
    std::unordered_map<std::string, std::string> string_attributes;
    std::unordered_map<std::string, std::vector<uint64_t>> integer_array_attributes;
    std::unordered_map<std::string, std::vector<double>> float_array_attributes;
};

struct ModelExecutionPlan {
    std::string architecture;
    std::string semantic_schema_id;
    std::string semantic_schema_version = "1";
    uint64_t layer_count = 0;
    uint64_t embedding_length = 0;
    uint64_t attention_heads = 0;
    uint64_t kv_heads = 0;
    uint64_t feed_forward_length = 0;
    uint64_t vocabulary_size = 0;
    uint64_t head_dimension = 0;
    uint64_t rotary_dimensions = 0;
    double rms_norm_epsilon = 0.00001;
    double rope_theta = 10000.0;
    bool rope_neox_style = true;
    bool uses_recurrent_state = false;
    bool uses_multi_axis_rope = false;
    std::vector<std::string> external_inputs;
    std::vector<ModelPlanOperation> operations;
    std::vector<std::string> blockers;
    bool structurally_validated = false;
    bool operator_coverage_validated = false;
    bool graph_compiled = false;
    bool execution_validated = false;
    bool gpu_ready = false;
    std::vector<std::string> retained_outputs;
};

struct ModelGraphCoverageReport {
    bool complete = false;
    std::vector<std::string> supported_operators;
    std::vector<std::string> unsupported_operators;
};

struct ModelGraphCompileReport {
    bool compiled = false;
    bool executed = false;
    size_t compiled_nodes = 0;
    uint64_t real_q6_k_parity_checks = 0;
    std::vector<std::string> missing_lowerings;
    std::vector<std::string> unresolved_inputs;
    std::vector<std::string> missing_weight_tensors;
    std::vector<std::string> errors;
};

class ModelLoweringRegistry {
public:
    using Lowering = std::function<Status(const ModelPlanOperation&, const ModelExecutionContext&, ComputeDispatch&)>;
    Status register_lowering(BackendKind backend, const std::string& operator_name,
                             Lowering lowering);
    const Lowering* find(BackendKind backend, const std::string& operator_name) const;
private:
    static std::string make_key(BackendKind backend, const std::string& operator_name);
    std::unordered_map<std::string, Lowering> lowerings_;
};

Status compile_model_execution_plan(ModelExecutionPlan& plan, BackendKind backend,
                                    const ModelLoweringRegistry& lowerings,
                                    const OperatorRegistry& operators,
                                    const std::vector<std::string>& external_inputs,
                                    const ModelExecutionContext& context,
                                    ComputeGraph& graph, ModelGraphCompileReport& report);
Status execute_model_execution_plan_streamed(ModelExecutionPlan& plan, Backend& backend,
                                    const ModelLoweringRegistry& lowerings,
                                    const OperatorRegistry& operators,
                                    const std::vector<std::string>& external_inputs,
                                    const GgufReader& reader, MemoryManager& memory,
                                    const DeviceInfo& device, uint64_t token_count,
                                    const std::atomic<bool>* cancellation,
                                    ModelExecutionContext& context,
                                    ModelGraphCompileReport& report);

Status execute_model_execution_plan(ModelExecutionPlan& plan, Backend& backend,
                                    const ModelLoweringRegistry& lowerings,
                                    const OperatorRegistry& operators,
                                    const std::vector<std::string>& external_inputs,
                                    const ModelExecutionContext& context,
                                    const std::atomic<bool>* cancellation,
                                    ModelGraphCompileReport& report);

Status bind_model_kv_cache_views(const ModelExecutionPlan& plan, KvCache& cache,
                                 uint64_t past_length, uint64_t sequence_length,
                                 ModelExecutionContext& context);

Status validate_model_plan_operator_coverage(ModelExecutionPlan& plan,
                                             const OperatorRegistry& registry,
                                             BackendKind backend,
                                             ModelGraphCoverageReport& report);

} // namespace ai_chat::gpu_engine
