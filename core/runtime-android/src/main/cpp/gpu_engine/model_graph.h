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
using ModelTensorBindings = std::unordered_map<std::string, Tensor>;

struct ModelExecutionContext {
    ModelTensorBindings tensors;
    Backend* backend = nullptr;
    std::vector<uint32_t> token_ids;
    KvCache* kv_cache = nullptr;
    uint64_t past_length = 0;
    uint64_t sequence_length = 0;
    uint64_t context_length = 0;
    bool prefill = false;
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
};

struct ModelExecutionPlan {
    std::string architecture;
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
    std::vector<std::string> external_inputs;
    std::vector<ModelPlanOperation> operations;
    std::vector<std::string> blockers;
    bool structurally_validated = false;
    bool operator_coverage_validated = false;
    bool graph_compiled = false;
    bool execution_validated = false;
    bool gpu_ready = false;
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
