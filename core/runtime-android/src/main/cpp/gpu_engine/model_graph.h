#pragma once

#include "gguf_reader.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace ai_chat::gpu_engine {

class OperatorRegistry;
class ComputeGraph;

struct ModelPlanOperation {
    std::string id;
    std::string operator_name;
    std::vector<std::string> inputs;
    std::vector<std::string> weight_tensors;
    std::string output;
    std::vector<uint64_t> output_shape;
};

struct ModelExecutionPlan {
    std::string architecture;
    uint64_t layer_count = 0;
    uint64_t embedding_length = 0;
    uint64_t attention_heads = 0;
    uint64_t kv_heads = 0;
    uint64_t feed_forward_length = 0;
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
    size_t compiled_nodes = 0;
    std::vector<std::string> missing_lowerings;
    std::vector<std::string> unresolved_inputs;
    std::vector<std::string> missing_weight_tensors;
    std::vector<std::string> errors;
};

class ModelLoweringRegistry {
public:
    using Lowering = std::function<Status(const ModelPlanOperation&, ComputeDispatch&)>;
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
                                    const std::vector<std::string>& available_weight_tensors,
                                    ComputeGraph& graph, ModelGraphCompileReport& report);

using TensorBindings = std::unordered_map<std::string, Tensor>;
using OperatorDispatchBuilder = std::function<Status(
    const ModelPlanOperation&, const std::vector<Tensor>&,
    const std::vector<Tensor>&, const Tensor&, ComputeDispatch&)>;

struct OperatorDispatchBuilderDefinition {
    std::string operator_name;
    BackendKind backend = BackendKind::Unknown;
    SupportState validation = SupportState::Unverified;
    OperatorDispatchBuilder build;
};

class OperatorDispatchRegistry {
public:
    Status register_builder(OperatorDispatchBuilderDefinition definition);
    const OperatorDispatchBuilderDefinition* find(const std::string& operator_name,
                                                   BackendKind backend) const noexcept;
private:
    std::unordered_map<std::string, OperatorDispatchBuilderDefinition> builders_;
};

class ModelGraphCompiler {
public:
    Status compile(const ModelExecutionPlan& plan, const TensorBindings& tensors,
                   const OperatorDispatchRegistry& builders,
                   const OperatorRegistry& operators, BackendKind backend,
                   ComputeGraph& graph) const;
};

Status validate_model_plan_operator_coverage(ModelExecutionPlan& plan,
                                             const OperatorRegistry& registry,
                                             BackendKind backend,
                                             ModelGraphCoverageReport& report);

} // namespace ai_chat::gpu_engine
