#pragma once

#include "gguf_reader.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ai_chat::gpu_engine {

class OperatorRegistry;

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
    std::vector<ModelPlanOperation> operations;
    std::vector<std::string> blockers;
    bool structurally_validated = false;
    bool operator_coverage_validated = false;
    bool execution_validated = false;
    bool gpu_ready = false;
};

struct ModelGraphCoverageReport {
    bool complete = false;
    std::vector<std::string> supported_operators;
    std::vector<std::string> unsupported_operators;
};

Status validate_model_plan_operator_coverage(ModelExecutionPlan& plan,
                                             const OperatorRegistry& registry,
                                             BackendKind backend,
                                             ModelGraphCoverageReport& report);

} // namespace ai_chat::gpu_engine
