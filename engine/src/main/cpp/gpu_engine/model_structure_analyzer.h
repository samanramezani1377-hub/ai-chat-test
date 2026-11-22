#pragma once

#include "gguf_reader.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ai_chat::gpu_engine {

// This is structural evidence only. It never authorizes model execution.
enum class ModelStructureFamily : uint8_t {
    Unknown = 0,
    DenseTransformer,
    MixtureOfExperts,
    HybridAttentionRecurrent,
    Recurrent,
    Multimodal,
};

struct ModelTensorRoleEvidence {
    std::string tensor_name;
    std::string role;
    uint64_t layer_index = UINT64_MAX;
};

struct ModelStructureAnalysis {
    ModelStructureFamily candidate_family = ModelStructureFamily::Unknown;
    uint32_t evidence_score = 0; // Heuristic evidence score, not a probability.
    uint64_t recognized_tensor_count = 0;
    uint64_t layer_count = 0;
    uint64_t attention_tensor_count = 0;
    uint64_t recurrent_tensor_count = 0;
    uint64_t expert_tensor_count = 0;
    uint64_t multimodal_tensor_count = 0;
    bool has_embedding = false;
    bool has_output_projection = false;
    bool has_attention = false;
    bool has_recurrent_path = false;
    bool has_experts = false;
    bool has_multimodal_components = false;
    std::vector<ModelTensorRoleEvidence> evidence;
    std::vector<std::string> blockers;
};

ModelStructureAnalysis analyze_model_structure(const GgufFileInfo& model,
                                                size_t maximum_evidence = 512);
const char* model_structure_family_name(ModelStructureFamily family) noexcept;

} // namespace ai_chat::gpu_engine
