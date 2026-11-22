#include "model_structure_analyzer.h"

#include <algorithm>
#include <charconv>
#include <initializer_list>
#include <set>
#include <string_view>

namespace ai_chat::gpu_engine {
namespace {

bool parse_layer_index(const std::string& name, uint64_t& index) {
    constexpr std::string_view prefix{"blk."};
    if (name.compare(0, prefix.size(), prefix) != 0) return false;
    const size_t end = name.find('.', prefix.size());
    if (end == std::string::npos || end == prefix.size()) return false;
    const char* begin = name.data() + prefix.size();
    const char* finish = name.data() + end;
    const auto parsed = std::from_chars(begin, finish, index);
    return parsed.ec == std::errc{} && parsed.ptr == finish;
}

bool contains_any(const std::string& value,
                  std::initializer_list<std::string_view> needles) {
    for (const auto needle : needles)
        if (value.find(needle.data(), 0, needle.size()) != std::string::npos) return true;
    return false;
}

std::string classify_tensor(const std::string& name, bool& attention,
                            bool& recurrent, bool& expert, bool& multimodal) {
    attention = contains_any(name, {"attn_", "attention", "self_attn", "q_proj", "k_proj", "v_proj"});
    recurrent = contains_any(name, {"ssm_", "linear_attn", "time_mix", "conv1d", "recurrent", "state_space"});
    expert = contains_any(name, {"_exps", "expert", "experts", "ffn_gate_inp", "router"});
    multimodal = contains_any(name, {"vision", "visual", "image", "audio", "mm_projector",
                                     "patch_embd", "vocab_audio", "token_embd_img"});
    if (name == "token_embd.weight") return "token_embedding";
    if (name == "output.weight") return "output_projection";
    if (name == "output_norm.weight") return "output_norm";
    if (attention) {
        if (contains_any(name, {"attn_q", "q_proj"})) return "attention_query";
        if (contains_any(name, {"attn_k", "k_proj"})) return "attention_key";
        if (contains_any(name, {"attn_v", "v_proj"})) return "attention_value";
        if (contains_any(name, {"attn_output", "o_proj"})) return "attention_output";
        if (contains_any(name, {"attn_norm"})) return "attention_norm";
        return "attention_component";
    }
    if (recurrent) return "recurrent_component";
    if (expert) {
        if (contains_any(name, {"gate_inp", "router"})) return "expert_router";
        if (contains_any(name, {"gate_exps"})) return "expert_gate";
        if (contains_any(name, {"up_exps"})) return "expert_up";
        if (contains_any(name, {"down_exps"})) return "expert_down";
        return "expert_component";
    }
    if (multimodal) return "multimodal_component";
    if (name.find("ffn_gate") != std::string::npos) return "feed_forward_gate";
    if (name.find("ffn_up") != std::string::npos) return "feed_forward_up";
    if (name.find("ffn_down") != std::string::npos) return "feed_forward_down";
    if (name.find("norm") != std::string::npos) return "normalization";
    return {};
}

} // namespace

const char* model_structure_family_name(ModelStructureFamily family) noexcept {
    switch (family) {
        case ModelStructureFamily::DenseTransformer: return "dense_transformer";
        case ModelStructureFamily::MixtureOfExperts: return "mixture_of_experts";
        case ModelStructureFamily::HybridAttentionRecurrent: return "hybrid_attention_recurrent";
        case ModelStructureFamily::Recurrent: return "recurrent";
        case ModelStructureFamily::Multimodal: return "multimodal";
        case ModelStructureFamily::Unknown: default: return "unknown";
    }
}

ModelStructureAnalysis analyze_model_structure(const GgufFileInfo& model,
                                                size_t maximum_evidence) {
    ModelStructureAnalysis result;
    std::set<uint64_t> layers;
    const bool has_general_architecture = !model.architecture.empty() && model.architecture != "unknown";
    if (!has_general_architecture) result.blockers.emplace_back("GGUF_GENERAL_ARCHITECTURE_MISSING");

    for (const auto& tensor : model.tensors) {
        bool attention = false, recurrent = false, expert = false, multimodal = false;
        std::string role = classify_tensor(tensor.name, attention, recurrent, expert, multimodal);
        uint64_t layer_index = UINT64_MAX;
        const bool is_layer_tensor = parse_layer_index(tensor.name, layer_index);
        if (is_layer_tensor) layers.insert(layer_index);
        if (role.empty()) continue;

        ++result.recognized_tensor_count;
        if (attention) { ++result.attention_tensor_count; result.has_attention = true; }
        if (recurrent) { ++result.recurrent_tensor_count; result.has_recurrent_path = true; }
        if (expert) { ++result.expert_tensor_count; result.has_experts = true; }
        if (multimodal) { ++result.multimodal_tensor_count; result.has_multimodal_components = true; }
        if (role == "token_embedding") result.has_embedding = true;
        if (role == "output_projection") result.has_output_projection = true;
        if (result.evidence.size() < maximum_evidence)
            result.evidence.push_back({tensor.name, std::move(role),
                                       is_layer_tensor ? layer_index : UINT64_MAX});
    }

    result.layer_count = layers.size();
    if (!layers.empty()) {
        uint64_t expected = 0;
        for (const uint64_t layer : layers) {
            if (layer != expected) {
                result.blockers.emplace_back("LAYER_INDEX_GAP_OR_NONZERO_START");
                break;
            }
            ++expected;
        }
    }
    if (result.recognized_tensor_count == 0)
        result.blockers.emplace_back("NO_KNOWN_TENSOR_ROLE_PATTERNS");
    if (result.has_multimodal_components)
        result.candidate_family = ModelStructureFamily::Multimodal;
    else if (result.has_attention && result.has_recurrent_path)
        result.candidate_family = ModelStructureFamily::HybridAttentionRecurrent;
    else if (result.has_experts)
        result.candidate_family = ModelStructureFamily::MixtureOfExperts;
    else if (result.has_attention)
        result.candidate_family = ModelStructureFamily::DenseTransformer;
    else if (result.has_recurrent_path)
        result.candidate_family = ModelStructureFamily::Recurrent;

    // A deterministic evidence-strength indicator, never a probability and never
    // sufficient to select executable semantics.
    uint32_t score = 0;
    if (has_general_architecture) score += 10;
    if (result.has_embedding) score += 15;
    if (result.has_output_projection) score += 10;
    if (result.has_attention) score += 20;
    if (result.has_recurrent_path) score += 20;
    if (result.has_experts) score += 15;
    if (result.layer_count > 0) score += 10;
    result.evidence_score = std::min<uint32_t>(score, 100);
    return result;
}

} // namespace ai_chat::gpu_engine
