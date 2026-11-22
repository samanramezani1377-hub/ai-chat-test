#include "model_registry.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>

namespace ai_chat::gpu_engine {
namespace {

bool metadata_u64_value(const GgufFileInfo& model, const std::string& key, uint64_t& value) {
    const auto found = model.metadata.find(key);
    if (found == model.metadata.end()) return false;
    if (const auto* item = std::get_if<uint64_t>(&found->second)) { value = *item; return true; }
    if (const auto* item = std::get_if<int64_t>(&found->second)) {
        if (*item < 0) return false;
        value = static_cast<uint64_t>(*item);
        return true;
    }
    return false;
}

bool metadata_f64_value(const GgufFileInfo& model, const std::string& key, double& value) {
    const auto found = model.metadata.find(key);
    if (found == model.metadata.end()) return false;
    if (const auto* item = std::get_if<double>(&found->second)) {
        if (!std::isfinite(*item)) return false;
        value = *item;
        return true;
    }
    if (const auto* item = std::get_if<uint64_t>(&found->second)) {
        value = static_cast<double>(*item);
        return true;
    }
    if (const auto* item = std::get_if<int64_t>(&found->second)) {
        value = static_cast<double>(*item);
        return true;
    }
    return false;
}

bool metadata_u64_array(const GgufFileInfo& model, const std::string& key,
                        std::vector<uint64_t>& values) {
    const auto found = model.metadata.find(key);
    if (found == model.metadata.end()) return false;
    const auto* array = std::get_if<GgufMetadataArray>(&found->second);
    if (!array) return false;
    if (const auto* items = std::get_if<std::vector<uint64_t>>(&array->values)) {
        values = *items;
        return true;
    }
    if (const auto* items = std::get_if<std::vector<int64_t>>(&array->values)) {
        values.clear();
        values.reserve(items->size());
        for (const auto item : *items) {
            if (item < 0) return false;
            values.push_back(static_cast<uint64_t>(item));
        }
        return true;
    }
    if (const auto* items = std::get_if<std::vector<bool>>(&array->values)) {
        values.clear();
        values.reserve(items->size());
        for (const bool item : *items) values.push_back(item ? 1u : 0u);
        return true;
    }
    return false;
}

bool metadata_u64_or_uniform_array(const GgufFileInfo& model, const std::string& key,
                                    uint64_t& value) {
    if (metadata_u64_value(model, key, value)) return true;
    std::vector<uint64_t> values;
    if (!metadata_u64_array(model, key, values) || values.empty()) return false;
    if (!std::all_of(values.begin(), values.end(),
                     [&values](uint64_t item) { return item == values.front(); }))
        return false;
    value = values.front();
    return true;
}

struct Qwen35HybridConfig {
    uint64_t layers = 0;
    uint64_t embedding = 0;
    uint64_t heads = 0;
    uint64_t kv_heads = 0;
    uint64_t feed_forward = 0;
    uint64_t conv_kernel = 0;
    uint64_t ssm_inner = 0;
    uint64_t ssm_state = 0;
    uint64_t ssm_dt_rank = 0;
    uint64_t ssm_groups = 0;
    uint64_t head_dimension = 0;
    uint64_t key_dimension = 0;
    uint64_t conv_dimension = 0;
    uint64_t recurrent_state_elements = 0;
    uint64_t vocabulary_size = 0;
    double rms_epsilon = 0.00001;
    double rope_theta = 10000.0;
    std::vector<uint64_t> rope_sections;
    std::vector<uint64_t> recurrent_layers;
};

bool load_qwen35_hyperparameters(const GgufFileInfo& model, Qwen35HybridConfig& out,
                                 std::string& error) {
    const std::string prefix = "qwen35.";
    if (!metadata_u64_value(model, prefix + "block_count", out.layers) ||
        !metadata_u64_value(model, prefix + "embedding_length", out.embedding) ||
        !metadata_u64_or_uniform_array(model, prefix + "attention.head_count", out.heads) ||
        !metadata_u64_or_uniform_array(model, prefix + "feed_forward_length", out.feed_forward) ||
        !metadata_u64_value(model, prefix + "ssm.conv_kernel", out.conv_kernel) ||
        !metadata_u64_value(model, prefix + "ssm.inner_size", out.ssm_inner) ||
        !metadata_u64_value(model, prefix + "ssm.state_size", out.ssm_state) ||
        !metadata_u64_value(model, prefix + "ssm.time_step_rank", out.ssm_dt_rank) ||
        !metadata_u64_value(model, prefix + "ssm.group_count", out.ssm_groups)) {
        error = "Qwen3.5 GGUF is missing required attention or gated-delta-net metadata";
        return false;
    }
    if (!metadata_u64_or_uniform_array(model, prefix + "attention.head_count_kv", out.kv_heads))
        out.kv_heads = out.heads;
    if (out.layers == 0 || out.layers > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
        out.embedding == 0 || out.embedding > UINT64_MAX / 2 || out.heads == 0 ||
        out.kv_heads == 0 || out.feed_forward == 0 || out.conv_kernel == 0 ||
        out.ssm_inner == 0 || out.ssm_state == 0 || out.ssm_dt_rank == 0 ||
        out.ssm_groups == 0 || out.conv_kernel < 2 || out.kv_heads > out.heads ||
        out.embedding % out.heads != 0 || out.ssm_inner % out.ssm_dt_rank != 0 ||
        out.ssm_state > UINT64_MAX / out.ssm_groups) {
        error = "Qwen3.5 GGUF contains invalid or inconsistent hybrid dimensions";
        return false;
    }
    out.head_dimension = out.embedding / out.heads;
    out.key_dimension = out.ssm_state * out.ssm_groups;
    if (out.key_dimension > UINT64_MAX / 2 ||
        out.ssm_inner > (UINT64_MAX - 2 * out.key_dimension) / 2 ||
        out.ssm_dt_rank > UINT64_MAX / 2 ||
        out.head_dimension > UINT64_MAX / out.kv_heads) {
        error = "Qwen3.5 gated-delta-net or attention dimensions overflow";
        return false;
    }
    out.conv_dimension = out.ssm_inner + 2 * out.key_dimension;
    const uint64_t value_head_dimension = out.ssm_inner / out.ssm_dt_rank;
    if (value_head_dimension != out.ssm_state) {
        error = "Qwen3.5 value-head dimension must equal the recurrent key/state dimension";
        return false;
    }
    if (value_head_dimension > UINT64_MAX / value_head_dimension ||
        value_head_dimension * value_head_dimension > UINT64_MAX / out.ssm_dt_rank) {
        error = "Qwen3.5 recurrent state dimensions overflow";
        return false;
    }
    out.recurrent_state_elements = value_head_dimension * value_head_dimension * out.ssm_dt_rank;
    if (!metadata_u64_array(model, prefix + "rope.dimension_sections", out.rope_sections) ||
        out.rope_sections.size() != 4) {
        error = "Qwen3.5 requires four RoPE dimension sections in GGUF metadata";
        return false;
    }
    if (!metadata_u64_array(model, prefix + "attention.recurrent_layers", out.recurrent_layers)) {
        uint64_t full_attention_interval = 4;
        metadata_u64_value(model, prefix + "full_attention_interval", full_attention_interval);
        if (full_attention_interval == 0) {
            error = "Qwen3.5 full_attention_interval must be non-zero";
            return false;
        }
        out.recurrent_layers.resize(static_cast<size_t>(out.layers));
        for (uint64_t layer = 0; layer < out.layers; ++layer)
            out.recurrent_layers[static_cast<size_t>(layer)] =
                ((layer + 1) % full_attention_interval) != 0 ? 1u : 0u;
    }
    if (out.recurrent_layers.size() != out.layers) {
        error = "Qwen3.5 recurrent_layers metadata length does not match block_count";
        return false;
    }
    if (!metadata_f64_value(model, prefix + "attention.layer_norm_rms_epsilon", out.rms_epsilon))
        out.rms_epsilon = 0.00001;
    if (!metadata_f64_value(model, prefix + "rope.freq_base", out.rope_theta))
        out.rope_theta = 10000.0;
    if (!std::isfinite(out.rms_epsilon) || out.rms_epsilon <= 0.0 ||
        !std::isfinite(out.rope_theta) || out.rope_theta <= 0.0) {
        error = "Qwen3.5 normalization or RoPE scalar metadata is invalid";
        return false;
    }
    return true;
}

std::string matmul_operator_for(const GgufTensorInfo& tensor) {
    const std::string& type = tensor.type.name;
    if (type == "Q8_0") return "MatMul_Q8_0";
    if (type == "Q4_0") return "MatMul_Q4_0";
    if (type == "Q4_1") return "MatMul_Q4_1";
    if (type == "Q5_0") return "MatMul_Q5_0";
    if (type == "Q5_1") return "MatMul_Q5_1";
    if (type == "F16") return "MatMul_F16";
    if (type == "BF16") return "MatMul_BF16";
    if (type == "Q6_K") return "MatMul_Q6_K";
    if (type == "Q4_K") return "MatMul_Q4_K";
    if (type == "Q5_K") return "MatMul_Q5_K";
    if (type == "Q2_K") return "MatMul_Q2_K";
    if (type == "Q3_K") return "MatMul_Q3_K";
    return "MatMul";
}

std::string embedding_operator_for(const GgufTensorInfo& tensor) {
    const std::string& type = tensor.type.name;
    if (type == "Q4_0") return "Embedding_Q4_0";
    if (type == "Q4_1") return "Embedding_Q4_1";
    if (type == "Q8_0") return "Embedding_Q8_0";
    if (type == "Q5_0") return "Embedding_Q5_0";
    if (type == "Q5_1") return "Embedding_Q5_1";
    if (type == "F16") return "Embedding_F16";
    if (type == "BF16") return "Embedding_BF16";
    if (type == "Q6_K") return "Embedding_Q6_K";
    return "Embedding";
}

class HybridGatedDeltaNetRecipe final : public ArchitectureAdapter {
public:
    HybridGatedDeltaNetRecipe() {
        descriptor_.name = "qwen35";
        descriptor_.family = ArchitectureFamily::Hybrid;
        descriptor_.version = "qwen35-hybrid-gdn-v1";
        descriptor_.required_operators = {
            "Embedding", "MatMul", "RMSNorm", "MultiAxisRoPE", "Attention",
            "KVCacheAppend", "Sigmoid", "Softplus", "SSMConv1D", "GdnL2Norm",
            "GatedDeltaNet", "Split", "SiLU", "Multiply", "Add"
        };
        descriptor_.uses_attention = true;
        descriptor_.uses_kv_cache = true;
        descriptor_.uses_recurrent_state = true;
        descriptor_.supports_dynamic_shapes = true;
        descriptor_.validation = SupportState::Unverified;
    }

    const ArchitectureDescriptor& descriptor() const noexcept override { return descriptor_; }

    Status validate_metadata(const GgufFileInfo& model) const override {
        if (model.architecture != descriptor_.name)
            return {StatusCode::InvalidArgument, "GGUF architecture does not match the hybrid gated-delta-net recipe"};
        Qwen35HybridConfig config;
        std::string error;
        if (!load_qwen35_hyperparameters(model, config, error))
            return {StatusCode::InvalidArgument, error};
        return Status::success();
    }

    Status build_execution_plan(const GgufFileInfo& model,
                                ModelExecutionPlan& plan) const override {
        plan = ModelExecutionPlan{};
        Qwen35HybridConfig config;
        std::string error;
        if (!load_qwen35_hyperparameters(model, config, error))
            return {StatusCode::InvalidArgument, error};

        std::unordered_map<std::string, const GgufTensorInfo*> tensors;
        tensors.reserve(model.tensors.size());
        for (const auto& tensor : model.tensors) tensors.emplace(tensor.name, &tensor);
        auto expect = [&](const std::string& name, const std::vector<uint64_t>& shape) {
            const auto found = tensors.find(name);
            if (found == tensors.end()) {
                plan.blockers.emplace_back("MISSING_TENSOR:" + name);
                return false;
            }
            if (found->second->dimensions != shape) {
                plan.blockers.emplace_back("TENSOR_SHAPE_MISMATCH:" + name);
                return false;
            }
            if (!found->second->byte_size_known || found->second->byte_size == 0) {
                plan.blockers.emplace_back("TENSOR_STORAGE_SIZE_UNKNOWN:" + name);
                return false;
            }
            return true;
        };
        const auto has = [&tensors](const std::string& name) {
            return tensors.find(name) != tensors.end();
        };

        if (model.tokenizer.tokens.empty()) plan.blockers.emplace_back("TOKENIZER_VOCABULARY_REQUIRED");
        else expect("token_embd.weight", {config.embedding, model.tokenizer.tokens.size()});
        expect("output_norm.weight", {config.embedding});
        const bool tied_output = !has("output.weight");
        if (!tied_output) expect("output.weight", {config.embedding, model.tokenizer.tokens.size()});

        for (uint64_t layer = 0; layer < config.layers; ++layer) {
            const std::string prefix = "blk." + std::to_string(layer) + ".";
            expect(prefix + "attn_norm.weight", {config.embedding});
            expect(prefix + "post_attention_norm.weight", {config.embedding});
            expect(prefix + "ffn_gate.weight", {config.embedding, config.feed_forward});
            expect(prefix + "ffn_up.weight", {config.embedding, config.feed_forward});
            expect(prefix + "ffn_down.weight", {config.feed_forward, config.embedding});

            if (config.recurrent_layers[static_cast<size_t>(layer)] != 0) {
                const uint64_t qkv_size = 2 * config.key_dimension + config.ssm_inner;
                const uint64_t qkvz_size = qkv_size + config.ssm_inner;
                if (has(prefix + "attn_qkv_gate.weight")) {
                    expect(prefix + "attn_qkv_gate.weight", {config.embedding, qkvz_size});
                } else {
                    expect(prefix + "attn_qkv.weight", {config.embedding, qkv_size});
                    expect(prefix + "attn_gate.weight", {config.embedding, config.ssm_inner});
                }
                expect(prefix + "ssm_conv1d.weight", {config.conv_kernel, config.conv_dimension});
                expect(prefix + "ssm_dt.bias", {config.ssm_dt_rank});
                expect(prefix + "ssm_a", {config.ssm_dt_rank});
                if (has(prefix + "ssm_beta_alpha.weight")) {
                    expect(prefix + "ssm_beta_alpha.weight",
                           {config.embedding, config.ssm_dt_rank * 2});
                } else {
                    expect(prefix + "ssm_beta.weight", {config.embedding, config.ssm_dt_rank});
                    expect(prefix + "ssm_alpha.weight", {config.embedding, config.ssm_dt_rank});
                }
                expect(prefix + "ssm_norm.weight", {config.ssm_inner / config.ssm_dt_rank});
                expect(prefix + "ssm_out.weight", {config.ssm_inner, config.embedding});
            } else {
                expect(prefix + "attn_q.weight", {config.embedding, config.embedding * 2});
                expect(prefix + "attn_k.weight", {config.embedding, config.head_dimension * config.kv_heads});
                expect(prefix + "attn_v.weight", {config.embedding, config.head_dimension * config.kv_heads});
                expect(prefix + "attn_output.weight", {config.embedding, config.embedding});
                expect(prefix + "attn_q_norm.weight", {config.head_dimension});
                expect(prefix + "attn_k_norm.weight", {config.head_dimension});
            }
        }
        if (!plan.blockers.empty())
            return {StatusCode::Unsupported, "Qwen3.5 hybrid tensor inventory does not match the declared recipe"};

        plan.architecture = descriptor_.name;
        plan.semantic_schema_id = "hybrid_gated_delta_net";
        plan.semantic_schema_version = "1";
        plan.layer_count = config.layers;
        plan.embedding_length = config.embedding;
        plan.attention_heads = config.heads;
        plan.kv_heads = config.kv_heads;
        plan.feed_forward_length = config.feed_forward;
        plan.vocabulary_size = model.tokenizer.tokens.size();
        config.vocabulary_size = plan.vocabulary_size;
        plan.head_dimension = config.head_dimension;
        plan.rotary_dimensions = config.head_dimension;
        plan.rms_norm_epsilon = config.rms_epsilon;
        plan.rope_theta = config.rope_theta;
        plan.rope_neox_style = true;
        plan.uses_recurrent_state = true;
        plan.uses_multi_axis_rope = true;
        plan.external_inputs = {"token_ids"};
        for (uint64_t layer = 0; layer < config.layers; ++layer) {
            if (config.recurrent_layers[static_cast<size_t>(layer)] == 0) continue;
            const std::string state_prefix = "layer." + std::to_string(layer) + ".";
            plan.external_inputs.push_back(state_prefix + "recurrent_state");
            plan.external_inputs.push_back(state_prefix + "conv_state");
        }
        plan.retained_outputs = {"logits"};

        auto add = [&](std::string id, std::string op, std::vector<std::string> inputs,
                       std::vector<std::string> weights, std::string output,
                       std::vector<uint64_t> shape = {}, bool side_effect = false) -> ModelPlanOperation& {
            ModelPlanOperation operation;
            operation.id = std::move(id);
            operation.operator_name = std::move(op);
            operation.inputs = std::move(inputs);
            operation.weight_tensors = std::move(weights);
            operation.output = std::move(output);
            operation.output_shape = std::move(shape);
            operation.side_effect = side_effect;
            if (operation.operator_name == "MatMul" && operation.weight_tensors.size() == 1) {
                const auto found = tensors.find(operation.weight_tensors.front());
                if (found != tensors.end()) operation.operator_name = matmul_operator_for(*found->second);
            }
            if (operation.operator_name == "Embedding" && operation.weight_tensors.size() == 1) {
                const auto found = tensors.find(operation.weight_tensors.front());
                if (found != tensors.end()) operation.operator_name = embedding_operator_for(*found->second);
            }
            if (operation.operator_name == "RMSNorm")
                operation.float_attributes["epsilon"] = config.rms_epsilon;
            if (operation.operator_name == "Attention") {
                operation.float_attributes["scale"] =
                    1.0 / std::sqrt(static_cast<double>(config.head_dimension));
                operation.integer_attributes["kv_heads"] = config.kv_heads;
                operation.bool_attributes["causal"] = true;
            }
            if (operation.operator_name == "Embedding")
                operation.integer_attributes["vocabulary_size"] = config.vocabulary_size;
            if (operation.operator_name == "KVCacheAppend") operation.side_effect = true;
            plan.operations.push_back(std::move(operation));
            return plan.operations.back();
        };

        add("embedding", "Embedding", {"token_ids"}, {"token_embd.weight"},
            "hidden", {0, config.embedding});
        std::string hidden = "hidden";
        for (uint64_t layer = 0; layer < config.layers; ++layer) {
            const std::string p = "blk." + std::to_string(layer) + ".";
            const std::string x = "layer." + std::to_string(layer) + ".";
            add(x + "attn_norm", "RMSNorm", {hidden}, {p + "attn_norm.weight"},
                x + "normalized", {0, config.embedding});

            std::string attention_output;
            if (config.recurrent_layers[static_cast<size_t>(layer)] != 0) {
                if (has(p + "attn_qkv_gate.weight")) {
                    add(x + "qkvz_projection", "MatMul", {x + "normalized"},
                        {p + "attn_qkv_gate.weight"}, x + "qkvz",
                        {0, 2 * config.key_dimension + 2 * config.ssm_inner});
                    add(x + "qkv_split", "Split", {x + "qkvz"}, {}, x + "qkv",
                        {0, 2 * config.key_dimension + config.ssm_inner});
                    plan.operations.back().integer_attributes["axis"] = 1;
                    plan.operations.back().integer_attributes["offset"] = 0;
                    plan.operations.back().integer_attributes["length"] = 2 * config.key_dimension + config.ssm_inner;
                    add(x + "z_split", "Split", {x + "qkvz"}, {}, x + "z",
                        {0, config.ssm_inner});
                    plan.operations.back().integer_attributes["axis"] = 1;
                    plan.operations.back().integer_attributes["offset"] = 2 * config.key_dimension + config.ssm_inner;
                    plan.operations.back().integer_attributes["length"] = config.ssm_inner;
                } else {
                    add(x + "qkv_projection", "MatMul", {x + "normalized"},
                        {p + "attn_qkv.weight"}, x + "qkv",
                        {0, 2 * config.key_dimension + config.ssm_inner});
                    add(x + "z_projection", "MatMul", {x + "normalized"},
                        {p + "attn_gate.weight"}, x + "z", {0, config.ssm_inner});
                }

                if (has(p + "ssm_beta_alpha.weight")) {
                    add(x + "beta_alpha_projection", "MatMul", {x + "normalized"},
                        {p + "ssm_beta_alpha.weight"}, x + "beta_alpha",
                        {0, config.ssm_dt_rank * 2});
                    add(x + "beta_split", "Split", {x + "beta_alpha"}, {}, x + "beta",
                        {0, config.ssm_dt_rank});
                    plan.operations.back().integer_attributes["axis"] = 1;
                    plan.operations.back().integer_attributes["offset"] = 0;
                    plan.operations.back().integer_attributes["length"] = config.ssm_dt_rank;
                    add(x + "alpha_split", "Split", {x + "beta_alpha"}, {}, x + "alpha",
                        {0, config.ssm_dt_rank});
                    plan.operations.back().integer_attributes["axis"] = 1;
                    plan.operations.back().integer_attributes["offset"] = config.ssm_dt_rank;
                    plan.operations.back().integer_attributes["length"] = config.ssm_dt_rank;
                } else {
                    add(x + "beta_projection", "MatMul", {x + "normalized"},
                        {p + "ssm_beta.weight"}, x + "beta", {0, config.ssm_dt_rank});
                    add(x + "alpha_projection", "MatMul", {x + "normalized"},
                        {p + "ssm_alpha.weight"}, x + "alpha", {0, config.ssm_dt_rank});
                }
                add(x + "beta_sigmoid", "Sigmoid", {x + "beta"}, {}, x + "beta_gated",
                    {0, config.ssm_dt_rank});
                add(x + "alpha_bias", "Add", {x + "alpha"}, {p + "ssm_dt.bias"},
                    x + "alpha_biased", {0, config.ssm_dt_rank});
                add(x + "alpha_softplus", "Softplus", {x + "alpha_biased"}, {},
                    x + "alpha_positive", {0, config.ssm_dt_rank});
                add(x + "delta_gate", "Multiply", {x + "alpha_positive"},
                    {p + "ssm_a"}, x + "delta_gate", {0, config.ssm_dt_rank});

                add(x + "ssm_conv", "SSMConv1D", {x + "qkv", x + "conv_state"},
                    {p + "ssm_conv1d.weight"}, x + "conv_output",
                    {0, config.conv_dimension});
                plan.operations.back().integer_attributes["layer_index"] = layer;
                plan.operations.back().integer_attributes["kernel_size"] = config.conv_kernel;
                add(x + "ssm_conv_silu", "SiLU", {x + "conv_output"}, {},
                    x + "conv_output_silu", {0, config.conv_dimension});
                add(x + "ssm_split_q", "Split", {x + "conv_output_silu"}, {}, x + "q",
                    {0, config.key_dimension});
                plan.operations.back().integer_attributes["axis"] = 1;
                plan.operations.back().integer_attributes["offset"] = 0;
                plan.operations.back().integer_attributes["length"] = config.key_dimension;
                add(x + "ssm_split_k", "Split", {x + "conv_output_silu"}, {}, x + "k",
                    {0, config.key_dimension});
                plan.operations.back().integer_attributes["axis"] = 1;
                plan.operations.back().integer_attributes["offset"] = config.key_dimension;
                plan.operations.back().integer_attributes["length"] = config.key_dimension;
                add(x + "ssm_split_v", "Split", {x + "conv_output_silu"}, {}, x + "v",
                    {0, config.ssm_inner});
                plan.operations.back().integer_attributes["axis"] = 1;
                plan.operations.back().integer_attributes["offset"] = 2 * config.key_dimension;
                plan.operations.back().integer_attributes["length"] = config.ssm_inner;
                add(x + "q_l2_norm", "GdnL2Norm", {x + "q"}, {}, x + "q_norm",
                    {0, config.key_dimension});
                plan.operations.back().integer_attributes["state_size"] = config.ssm_state;
                plan.operations.back().integer_attributes["group_count"] = config.ssm_groups;
                plan.operations.back().float_attributes["epsilon"] = config.rms_epsilon;
                add(x + "k_l2_norm", "GdnL2Norm", {x + "k"}, {}, x + "k_norm",
                    {0, config.key_dimension});
                plan.operations.back().integer_attributes["state_size"] = config.ssm_state;
                plan.operations.back().integer_attributes["group_count"] = config.ssm_groups;
                plan.operations.back().float_attributes["epsilon"] = config.rms_epsilon;
                add(x + "gated_delta_net", "GatedDeltaNet",
                    {x + "q_norm", x + "k_norm", x + "v", x + "delta_gate",
                     x + "beta_gated", x + "recurrent_state"},
                    {}, x + "recurrent_output", {0, config.ssm_inner});
                plan.operations.back().integer_attributes["layer_index"] = layer;
                plan.operations.back().integer_attributes["state_size"] = config.ssm_state;
                plan.operations.back().integer_attributes["key_heads"] = config.ssm_groups;
                plan.operations.back().integer_attributes["value_heads"] = config.ssm_dt_rank;
                add(x + "gdn_rms_norm", "RMSNorm",
                    {x + "recurrent_output"}, {p + "ssm_norm.weight"},
                    x + "gdn_rms_normalized", {0, config.ssm_inner});
                add(x + "gdn_gate_silu", "SiLU", {x + "z"}, {},
                    x + "gdn_gate_activated", {0, config.ssm_inner});
                add(x + "gdn_gate_multiply", "Multiply",
                    {x + "gdn_rms_normalized", x + "gdn_gate_activated"}, {},
                    x + "gdn_normalized", {0, config.ssm_inner});
                add(x + "recurrent_output_projection", "MatMul",
                    {x + "gdn_normalized"}, {p + "ssm_out.weight"},
                    x + "attention_output", {0, config.embedding});
            } else {
                add(x + "q_projection", "MatMul", {x + "normalized"},
                    {p + "attn_q.weight"}, x + "q_full",
                    {0, config.heads, config.head_dimension * 2});
                add(x + "q_split", "Split", {x + "q_full"}, {}, x + "q",
                    {0, config.heads, config.head_dimension});
                plan.operations.back().integer_attributes["axis"] = 2;
                plan.operations.back().integer_attributes["offset"] = 0;
                plan.operations.back().integer_attributes["length"] = config.head_dimension;
                add(x + "query_gate_split", "Split", {x + "q_full"}, {}, x + "query_gate",
                    {0, config.heads, config.head_dimension});
                plan.operations.back().integer_attributes["axis"] = 2;
                plan.operations.back().integer_attributes["offset"] = config.head_dimension;
                plan.operations.back().integer_attributes["length"] = config.head_dimension;
                add(x + "k_projection", "MatMul", {x + "normalized"},
                    {p + "attn_k.weight"}, x + "k",
                    {0, config.kv_heads, config.head_dimension});
                add(x + "v_projection", "MatMul", {x + "normalized"},
                    {p + "attn_v.weight"}, x + "v",
                    {0, config.kv_heads, config.head_dimension});
                add(x + "q_norm", "RMSNorm", {x + "q"}, {p + "attn_q_norm.weight"},
                    x + "q_normalized", {0, config.heads, config.head_dimension});
                add(x + "k_norm", "RMSNorm", {x + "k"}, {p + "attn_k_norm.weight"},
                    x + "k_normalized", {0, config.kv_heads, config.head_dimension});
                for (const std::string suffix : {"q", "k"}) {
                    auto& rope = add(x + suffix + "_mrope", "MultiAxisRoPE",
                        {x + (suffix == "q" ? "q_normalized" : "k_normalized")},
                        {}, x + suffix + "_rotated",
                        {0, suffix == "q" ? config.heads : config.kv_heads, config.head_dimension});
                    rope.float_attributes["theta"] = config.rope_theta;
                    rope.integer_attributes["rotary_dimensions"] = config.head_dimension;
                    rope.integer_array_attributes["sections"] = config.rope_sections;
                    // In text-only mode all M-RoPE axes share the same position;
                    // standard NeoX RoPE is therefore equivalent for this path.
                    rope.string_attributes["position_encoding"] = "qwen35_text_only_mrope_equivalent";
                }
                add(x + "kv_cache_append", "KVCacheAppend",
                    {x + "k_rotated", x + "v"}, {}, x + "kv_cache_updated", {}, true);
                plan.operations.back().integer_attributes["layer_index"] = layer;
                add(x + "attention", "Attention",
                    {x + "q_rotated", x + "cache_key", x + "cache_value"},
                    {}, x + "attention_context", {0, config.heads, config.head_dimension});
                plan.operations.back().integer_attributes["layer_index"] = layer;
                plan.operations.back().control_inputs.push_back(x + "kv_cache_updated");
                add(x + "attention_gate_sigmoid", "Sigmoid", {x + "query_gate"}, {},
                    x + "attention_gate", {0, config.heads, config.head_dimension});
                add(x + "attention_gated", "Multiply",
                    {x + "attention_context", x + "attention_gate"}, {},
                    x + "attention_gated", {0, config.heads, config.head_dimension});
                add(x + "attention_output_projection", "MatMul",
                    {x + "attention_gated"}, {p + "attn_output.weight"},
                    x + "attention_output", {0, config.embedding});
            }
            attention_output = x + "attention_output";
            add(x + "attention_residual", "Add",
                {hidden, attention_output}, {}, x + "attention_residual", {0, config.embedding});
            add(x + "post_attention_norm", "RMSNorm",
                {x + "attention_residual"}, {p + "post_attention_norm.weight"},
                x + "ffn_normalized", {0, config.embedding});
            add(x + "ffn_gate", "MatMul", {x + "ffn_normalized"},
                {p + "ffn_gate.weight"}, x + "ffn_gate_values", {0, config.feed_forward});
            add(x + "ffn_up", "MatMul", {x + "ffn_normalized"},
                {p + "ffn_up.weight"}, x + "ffn_up_values", {0, config.feed_forward});
            add(x + "ffn_silu", "SiLU", {x + "ffn_gate_values"}, {},
                x + "ffn_gate_activated", {0, config.feed_forward});
            add(x + "ffn_multiply", "Multiply",
                {x + "ffn_gate_activated", x + "ffn_up_values"}, {},
                x + "ffn_product", {0, config.feed_forward});
            add(x + "ffn_down", "MatMul", {x + "ffn_product"},
                {p + "ffn_down.weight"}, x + "ffn_output", {0, config.embedding});
            add(x + "ffn_residual", "Add",
                {x + "attention_residual", x + "ffn_output"}, {},
                x + "hidden", {0, config.embedding});
            hidden = x + "hidden";
        }
        add("final_norm", "RMSNorm", {hidden}, {"output_norm.weight"},
            "final_hidden", {0, config.embedding});
        add("logits", "MatMul", {"final_hidden"},
            {tied_output ? "token_embd.weight" : "output.weight"},
            "logits", {0, config.vocabulary_size});
        plan.structurally_validated = true;
        plan.blockers.emplace_back("HYBRID_GDN_GPU_KERNEL_VALIDATION_PENDING");
        plan.blockers.emplace_back("MULTI_AXIS_ROPE_NUMERICAL_PARITY_PENDING");
        plan.blockers.emplace_back("RECURRENT_STATE_LIFECYCLE_VALIDATION_PENDING");
        plan.gpu_ready = false;
        return Status::success();
    }

private:
    ArchitectureDescriptor descriptor_;
};

} // namespace

Status register_builtin_hybrid_architecture_recipes(ArchitectureRegistry& registry) {
    return registry.register_adapter(std::make_shared<HybridGatedDeltaNetRecipe>());
}

} // namespace ai_chat::gpu_engine
