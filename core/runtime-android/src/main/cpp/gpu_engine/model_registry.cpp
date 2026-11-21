#include "model_registry.h"
#include "tokenizer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace ai_chat::gpu_engine {


namespace {

bool metadata_u64(const GgufFileInfo& model, const std::string& key, uint64_t& value) {
    const auto found = model.metadata.find(key);
    if (found == model.metadata.end()) return false;
    if (const auto* unsigned_value = std::get_if<uint64_t>(&found->second)) {
        value = *unsigned_value;
        return true;
    }
    if (const auto* signed_value = std::get_if<int64_t>(&found->second)) {
        if (*signed_value <= 0) return false;
        value = static_cast<uint64_t>(*signed_value);
        return true;
    }
    return false;
}

bool metadata_f64(const GgufFileInfo& model, const std::string& key, double& value) {
    const auto found = model.metadata.find(key);
    if (found == model.metadata.end()) return false;
    if (const auto* floating = std::get_if<double>(&found->second)) {
        if (!std::isfinite(*floating)) return false;
        value = *floating;
        return true;
    }
    if (const auto* unsigned_value = std::get_if<uint64_t>(&found->second)) {
        value = static_cast<double>(*unsigned_value);
        return true;
    }
    if (const auto* signed_value = std::get_if<int64_t>(&found->second)) {
        value = static_cast<double>(*signed_value);
        return true;
    }
    return false;
}

class DenseTransformerArchitectureAdapter final : public ArchitectureAdapter {
public:
    DenseTransformerArchitectureAdapter(std::string architecture_name, bool uses_qk_norm,
                                        bool uses_qkv_bias = false)
        : architecture_name_(std::move(architecture_name)), uses_qk_norm_(uses_qk_norm),
          uses_qkv_bias_(uses_qkv_bias) {
        descriptor_.name = architecture_name_;
        descriptor_.family = ArchitectureFamily::DenseTransformer;
        descriptor_.required_operators = {"Embedding", "MatMul", "RMSNorm", "RoPE",
                                          "Attention", "SiLU", "Multiply", "Add", "Softmax"};
        descriptor_.uses_attention = true;
        descriptor_.uses_kv_cache = true;
        descriptor_.supports_dynamic_shapes = true;
        descriptor_.validation = SupportState::Unverified;
    }

    const ArchitectureDescriptor& descriptor() const noexcept override { return descriptor_; }

    Status build_execution_plan(const GgufFileInfo& model,
                                ModelExecutionPlan& plan) const override {
        plan = ModelExecutionPlan{};
        Status status = validate_metadata(model);
        if (!status.ok()) return status;
        uint64_t layers = 0, embedding = 0, heads = 0, kv_heads = 0, feed_forward = 0;
        const std::string metadata_prefix = architecture_name_ + ".";
        if (!metadata_u64(model, metadata_prefix + "block_count", layers) ||
            !metadata_u64(model, metadata_prefix + "embedding_length", embedding) ||
            !metadata_u64(model, metadata_prefix + "attention.head_count", heads) ||
            !metadata_u64(model, metadata_prefix + "feed_forward_length", feed_forward))
            return {StatusCode::InvalidArgument, "Llama execution plan metadata is incomplete"};
        if (!metadata_u64(model, metadata_prefix + "attention.head_count_kv", kv_heads)) kv_heads = heads;

        std::unordered_map<std::string, const GgufTensorInfo*> tensors;
        for (const auto& tensor : model.tensors) tensors.emplace(tensor.name, &tensor);
        const auto has_tensor = [&tensors](const std::string& name) {
            return tensors.find(name) != tensors.end();
        };
        const auto expect_shape = [&tensors, &plan](const std::string& name,
                                                    const std::vector<uint64_t>& expected) {
            const auto found = tensors.find(name);
            if (found == tensors.end()) {
                plan.blockers.emplace_back("MISSING_TENSOR:" + name);
                return;
            }
            if (found->second->dimensions != expected)
                plan.blockers.emplace_back("TENSOR_SHAPE_MISMATCH:" + name);
            if (!found->second->byte_size_known || found->second->byte_size == 0)
                plan.blockers.emplace_back("TENSOR_STORAGE_SIZE_UNKNOWN:" + name);
        };
        const std::string embedding_weight = "token_embd.weight";
        const std::string output_norm = "output_norm.weight";
        if (model.tokenizer.tokens.empty()) {
            plan.blockers.emplace_back("TOKENIZER_VOCABULARY_REQUIRED_FOR_EMBEDDING_SHAPE");
        } else {
            expect_shape(embedding_weight, {embedding, model.tokenizer.tokens.size()});
        }
        expect_shape(output_norm, {embedding});
        const uint64_t head_dimension = embedding / heads;
        const uint64_t kv_dimension = head_dimension * kv_heads;
        bool tied_output = false;
        if (!has_tensor("output.weight")) {
            const auto tie = model.metadata.find(metadata_prefix + "tie_word_embeddings");
            const bool* tied = tie == model.metadata.end() ? nullptr : std::get_if<bool>(&tie->second);
            if (!tied || !*tied)
                plan.blockers.emplace_back("OUTPUT_WEIGHT_MISSING_WITHOUT_EXPLICIT_TIED_EMBEDDINGS_METADATA");
            else
                tied_output = true;
        }
        if (!tied_output && !model.tokenizer.tokens.empty())
            expect_shape("output.weight", {embedding, model.tokenizer.tokens.size()});
        for (uint64_t layer = 0; layer < layers; ++layer) {
            const std::string prefix = "blk." + std::to_string(layer) + ".";
            expect_shape(prefix + "attn_norm.weight", {embedding});
            expect_shape(prefix + "attn_q.weight", {embedding, embedding});
            expect_shape(prefix + "attn_k.weight", {embedding, kv_dimension});
            if (uses_qk_norm_) {
                expect_shape(prefix + "attn_q_norm.weight", {head_dimension});
                expect_shape(prefix + "attn_k_norm.weight", {head_dimension});
            }
            expect_shape(prefix + "attn_v.weight", {embedding, kv_dimension});
            if (uses_qkv_bias_) {
                expect_shape(prefix + "attn_q.bias", {embedding});
                expect_shape(prefix + "attn_k.bias", {kv_dimension});
                expect_shape(prefix + "attn_v.bias", {kv_dimension});
                expect_shape(prefix + "attn_output.bias", {embedding});
                const std::pair<const char*, uint64_t> bias_layouts[] = {
                    {"attn_q.bias", embedding}, {"attn_k.bias", kv_dimension},
                    {"attn_v.bias", kv_dimension}, {"attn_output.bias", embedding}
                };
                for (const auto& bias : bias_layouts) {
                    const auto found = tensors.find(prefix + bias.first);
                    if (found != tensors.end() &&
                        (found->second->type.name != "F32" ||
                         bias.second > std::numeric_limits<uint64_t>::max() / sizeof(float) ||
                         found->second->byte_size != bias.second * sizeof(float))) {
                        plan.blockers.emplace_back("UNSUPPORTED_QWEN2_BIAS_STORAGE_OR_DTYPE:" + prefix + bias.first);
                    }
                }
            }
            expect_shape(prefix + "attn_output.weight", {embedding, embedding});
            expect_shape(prefix + "ffn_norm.weight", {embedding});
            expect_shape(prefix + "ffn_gate.weight", {embedding, feed_forward});
            expect_shape(prefix + "ffn_up.weight", {embedding, feed_forward});
            expect_shape(prefix + "ffn_down.weight", {feed_forward, embedding});
        }
        if (!plan.blockers.empty()) return {StatusCode::Unsupported, "Llama GGUF is missing tensors required by the dense execution plan"};

        plan.architecture = descriptor_.name;
        plan.layer_count = layers;
        plan.embedding_length = embedding;
        plan.attention_heads = heads;
        plan.kv_heads = kv_heads;
        plan.feed_forward_length = feed_forward;
        plan.vocabulary_size = model.tokenizer.tokens.size();
        plan.head_dimension = head_dimension;
        plan.rotary_dimensions = head_dimension;
        metadata_u64(model, metadata_prefix + "rope.dimension_count", plan.rotary_dimensions);
        if (!metadata_f64(model, metadata_prefix + "attention.layer_norm_rms_epsilon", plan.rms_norm_epsilon))
            plan.rms_norm_epsilon = 0.00001;
        if (!metadata_f64(model, metadata_prefix + "rope.freq_base", plan.rope_theta))
            plan.rope_theta = 10000.0;
        if (plan.rotary_dimensions == 0 || plan.rotary_dimensions > head_dimension ||
            (plan.rotary_dimensions & 1u) != 0)
            plan.blockers.emplace_back("INVALID_ROTARY_DIMENSION");
        plan.external_inputs = {"token_ids"};
        auto add = [&plan, head_dimension, kv_heads](std::string id, std::string op,
                           std::vector<std::string> inputs,
                           std::vector<std::string> weights,
                           std::string output, std::vector<uint64_t> shape = {}) {
            ModelPlanOperation operation{std::move(id), std::move(op), std::move(inputs),
                                         std::move(weights), std::move(output), std::move(shape), {}, {}, {}, {}, false};
            if (operation.operator_name == "RMSNorm")
                operation.float_attributes["epsilon"] = plan.rms_norm_epsilon;
            if (operation.operator_name == "RoPE") {
                operation.float_attributes["theta"] = plan.rope_theta;
                operation.integer_attributes["rotary_dimensions"] = plan.rotary_dimensions;
                operation.bool_attributes["neox_style"] = plan.rope_neox_style;
            }
            if (operation.operator_name == "Attention") {
                operation.float_attributes["scale"] = 1.0 / std::sqrt(static_cast<double>(head_dimension));
                operation.integer_attributes["kv_heads"] = kv_heads;
                operation.bool_attributes["causal"] = true;
            }
            if (operation.operator_name == "Embedding")
                operation.integer_attributes["vocabulary_size"] = plan.vocabulary_size;
            if (operation.operator_name == "KVCacheAppend")
                operation.side_effect = true;
            plan.operations.push_back(std::move(operation));
        };
        add("embedding", "Embedding", {"token_ids"}, {embedding_weight},
            "hidden", {0, embedding});
        std::string hidden_name = "hidden";
        for (uint64_t layer = 0; layer < layers; ++layer) {
            const std::string p = "blk." + std::to_string(layer) + ".";
            const std::string x = "layer." + std::to_string(layer) + ".";
            add(x + "attn_norm", "RMSNorm", {hidden_name}, {p + "attn_norm.weight"}, x + "normalized", {0, embedding});
            if (uses_qkv_bias_) {
                add(x + "q_proj", "MatMul", {x + "normalized"}, {p + "attn_q.weight"}, x + "q_linear", {0, heads, head_dimension});
                add(x + "q_bias.apply", "Add", {x + "q_linear"}, {p + "attn_q.bias"}, x + "q", {0, heads, head_dimension});
                add(x + "k_proj", "MatMul", {x + "normalized"}, {p + "attn_k.weight"}, x + "k_linear", {0, kv_heads, head_dimension});
                add(x + "k_bias.apply", "Add", {x + "k_linear"}, {p + "attn_k.bias"}, x + "k", {0, kv_heads, head_dimension});
                add(x + "v_proj", "MatMul", {x + "normalized"}, {p + "attn_v.weight"}, x + "v_linear", {0, kv_heads, head_dimension});
                add(x + "v_bias.apply", "Add", {x + "v_linear"}, {p + "attn_v.bias"}, x + "v", {0, kv_heads, head_dimension});
            } else {
                add(x + "q_proj", "MatMul", {x + "normalized"}, {p + "attn_q.weight"}, x + "q", {0, heads, head_dimension});
                add(x + "k_proj", "MatMul", {x + "normalized"}, {p + "attn_k.weight"}, x + "k", {0, kv_heads, head_dimension});
                add(x + "v_proj", "MatMul", {x + "normalized"}, {p + "attn_v.weight"}, x + "v", {0, kv_heads, head_dimension});
            }
            if (uses_qk_norm_) {
                add(x + "q_norm", "RMSNorm", {x + "q"}, {p + "attn_q_norm.weight"}, x + "q_normalized", {0, heads, head_dimension});
                add(x + "k_norm", "RMSNorm", {x + "k"}, {p + "attn_k_norm.weight"}, x + "k_normalized", {0, kv_heads, head_dimension});
            }
            add(x + "q_rope", "RoPE", {uses_qk_norm_ ? x + "q_normalized" : x + "q"}, {}, x + "q_rotated", {0, heads, head_dimension});
            add(x + "k_rope", "RoPE", {uses_qk_norm_ ? x + "k_normalized" : x + "k"}, {}, x + "k_rotated", {0, kv_heads, head_dimension});
            add(x + "kv_cache_append", "KVCacheAppend", {x + "k_rotated", x + "v"}, {},
                x + "kv_cache_updated");
            plan.operations.back().integer_attributes["layer_index"] = layer;
            add(x + "attention", "Attention",
                {x + "q_rotated", x + "cache_key", x + "cache_value"}, {},
                x + "attention_output", {0, heads, head_dimension});
            plan.operations.back().integer_attributes["layer_index"] = layer;
            plan.operations.back().control_inputs.push_back(x + "kv_cache_updated");
            if (uses_qkv_bias_) {
                add(x + "attention_output", "MatMul", {x + "attention_output"},
                    {p + "attn_output.weight"}, x + "attention_projected_linear", {0, embedding});
                add(x + "attention_output_bias.apply", "Add", {x + "attention_projected_linear"},
                    {p + "attn_output.bias"}, x + "attention_projected", {0, embedding});
            } else {
                add(x + "attention_output", "MatMul", {x + "attention_output"},
                    {p + "attn_output.weight"}, x + "attention_projected", {0, embedding});
            }
            add(x + "attention_residual", "Add", {hidden_name, x + "attention_projected"}, {},
                x + "attention_residual", {0, embedding});
            add(x + "ffn_norm", "RMSNorm", {x + "attention_residual"},
                {p + "ffn_norm.weight"}, x + "ffn_normalized", {0, embedding});
            add(x + "ffn_gate", "MatMul", {x + "ffn_normalized"},
                {p + "ffn_gate.weight"}, x + "ffn_gate_values", {0, feed_forward});
            add(x + "ffn_up", "MatMul", {x + "ffn_normalized"},
                {p + "ffn_up.weight"}, x + "ffn_up_values", {0, feed_forward});
            add(x + "ffn_activation", "SiLU", {x + "ffn_gate_values"}, {},
                x + "ffn_gate_activated", {0, feed_forward});
            add(x + "ffn_multiply", "Multiply", {x + "ffn_gate_activated", x + "ffn_up_values"},
                {}, x + "ffn_product", {0, feed_forward});
            add(x + "ffn_down", "MatMul", {x + "ffn_product"},
                {p + "ffn_down.weight"}, x + "ffn_down_projected", {0, embedding});
            add(x + "ffn_residual", "Add", {x + "attention_residual", x + "ffn_down_projected"},
                {}, x + "hidden", {0, embedding});
            hidden_name = x + "hidden";
        }
        add("final_norm", "RMSNorm", {hidden_name}, {output_norm}, "final_hidden", {0, embedding});
        add("logits", "MatMul", {"final_hidden"},
            {tied_output ? embedding_weight : "output.weight"}, "logits", {0, model.tokenizer.tokens.size()});
        plan.retained_outputs = {"logits"};
        plan.structurally_validated = true;
        plan.blockers.emplace_back("TOKENIZER_EXECUTION_NOT_VALIDATED");
        plan.blockers.emplace_back("OPERATOR_KERNEL_COVERAGE_NOT_VALIDATED");
        plan.blockers.emplace_back("MODEL_GRAPH_EXECUTION_NOT_VALIDATED");
        plan.blockers.emplace_back("ROPE_STYLE_NUMERICAL_VALIDATION_PENDING");
        plan.gpu_ready = false;
        return Status::success();
    }

    Status validate_metadata(const GgufFileInfo& model) const override {
        if (model.architecture != descriptor_.name)
            return {StatusCode::InvalidArgument, "GGUF architecture does not match the dense transformer adapter"};
        const std::string metadata_prefix = architecture_name_ + ".";
        uint64_t layers = 0, embedding = 0, heads = 0, kv_heads = 0, feed_forward = 0;
        if (!metadata_u64(model, metadata_prefix + "block_count", layers) ||
            !metadata_u64(model, metadata_prefix + "embedding_length", embedding) ||
            !metadata_u64(model, metadata_prefix + "attention.head_count", heads) ||
            !metadata_u64(model, metadata_prefix + "feed_forward_length", feed_forward)) {
            return {StatusCode::InvalidArgument, "Dense transformer GGUF is missing required layer, embedding, head or feed-forward metadata"};
        }
        if (!metadata_u64(model, metadata_prefix + "attention.head_count_kv", kv_heads)) kv_heads = heads;
        if (layers == 0 || embedding == 0 || heads == 0 || kv_heads == 0 || feed_forward == 0 ||
            kv_heads > heads || embedding % heads != 0) {
            return {StatusCode::InvalidArgument, "Dense transformer GGUF contains inconsistent model dimensions"};
        }
        return Status::success();
    }

private:
    std::string architecture_name_;
    bool uses_qk_norm_ = false;
    bool uses_qkv_bias_ = false;
    ArchitectureDescriptor descriptor_;
};

} // namespace

Status register_builtin_architecture_adapters(ArchitectureRegistry& registry) {
    Status status = registry.register_adapter(
        std::make_shared<DenseTransformerArchitectureAdapter>("llama", false));
    if (!status.ok()) return status;
    status = registry.register_adapter(
        std::make_shared<DenseTransformerArchitectureAdapter>("qwen2", false, true));
    if (!status.ok()) return status;
    return registry.register_adapter(
        std::make_shared<DenseTransformerArchitectureAdapter>("qwen3", true));
}

Status QuantizationRegistry::register_format(QuantizationFormat format) {
    if (format.name.empty() || format.block_elements == 0 || format.bytes_per_block == 0) {
        return {StatusCode::InvalidArgument, "Quantization format requires a name and valid block layout"};
    }
    if (formats_.find(format.file_type_id) != formats_.end()) {
        return {StatusCode::InvalidArgument, "Quantization file type id is already registered"};
    }
    formats_.emplace(format.file_type_id, std::move(format));
    return Status::success();
}

const QuantizationFormat* QuantizationRegistry::find(uint32_t file_type_id) const noexcept {
    const auto found = formats_.find(file_type_id);
    return found == formats_.end() ? nullptr : &found->second;
}

bool QuantizationRegistry::can_decode(uint32_t file_type_id) const noexcept {
    const auto* format = find(file_type_id);
    return format && format->decoder_state == SupportState::Validated;
}

bool QuantizationRegistry::can_execute_on_gpu(uint32_t file_type_id) const noexcept {
    const auto* format = find(file_type_id);
    if (!format || format->decoder_state != SupportState::Validated) return false;
    if (format->backend_gpu_kernel_states.empty())
        return format->gpu_kernel_state == SupportState::Validated;
    for (const auto& state : format->backend_gpu_kernel_states)
        if (state.second == SupportState::Validated) return true;
    return false;
}

bool QuantizationRegistry::can_execute_on_gpu(uint32_t file_type_id, BackendKind backend) const noexcept {
    const auto* format = find(file_type_id);
    if (!format || format->decoder_state != SupportState::Validated) return false;
    const auto state = format->backend_gpu_kernel_states.find(backend);
    return state != format->backend_gpu_kernel_states.end() &&
           state->second == SupportState::Validated;
}

std::vector<QuantizationFormat> QuantizationRegistry::formats() const {
    std::vector<QuantizationFormat> result;
    result.reserve(formats_.size());
    for (const auto& entry : formats_) result.push_back(entry.second);
    std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
        return a.file_type_id < b.file_type_id;
    });
    return result;
}

Status ArchitectureRegistry::register_adapter(std::shared_ptr<ArchitectureAdapter> adapter) {
    if (!adapter) return {StatusCode::InvalidArgument, "Cannot register a null architecture adapter"};
    const ArchitectureDescriptor& descriptor = adapter->descriptor();
    if (descriptor.name.empty() || descriptor.family == ArchitectureFamily::Unknown) {
        return {StatusCode::InvalidArgument, "Architecture adapter requires a name and known family"};
    }
    if (adapters_.find(descriptor.name) != adapters_.end()) {
        return {StatusCode::InvalidArgument, "Architecture adapter is already registered"};
    }
    adapters_.emplace(descriptor.name, std::move(adapter));
    return Status::success();
}

std::shared_ptr<const ArchitectureAdapter> ArchitectureRegistry::find(
        const std::string& architecture) const {
    const auto found = adapters_.find(architecture);
    return found == adapters_.end() ? nullptr : found->second;
}

std::vector<std::string> ArchitectureRegistry::registered_architectures() const {
    std::vector<std::string> result;
    result.reserve(adapters_.size());
    for (const auto& entry : adapters_) result.push_back(entry.first);
    std::sort(result.begin(), result.end());
    return result;
}


Status GgufModelLoader::inspect(const std::string& path,
                                const ArchitectureRegistry& architectures,
                                const QuantizationRegistry& quantizations) {
    unload();
    GgufReader candidate;
    Status status = GgufReader::open(path, candidate);
    if (!status.ok()) return status;
    ModelPreflightReport report = evaluate_model_preflight(candidate.info(), architectures, quantizations);
    reader_ = std::move(candidate);
    preflight_ = std::move(report);
    loaded_ = true;
    return Status::success();
}

void GgufModelLoader::unload() noexcept {
    reader_ = GgufReader{};
    preflight_ = ModelPreflightReport{};
    loaded_ = false;
}

const GgufReader* GgufModelLoader::reader() const noexcept {
    return loaded_ ? &reader_ : nullptr;
}

const ModelPreflightReport& GgufModelLoader::preflight_report() const noexcept {
    return preflight_;
}

bool GgufModelLoader::gpu_ready() const noexcept {
    return loaded_ && preflight_.gpu_execution_ready;
}

ModelPreflightReport evaluate_model_preflight(const GgufFileInfo& model,
                                              const ArchitectureRegistry& architectures,
                                              const QuantizationRegistry& quantizations) {
    ModelPreflightReport report;
    report.architecture = model.architecture;
    report.tensor_count = model.tensor_count;
    if (model.tokenizer.model == "unknown" || model.tokenizer.tokens.empty()) {
        report.blockers.emplace_back("GGUF_TOKENIZER_METADATA_MISSING");
    }
    report.all_tensor_storage_known = true;
    report.all_quantization_decoders_validated = true;
    report.all_quantization_gpu_kernels_validated = true;

    const auto adapter = architectures.find(model.architecture);
    report.architecture_known = static_cast<bool>(adapter);
    if (!adapter) {
        report.blockers.emplace_back("UNSUPPORTED_ARCHITECTURE:" + model.architecture);
    } else {
        const Status metadata_status = adapter->validate_metadata(model);
        if (!metadata_status.ok()) {
            report.blockers.emplace_back("ARCHITECTURE_METADATA_INVALID:" + metadata_status.message);
        }
        if (adapter->descriptor().validation != SupportState::Validated) {
            report.blockers.emplace_back("UNVERIFIED_ARCHITECTURE_ADAPTER:" + model.architecture);
        }
    }

    for (const GgufTensorInfo& tensor : model.tensors) {
        if (!tensor.byte_size_known) {
            report.all_tensor_storage_known = false;
            ++report.unknown_storage_tensors;
            report.blockers.emplace_back("UNKNOWN_TENSOR_STORAGE:" + tensor.name);
            continue;
        }
        if (tensor.byte_size > std::numeric_limits<uint64_t>::max() - report.known_weight_bytes) {
            report.all_tensor_storage_known = false;
            report.blockers.emplace_back("WEIGHT_SIZE_OVERFLOW:" + tensor.name);
        } else {
            report.known_weight_bytes += tensor.byte_size;
        }

        if (tensor.type.quantized) {
            if (std::find(report.quantization_formats.begin(), report.quantization_formats.end(),
                          tensor.type.name) == report.quantization_formats.end()) {
                report.quantization_formats.push_back(tensor.type.name);
            }
            const QuantizationFormat* format = quantizations.find(tensor.type_id);
            if (!format || format->decoder_state != SupportState::Validated) {
                report.all_quantization_decoders_validated = false;
                report.all_quantization_gpu_kernels_validated = false;
                ++report.unverified_quantized_tensors;
                report.blockers.emplace_back("UNVERIFIED_QUANT_DECODER:" + tensor.type.name);
            } else if (format->gpu_kernel_state != SupportState::Validated) {
                report.all_quantization_gpu_kernels_validated = false;
                report.blockers.emplace_back("UNVERIFIED_GPU_QUANT_KERNEL:" + tensor.type.name);
            }
        }
    }

    // Tokenizer algorithms are independently registered and fail closed until their
    // reference parity suites are complete for the exact GGUF profile.
    TokenizerRegistry tokenizer_registry;
    const Status tokenizer_registration = register_builtin_tokenizer_adapters(tokenizer_registry);
    if (!tokenizer_registration.ok()) {
        report.blockers.emplace_back("TOKENIZER_ADAPTER_REGISTRY_FAILED");
    } else {
        const TokenizerPreflightReport tokenizer_report =
            evaluate_tokenizer_preflight(model.tokenizer, tokenizer_registry);
        report.tokenizer_metadata_valid = tokenizer_report.metadata_valid;
        report.tokenizer_adapter_registered = tokenizer_report.adapter_registered;
        report.tokenizer_adapter_validated = tokenizer_report.adapter_validated;
        for (const std::string& blocker : tokenizer_report.blockers)
            report.blockers.emplace_back("TOKENIZER:" + blocker);
    }

    // Preflight is necessary but not sufficient: model-specific graph construction,
    // operator/kernel coverage and numerical validation have not been completed here.
    report.blockers.emplace_back("SAMPLER_MODEL_INTEGRATION_PENDING");
    report.blockers.emplace_back("MODEL_GRAPH_AND_KERNEL_VALIDATION_PENDING");
    report.gpu_execution_ready = false;
    return report;
}

} // namespace ai_chat::gpu_engine
