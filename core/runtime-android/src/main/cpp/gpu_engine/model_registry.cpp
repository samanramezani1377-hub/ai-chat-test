#include "model_registry.h"
#include "tokenizer.h"

#include <algorithm>
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

class LlamaDenseArchitectureAdapter final : public ArchitectureAdapter {
public:
    LlamaDenseArchitectureAdapter() {
        descriptor_.name = "llama";
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
        if (!metadata_u64(model, "llama.block_count", layers) ||
            !metadata_u64(model, "llama.embedding_length", embedding) ||
            !metadata_u64(model, "llama.attention.head_count", heads) ||
            !metadata_u64(model, "llama.feed_forward_length", feed_forward))
            return {StatusCode::InvalidArgument, "Llama execution plan metadata is incomplete"};
        if (!metadata_u64(model, "llama.attention.head_count_kv", kv_heads)) kv_heads = heads;

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
            } else if (found->second->dimensions != expected) {
                plan.blockers.emplace_back("TENSOR_SHAPE_MISMATCH:" + name);
            }
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
        const bool tied_output = !has_tensor("output.weight");
        if (!tied_output && !model.tokenizer.tokens.empty())
            expect_shape("output.weight", {embedding, model.tokenizer.tokens.size()});
        for (uint64_t layer = 0; layer < layers; ++layer) {
            const std::string prefix = "blk." + std::to_string(layer) + ".";
            expect_shape(prefix + "attn_norm.weight", {embedding});
            expect_shape(prefix + "attn_q.weight", {embedding, embedding});
            expect_shape(prefix + "attn_k.weight", {embedding, kv_dimension});
            expect_shape(prefix + "attn_v.weight", {embedding, kv_dimension});
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
        auto add = [&plan](std::string id, std::string op,
                           std::vector<std::string> inputs,
                           std::vector<std::string> weights,
                           std::string output, std::vector<uint64_t> shape = {}) {
            plan.operations.push_back({std::move(id), std::move(op), std::move(inputs),
                                       std::move(weights), std::move(output), std::move(shape)});
        };
        add("embedding", "Embedding", {"token_ids"}, {embedding_weight},
            "hidden", {embedding});
        for (uint64_t layer = 0; layer < layers; ++layer) {
            const std::string p = "blk." + std::to_string(layer) + ".";
            const std::string x = "layer." + std::to_string(layer) + ".";
            add(x + "attn_norm", "RMSNorm", {"hidden"}, {p + "attn_norm.weight"}, x + "normalized");
            add(x + "q_proj", "MatMul", {x + "normalized"}, {p + "attn_q.weight"}, x + "q");
            add(x + "k_proj", "MatMul", {x + "normalized"}, {p + "attn_k.weight"}, x + "k");
            add(x + "v_proj", "MatMul", {x + "normalized"}, {p + "attn_v.weight"}, x + "v");
            add(x + "q_rope", "RoPE", {x + "q"}, {}, x + "q_rotated");
            add(x + "k_rope", "RoPE", {x + "k"}, {}, x + "k_rotated");
            add(x + "attention", "Attention", {x + "q_rotated", x + "k_rotated", x + "v", "kv_cache"},
                {}, x + "attention_output");
            add(x + "attention_output", "MatMul", {x + "attention_output"},
                {p + "attn_output.weight"}, x + "attention_projected");
            add(x + "attention_residual", "Add", {"hidden", x + "attention_projected"}, {},
                x + "attention_residual");
            add(x + "ffn_norm", "RMSNorm", {x + "attention_residual"},
                {p + "ffn_norm.weight"}, x + "ffn_normalized");
            add(x + "ffn_gate", "MatMul", {x + "ffn_normalized"},
                {p + "ffn_gate.weight"}, x + "ffn_gate_values");
            add(x + "ffn_up", "MatMul", {x + "ffn_normalized"},
                {p + "ffn_up.weight"}, x + "ffn_up_values");
            add(x + "ffn_activation", "SiLU", {x + "ffn_gate_values"}, {},
                x + "ffn_gate_activated");
            add(x + "ffn_multiply", "Multiply", {x + "ffn_gate_activated", x + "ffn_up_values"},
                {}, x + "ffn_product");
            add(x + "ffn_down", "MatMul", {x + "ffn_product"},
                {p + "ffn_down.weight"}, x + "ffn_down_projected");
            add(x + "ffn_residual", "Add", {x + "attention_residual", x + "ffn_down_projected"},
                {}, "hidden");
        }
        add("final_norm", "RMSNorm", {"hidden"}, {output_norm}, "final_hidden");
        add("logits", "MatMul", {"final_hidden"},
            {tied_output ? embedding_weight : "output.weight"}, "logits");
        plan.structurally_validated = true;
        plan.blockers.emplace_back("TOKENIZER_EXECUTION_NOT_VALIDATED");
        plan.blockers.emplace_back("OPERATOR_KERNEL_COVERAGE_NOT_VALIDATED");
        plan.blockers.emplace_back("MODEL_GRAPH_EXECUTION_NOT_VALIDATED");
        plan.gpu_ready = false;
        return Status::success();
    }

    Status validate_metadata(const GgufFileInfo& model) const override {
        if (model.architecture != descriptor_.name)
            return {StatusCode::InvalidArgument, "GGUF architecture does not match the llama adapter"};
        uint64_t layers = 0, embedding = 0, heads = 0, kv_heads = 0, feed_forward = 0;
        if (!metadata_u64(model, "llama.block_count", layers) ||
            !metadata_u64(model, "llama.embedding_length", embedding) ||
            !metadata_u64(model, "llama.attention.head_count", heads) ||
            !metadata_u64(model, "llama.feed_forward_length", feed_forward)) {
            return {StatusCode::InvalidArgument, "Llama GGUF is missing required layer, embedding, head or feed-forward metadata"};
        }
        if (!metadata_u64(model, "llama.attention.head_count_kv", kv_heads)) kv_heads = heads;
        if (layers == 0 || embedding == 0 || heads == 0 || kv_heads == 0 || feed_forward == 0 ||
            kv_heads > heads || embedding % heads != 0) {
            return {StatusCode::InvalidArgument, "Llama GGUF contains inconsistent model dimensions"};
        }
        return Status::success();
    }

private:
    ArchitectureDescriptor descriptor_;
};

} // namespace

Status register_builtin_architecture_adapters(ArchitectureRegistry& registry) {
    return registry.register_adapter(std::make_shared<LlamaDenseArchitectureAdapter>());
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
    return format && format->decoder_state == SupportState::Validated &&
           format->gpu_kernel_state == SupportState::Validated;
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
