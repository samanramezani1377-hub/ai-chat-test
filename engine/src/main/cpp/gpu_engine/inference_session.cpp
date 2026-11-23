#include "inference_session.h"
#include "operators.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <sstream>
#include <unordered_set>

namespace ai_chat::gpu_engine {
namespace {

constexpr uint64_t kGpuMemorySafetyReserveDivisor = 8;

const char* backend_label(BackendKind backend) {
    switch (backend) {
        case BackendKind::DawnWebGpu: return "DawnWebGPU";
        case BackendKind::OpenGles: return "UniversalGLES";
        default: return "Unknown";
    }
}

bool add_overflow(uint64_t lhs, uint64_t rhs, uint64_t& result) {
    if (lhs > UINT64_MAX - rhs) return true;
    result = lhs + rhs;
    return false;
}

std::string safe_diagnostic_atom(const std::string& value, size_t maximum = 256) {
    std::string out;
    out.reserve(std::min(value.size(), maximum));
    for (unsigned char ch : value) {
        if (out.size() >= maximum) break;
        if (ch == ';' || ch == '=' || ch == '|' || ch == ',' ||
            ch == '\n' || ch == '\r' || ch == '\t') out.push_back('_');
        else if (ch >= 0x20 && ch != 0x7f) out.push_back(static_cast<char>(ch));
    }
    if (value.size() > out.size() && out.size() == maximum) out += "~";
    return out;
}

std::string metadata_value_summary(const GgufMetadataValue& value) {
    if (const auto* v = std::get_if<std::string>(&value)) return "s:" + safe_diagnostic_atom(*v, 128);
    if (const auto* v = std::get_if<uint64_t>(&value)) return "u:" + std::to_string(*v);
    if (const auto* v = std::get_if<int64_t>(&value)) return "i:" + std::to_string(*v);
    if (const auto* v = std::get_if<double>(&value)) {
        std::ostringstream out; out << "f:" << *v; return out.str();
    }
    if (const auto* v = std::get_if<bool>(&value)) return *v ? "b:1" : "b:0";
    if (const auto* v = std::get_if<GgufArrayInfo>(&value))
        return "array:" + std::to_string(v->count) + ":type" + std::to_string(v->element_type);
    if (const auto* v = std::get_if<GgufMetadataArray>(&value))
        return "array:" + std::to_string(v->size()) + ":type" + std::to_string(v->element_type);
    if (const auto* v = std::get_if<std::vector<std::string>>(&value))
        return "strings:" + std::to_string(v->size());
    if (const auto* v = std::get_if<std::vector<double>>(&value))
        return "floats:" + std::to_string(v->size());
    if (const auto* v = std::get_if<std::vector<int64_t>>(&value))
        return "ints:" + std::to_string(v->size());
    return "unknown";
}

std::string describe_unsupported_gguf_structure(const GgufFileInfo& model,
                                                size_t maximum_characters = 14000) {
    std::ostringstream out;
    out << "GGUF_STRUCTURE{version=" << model.version
        << ",architecture=" << safe_diagnostic_atom(model.architecture)
        << ",model=" << safe_diagnostic_atom(model.model_name)
        << ",metadata_count=" << model.metadata_count
        << ",tensor_count=" << model.tensor_count
        << ",tokenizer_model=" << safe_diagnostic_atom(model.tokenizer.model)
        << ",pre_tokenizer=" << safe_diagnostic_atom(model.tokenizer.pre_tokenizer)
        << ",vocab=" << model.tokenizer.tokens.size()
        << ",merges=" << model.tokenizer.merges.size()
        << ",architecture_metadata=[";
    bool first = true;
    bool truncated = false;
    const std::string prefix = model.architecture + ".";
    std::vector<std::string> metadata_keys;
    metadata_keys.reserve(model.metadata.size());
    for (const auto& entry : model.metadata) {
        if (entry.first == "general.architecture" || entry.first == "general.name" ||
            entry.first.compare(0, prefix.size(), prefix) == 0)
            metadata_keys.push_back(entry.first);
    }
    std::sort(metadata_keys.begin(), metadata_keys.end());
    for (const auto& key : metadata_keys) {
        if (static_cast<size_t>(out.tellp()) >= maximum_characters - 512) {
            truncated = true;
            break;
        }
        const auto found = model.metadata.find(key);
        if (found == model.metadata.end()) continue;
        if (!first) out << ",";
        first = false;
        out << safe_diagnostic_atom(key) << ":" << metadata_value_summary(found->second);
    }
    out << "],tensors=[";
    first = true;
    for (const auto& tensor : model.tensors) {
        if (static_cast<size_t>(out.tellp()) >= maximum_characters - 512) {
            truncated = true;
            break;
        }
        if (!first) out << ",";
        first = false;
        out << safe_diagnostic_atom(tensor.name) << "[";
        for (size_t i = 0; i < tensor.dimensions.size(); ++i) {
            if (i) out << "x";
            out << tensor.dimensions[i];
        }
        out << "]:" << safe_diagnostic_atom(tensor.type.name)
            << "#" << tensor.type_id << ":";
        if (tensor.byte_size_known) out << tensor.byte_size;
        else out << "?";
    }
    out << "]";
    if (truncated) out << ",truncated=1";
    out << "}";
    return out.str();
}

} // namespace

GpuInferenceSession::~GpuInferenceSession() { unload(); }

void GpuInferenceSession::unload_locked() noexcept {
    if (engine_) engine_->shutdown();
    context_ = ModelExecutionContext{};
    kv_cache_.reset();
    memory_.reset();
    loader_.reset();
    lowerings_.reset();
    tokenizers_.reset();
    architectures_.reset();
    quantizations_.reset();
    operators_.reset();
    engine_.reset();
    plan_ = ModelExecutionPlan{};
    memory_estimate_ = ModelGpuMemoryEstimateReport{};
    context_length_ = 0;
    memory_budget_bytes_ = 0;
    total_weight_bytes_ = 0;
    largest_weight_tensor_bytes_ = 0;
    peak_operation_weight_bytes_ = 0;
    kv_cache_bytes_ = 0;
    last_tokenization_ms_ = 0;
    last_prefill_ms_ = 0;
    last_prefill_readback_ms_ = 0;
    last_ttft_ms_ = 0;
    last_decode_ms_ = 0;
    last_tokenization_timing_valid_ = false;
    last_prefill_timing_valid_ = false;
    last_prefill_readback_timing_valid_ = false;
    last_ttft_timing_valid_ = false;
    last_decode_timing_valid_ = false;
    last_generation_stage_ = "not_started";
    last_model_graph_validation_ = "not_run";
    cross_backend_quantization_validation_ = "not_run";
    last_prompt_tokens_ = 0;
    last_output_tokens_ = 0;
    last_decode_tokens_per_second_ = 0.0;
    last_generation_completed_ = false;
    model_path_.clear();
    backend_fallback_reason_.clear();
    loaded_ = false;
}

void GpuInferenceSession::unload() noexcept {
    cancelled_.store(true, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(mutex_);
    unload_locked();
    cancelled_.store(false, std::memory_order_relaxed);
}

void GpuInferenceSession::cancel() noexcept {
    cancelled_.store(true, std::memory_order_relaxed);
}

void GpuInferenceSession::reset_stop() noexcept {
    cancelled_.store(false, std::memory_order_relaxed);
}

bool GpuInferenceSession::loaded() const noexcept {
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        return loaded_;
    } catch (...) {
        return false;
    }
}

bool GpuInferenceSession::uses_qwen35_tokenizer() const noexcept {
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        return loaded_ && loader_ && loader_->reader() &&
               loader_->reader()->info().tokenizer.model == "gpt2" &&
               loader_->reader()->info().tokenizer.pre_tokenizer == "qwen35";
    } catch (...) {
        return false;
    }
}

bool GpuInferenceSession::uses_nfc_tokenizer() const noexcept {
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!loaded_ || !loader_ || !loader_->reader()) return false;
        const auto& tokenizer = loader_->reader()->info().tokenizer;
        return tokenizer.model == "gpt2" &&
               (tokenizer.pre_tokenizer == "qwen35" ||
                tokenizer.pre_tokenizer == "qwen2" ||
                tokenizer.pre_tokenizer == "qwen3");
    } catch (...) {
        return false;
    }
}

std::string GpuInferenceSession::backend_name() const {
    return engine_ ? backend_label(engine_->diagnostics().selected_backend) : "Unknown";
}

Status GpuInferenceSession::load_model(const std::string& path, uint32_t context_length,
                                       uint64_t memory_budget_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    unload_locked();
    last_failed_memory_budget_bytes_ = 0;
    last_failed_memory_required_bytes_ = 0;
    last_failed_memory_usable_bytes_ = 0;
    last_failed_memory_kv_bytes_ = 0;
    last_failed_memory_scratch_bytes_ = 0;
    last_failed_tensor_bytes_ = 0;
    last_failed_storage_binding_limit_bytes_ = 0;
    last_attempt_model_path_ = path;
    last_attempt_architecture_.clear();
    last_attempt_backend_.clear();
    last_attempt_gpu_vendor_.clear();
    last_attempt_gpu_device_.clear();
    last_attempt_gpu_architecture_.clear();
    last_attempt_gpu_driver_.clear();
    last_attempt_gpu_api_.clear();
    last_attempt_gpu_features_.clear();
    last_attempt_memory_budget_bytes_ = memory_budget_bytes;
    last_attempt_weight_bytes_ = 0;
    last_attempt_largest_weight_tensor_bytes_ = 0;
    last_attempt_peak_operation_weight_bytes_ = 0;
    last_attempt_kv_cache_bytes_ = 0;
    last_attempt_gpu_max_buffer_bytes_ = 0;
    last_attempt_gpu_max_storage_binding_bytes_ = 0;
    last_attempt_context_length_ = context_length;
    last_attempt_gpu_max_workgroup_invocations_ = 0;
    last_attempt_gpu_compute_available_ = false;
    last_attempt_gpu_self_test_passed_ = false;
    cancelled_.store(false, std::memory_order_relaxed);
    last_error_.clear();

    if (path.empty() || context_length == 0 || memory_budget_bytes == 0) {
        last_error_ = "Model path, context length and explicit GPU memory budget are required";
        return {StatusCode::InvalidArgument, last_error_};
    }

    engine_ = std::make_unique<GpuEngine>();
    operators_ = std::make_unique<OperatorRegistry>();
    quantizations_ = std::make_unique<QuantizationRegistry>();
    architectures_ = std::make_unique<ArchitectureRegistry>();
    tokenizers_ = std::make_unique<TokenizerRegistry>();
    lowerings_ = std::make_unique<ModelLoweringRegistry>();
    loader_ = std::make_unique<GgufModelLoader>();

    Status status = initialize_android_gpu_runtime(*engine_, *operators_, *quantizations_,
        *architectures_, *tokenizers_, *lowerings_);
    if (!status.ok()) {
        last_error_ = "No Android GPU backend passed baseline operator validation: " + status.message;
        unload_locked();
        return {status.code, last_error_};
    }

    status = loader_->inspect(path, *architectures_, *quantizations_);
    if (!status.ok() || !loader_->reader()) {
        last_error_ = "GGUF inspection failed: " + status.message;
        unload_locked();
        return {status.ok() ? StatusCode::BackendFailure : status.code, last_error_};
    }

    // The aggregate preflight report also contains intentionally deferred
    // graph/sampler milestones. Do not treat those advisory blockers as a
    // substitute for the backend-specific tokenizer, quantization, operator
    // coverage and execution checks performed below and during generation.
    const GgufReader& reader = *loader_->reader();
    const GgufFileInfo& model = reader.info();
    BackendKind backend_kind = engine_->active_backend()->kind();

    const auto tokenizer_preflight = evaluate_tokenizer_preflight(model.tokenizer, *tokenizers_);
    if (!tokenizer_preflight.ready) {
        last_error_ = "Tokenizer is not validated for this GGUF model";
        last_error_ += " | gguf_architecture=" + safe_diagnostic_atom(model.architecture);
        last_error_ += " | gguf_model_name=" + safe_diagnostic_atom(model.model_name);
        last_error_ += " | " + describe_tokenizer_metadata(model.tokenizer, *tokenizers_);
        for (const auto& blocker : tokenizer_preflight.blockers) last_error_ += " | " + blocker;
        unload_locked();
        return {StatusCode::Unsupported, last_error_};
    }
    const auto architecture = architectures_->find(model.architecture);
    if (!architecture) {
        last_error_ = "Unsupported model architecture: " + model.architecture;
        last_error_ += " | " + describe_unsupported_gguf_structure(model);
        unload_locked();
        return {StatusCode::Unsupported, last_error_};
    }
    status = architecture->validate_metadata(model);
    if (!status.ok()) {
        last_error_ = "Model architecture metadata rejected: " + status.message;
        unload_locked();
        return status;
    }
    status = architecture->build_execution_plan(model, plan_);
    if (status.ok()) {
        auto logits_operation = std::find_if(plan_.operations.begin(), plan_.operations.end(),
            [](const ModelPlanOperation& operation) { return operation.output == "logits"; });
        if (logits_operation != plan_.operations.end() && plan_.vocabulary_size > 0)
            logits_operation->output_shape = {1, plan_.vocabulary_size};
    }
    if (!status.ok() || !plan_.structurally_validated || plan_.operations.empty()) {
        last_error_ = "Model execution plan is not structurally valid: " + status.message;
        for (const auto& blocker : plan_.blockers) last_error_ += " | " + blocker;
        unload_locked();
        return {status.ok() ? StatusCode::Unsupported : status.code, last_error_};
    }

    std::unordered_set<std::string> model_weight_names;
    for (const auto& operation : plan_.operations)
        for (const auto& name : operation.weight_tensors) model_weight_names.insert(name);

    std::unordered_set<uint32_t> quantization_types;
    total_weight_bytes_ = 0;
    largest_weight_tensor_bytes_ = 0;
    for (const auto& name : model_weight_names) {
        const GgufTensorInfo* tensor = reader.find_tensor(name);
        if (!tensor || !tensor->byte_size_known) {
            last_error_ = "Model weight storage size is unknown: " + name;
            unload_locked();
            return {StatusCode::Unsupported, last_error_};
        }
        uint64_t next_total = 0;
        if (add_overflow(total_weight_bytes_, tensor->byte_size, next_total)) {
            last_error_ = "Model weight size estimate overflowed";
            unload_locked();
            return {StatusCode::InvalidArgument, last_error_};
        }
        total_weight_bytes_ = next_total;
        largest_weight_tensor_bytes_ = std::max(largest_weight_tensor_bytes_, tensor->byte_size);

        if (tensor->type.name == "F32") continue;
        if (tensor->type.name == "Q4_0") {
            quantization_types.insert(2);
            continue;
        }
        if (tensor->type.name == "Q4_1") {
            quantization_types.insert(3);
            continue;
        }
        if (tensor->type.name == "Q8_0") {
            quantization_types.insert(8);
            continue;
        }
        if (tensor->type.name == "Q5_0") {
            quantization_types.insert(6);
            continue;
        }
        if (tensor->type.name == "Q5_1") {
            quantization_types.insert(7);
            continue;
        }
        if (tensor->type.name == "F16") {
            quantization_types.insert(1);
            continue;
        }
        if (tensor->type.name == "BF16") {
            quantization_types.insert(30);
            continue;
        }
        if (tensor->type.name == "Q6_K") {
            quantization_types.insert(14);
            continue;
        }
        if (tensor->type.name == "Q4_K") {
            quantization_types.insert(12);
            continue;
        }
        if (tensor->type.name == "Q5_K") {
            quantization_types.insert(13);
            continue;
        }
        if (tensor->type.name == "Q2_K") {
            quantization_types.insert(10);
            continue;
        }
        if (tensor->type.name == "Q3_K") {
            quantization_types.insert(11);
            continue;
        }
        last_error_ = "Weight type is not implemented by the current GPU loader: " +
                      tensor->type.name + " (" + name + ")";
        unload_locked();
        return {StatusCode::Unsupported, last_error_};
    }

    // Estimate the actual peak resident weight set. Q6_K MatMul streams
    // complete output-channel rows; Q6_K Embedding gathers only the requested
    // token rows instead of pinning the entire vocabulary matrix.
    peak_operation_weight_bytes_ = 0;
    const DeviceInfo* memory_device = engine_->active_device();
    const uint64_t weight_binding_limit = memory_device
        ? std::min(memory_device->limits.max_storage_buffer_binding_size,
                   memory_device->limits.max_buffer_size == 0
                       ? memory_device->limits.max_storage_buffer_binding_size
                       : memory_device->limits.max_buffer_size)
        : 0;
    for (const auto& operation : plan_.operations) {
        uint64_t operation_weight_bytes = 0;
        std::unordered_set<std::string> operation_weights;
        for (const auto& name : operation.weight_tensors) {
            if (!operation_weights.insert(name).second) continue;
            const GgufTensorInfo* tensor = reader.find_tensor(name);
            if (!tensor || !tensor->byte_size_known) {
                last_error_ = "Operation weight storage size is unknown: " + name;
                unload_locked();
                return {StatusCode::Unsupported, last_error_};
            }
            if (tensor->byte_size > UINT64_MAX - 3) {
                last_error_ = "Operation weight alignment size overflowed: " + name;
                unload_locked();
                return {StatusCode::InvalidArgument, last_error_};
            }
            uint64_t resident_bytes = (tensor->byte_size + 3) & ~uint64_t{3};
            if (tensor->type.name == "Q6_K" && tensor->type_id == 14 &&
                tensor->dimensions.size() == 2 && tensor->dimensions[0] % 256 == 0) {
                const uint64_t row_bytes = (tensor->dimensions[0] / 256) * 210;
                if (operation.operator_name == "MatMul_Q6_K" && row_bytes != 0 &&
                    weight_binding_limit >= row_bytes) {
                    resident_bytes = std::min(resident_bytes,
                        (weight_binding_limit / row_bytes) * row_bytes);
                } else if (operation.operator_name == "Embedding_Q6_K" && row_bytes != 0 &&
                           context_length <= UINT64_MAX / row_bytes) {
                    resident_bytes = std::min(resident_bytes, context_length * row_bytes);
                }
            }
            uint64_t next_operation_bytes = 0;
            if (add_overflow(operation_weight_bytes, resident_bytes, next_operation_bytes)) {
                last_error_ = "Peak per-operation weight set size overflowed";
                unload_locked();
                return {StatusCode::InvalidArgument, last_error_};
            }
            operation_weight_bytes = next_operation_bytes;
        }
        peak_operation_weight_bytes_ = std::max(peak_operation_weight_bytes_, operation_weight_bytes);
    }

    for (const auto& operation : plan_.operations) {
        for (const auto& name : operation.weight_tensors) {
            const GgufTensorInfo* tensor = reader.find_tensor(name);
            if (!tensor) continue;
            const bool compatible_q4_0 = tensor->type.name == "Q4_0" &&
                (operation.operator_name == "MatMul_Q4_0" || operation.operator_name == "Embedding_Q4_0");
            const bool compatible_q4_1 = tensor->type.name == "Q4_1" &&
                (operation.operator_name == "MatMul_Q4_1" || operation.operator_name == "Embedding_Q4_1");
            const bool compatible_q8_0 = tensor->type.name == "Q8_0" &&
                (operation.operator_name == "MatMul_Q8_0" ||
                 operation.operator_name == "Embedding_Q8_0");
            const bool compatible_q5_0 = tensor->type.name == "Q5_0" &&
                (operation.operator_name == "MatMul_Q5_0" || operation.operator_name == "Embedding_Q5_0");
            const bool compatible_q5_1 = tensor->type.name == "Q5_1" &&
                (operation.operator_name == "MatMul_Q5_1" || operation.operator_name == "Embedding_Q5_1");
            const bool compatible_f16 = tensor->type.name == "F16" &&
                (operation.operator_name == "MatMul_F16" || operation.operator_name == "Embedding_F16");
            const bool compatible_bf16 = tensor->type.name == "BF16" &&
                (operation.operator_name == "MatMul_BF16" || operation.operator_name == "Embedding_BF16");
            const bool compatible_q6_k = tensor->type.name == "Q6_K" &&
                (operation.operator_name == "MatMul_Q6_K" ||
                 operation.operator_name == "Embedding_Q6_K");
            const bool compatible_q4_k = tensor->type.name == "Q4_K" &&
                operation.operator_name == "MatMul_Q4_K";
            const bool compatible_q5_k = tensor->type.name == "Q5_K" &&
                operation.operator_name == "MatMul_Q5_K";
            const bool compatible_q2_k = tensor->type.name == "Q2_K" &&
                operation.operator_name == "MatMul_Q2_K";
            const bool compatible_q3_k = tensor->type.name == "Q3_K" &&
                operation.operator_name == "MatMul_Q3_K";
            if (tensor->type.name != "F32" &&
                !compatible_q4_0 && !compatible_q4_1 && !compatible_q8_0 &&
                !compatible_q5_0 && !compatible_q5_1 && !compatible_f16 && !compatible_bf16 && !compatible_q6_k && !compatible_q4_k && !compatible_q5_k && !compatible_q2_k && !compatible_q3_k) {
                last_error_ = "Quantized weight is attached to an operator without a matching packed kernel: " +
                              operation.operator_name + ":" + name;
                unload_locked();
                return {StatusCode::Unsupported, last_error_};
            }
        }
    }

    auto validate_quantization_kernels_for_active_backend =
        [&](bool continue_after_failure = false) -> Status {
        const auto has_operator = [&](const std::string& name) {
            return std::any_of(plan_.operations.begin(), plan_.operations.end(),
                [&](const ModelPlanOperation& operation) { return operation.operator_name == name; });
        };
        std::vector<std::string> failures;
        Status first_failure = Status::success();
        const auto record_failure = [&](const std::string& type_name,
                                        const std::string& stage,
                                        const Status& failure) {
            const std::string message = type_name + ":" + stage + ":" + failure.message;
            failures.push_back(message);
            if (first_failure.ok()) first_failure = failure;
            last_error_ = "GPU quantization kernel validation failed on the active backend: " + message;
        };

        for (uint32_t type_id : quantization_types) {
            std::string type_name, matmul_name, embedding_name;
            if (type_id == 2) { type_name="Q4_0"; matmul_name="MatMul_Q4_0"; embedding_name="Embedding_Q4_0"; }
            else if (type_id == 3) { type_name="Q4_1"; matmul_name="MatMul_Q4_1"; embedding_name="Embedding_Q4_1"; }
            else if (type_id == 6) { type_name="Q5_0"; matmul_name="MatMul_Q5_0"; embedding_name="Embedding_Q5_0"; }
            else if (type_id == 7) { type_name="Q5_1"; matmul_name="MatMul_Q5_1"; embedding_name="Embedding_Q5_1"; }
            else if (type_id == 1) { type_name="F16"; matmul_name="MatMul_F16"; embedding_name="Embedding_F16"; }
            else if (type_id == 30) { type_name="BF16"; matmul_name="MatMul_BF16"; embedding_name="Embedding_BF16"; }
            else if (type_id == 14) { type_name="Q6_K"; matmul_name="MatMul_Q6_K"; embedding_name="Embedding_Q6_K"; }
            else if (type_id == 12) { type_name="Q4_K"; matmul_name="MatMul_Q4_K"; }
            else if (type_id == 13) { type_name="Q5_K"; matmul_name="MatMul_Q5_K"; }
            else if (type_id == 10) { type_name="Q2_K"; matmul_name="MatMul_Q2_K"; }
            else if (type_id == 11) { type_name="Q3_K"; matmul_name="MatMul_Q3_K"; }
            else if (type_id == 8) { type_name="Q8_0"; matmul_name="MatMul_Q8_0"; embedding_name="Embedding_Q8_0"; }
            else return {StatusCode::Unsupported,
                         "No GPU validation route exists for GGUF quantization type " + std::to_string(type_id)};

            const bool require_matmul = has_operator(matmul_name) ||
                type_id == 2 || type_id == 3 || type_id == 8;
            if (require_matmul) {
                Status validation_status = Status::success();
                if (type_id == 2) validation_status=validate_q4_0_matmul_kernel(*engine_->active_backend(),*operators_,*quantizations_);
                else if (type_id == 3) validation_status=validate_q4_1_matmul_kernel(*engine_->active_backend(),*operators_,*quantizations_);
                else if (type_id == 6) validation_status=validate_q5_0_matmul_kernel(*engine_->active_backend(),*operators_,*quantizations_);
                else if (type_id == 7) validation_status=validate_q5_1_matmul_kernel(*engine_->active_backend(),*operators_,*quantizations_);
                else if (type_id == 1) validation_status=validate_f16_matmul_kernel(*engine_->active_backend(),*operators_,*quantizations_);
                else if (type_id == 30) validation_status=validate_bf16_matmul_kernel(*engine_->active_backend(),*operators_,*quantizations_);
                else if (type_id == 14) validation_status=validate_q6_k_matmul_kernel(*engine_->active_backend(),*operators_,*quantizations_);
                else if (type_id == 12) validation_status=validate_q4_k_matmul_kernel(*engine_->active_backend(),*operators_,*quantizations_);
                else if (type_id == 13) validation_status=validate_q5_k_matmul_kernel(*engine_->active_backend(),*operators_,*quantizations_);
                else if (type_id == 10) validation_status=validate_q2_k_matmul_kernel(*engine_->active_backend(),*operators_,*quantizations_);
                else if (type_id == 11) validation_status=validate_q3_k_matmul_kernel(*engine_->active_backend(),*operators_,*quantizations_);
                else validation_status=validate_q8_0_matmul_kernel(*engine_->active_backend(),*operators_,*quantizations_);
                if (!validation_status.ok()) {
                    record_failure(type_name, "matmul", validation_status);
                    if (!continue_after_failure) return validation_status;
                }
            }

            if (has_operator(embedding_name)) {
                Status validation_status = Status::success();
                if (type_id == 2) validation_status=validate_q4_0_embedding_kernel(*engine_->active_backend(),*operators_);
                else if (type_id == 3) validation_status=validate_q4_1_embedding_kernel(*engine_->active_backend(),*operators_);
                else if (type_id == 6) validation_status=validate_q5_0_embedding_kernel(*engine_->active_backend(),*operators_,*quantizations_);
                else if (type_id == 7) validation_status=validate_q5_1_embedding_kernel(*engine_->active_backend(),*operators_,*quantizations_);
                else if (type_id == 1) validation_status=validate_f16_embedding_kernel(*engine_->active_backend(),*operators_,*quantizations_);
                else if (type_id == 30) validation_status=validate_bf16_embedding_kernel(*engine_->active_backend(),*operators_,*quantizations_);
                else if (type_id == 14) validation_status=validate_q6_k_embedding_kernel(*engine_->active_backend(),*operators_,*quantizations_);
                else if (type_id == 8) validation_status=validate_q8_0_embedding_kernel(*engine_->active_backend(),*operators_);
                else validation_status={StatusCode::Unsupported,
                    "No embedding kernel validation route exists for this GGUF format"};
                if (!validation_status.ok()) {
                    record_failure(type_name, "embedding", validation_status);
                    if (!continue_after_failure) return validation_status;
                }
            }
        }
        if (!failures.empty()) {
            std::string message;
            for (const auto& failure : failures) {
                if (!message.empty()) message += " | ";
                message += failure;
            }
            last_error_ = "GPU quantization kernel validation failed on the active backend: " + message;
            return {first_failure.code, message};
        }
        return Status::success();
    };

    status = validate_quantization_kernels_for_active_backend();
    if (!status.ok()) {
        const BackendKind alternate_backend = backend_kind == BackendKind::DawnWebGpu
            ? BackendKind::OpenGles : BackendKind::DawnWebGpu;
        const Status first_backend_failure = status;
        engine_->shutdown();
        status = engine_->initialize(alternate_backend);
        if (status.ok()) {
            KernelValidationReport alternate_operator_report;
            status = validate_builtin_operator_kernels(
                *engine_->active_backend(), *operators_, alternate_operator_report);
            if (status.ok() && !alternate_operator_report.complete) {
                std::string detail = "Alternate backend failed baseline operator validation";
                for (const auto& item : alternate_operator_report.unavailable)
                    detail += " | unavailable=" + item;
                for (const auto& item : alternate_operator_report.rejected)
                    detail += " | rejected=" + item;
                status = {StatusCode::Unsupported, std::move(detail)};
            }
            if (status.ok()) {
                backend_kind = alternate_backend;
                backend_fallback_reason_ = first_backend_failure.message;
                status = validate_quantization_kernels_for_active_backend();
            }
        }
        if (!status.ok()) {
            last_error_ = "Quantized kernels are unavailable on both GPU backends; first backend: " +
                          first_backend_failure.message + "; alternate backend: " + status.message;
            unload_locked();
            return {status.code, last_error_};
        }
    }


    if (!quantization_types.empty()) {
        const std::string primary_error = last_error_;
        cross_backend_quantization_validation_ =
            std::string("decode_reference=not_run;active_backend_kernel_validation=") +
            backend_label(backend_kind) + ":passed";
        const BackendKind alternate_backend = backend_kind == BackendKind::DawnWebGpu
            ? BackendKind::OpenGles : BackendKind::DawnWebGpu;

        engine_->shutdown();
        Status alternate_status = engine_->initialize(alternate_backend);
        if (alternate_status.ok() && engine_->active_backend() &&
            engine_->active_backend()->kind() == alternate_backend) {
            const std::string alternate_previous_error = last_error_;
            alternate_status = validate_quantization_kernels_for_active_backend(true);
            const std::string alternate_detail = alternate_status.ok()
                ? "all_model_gguf_formats_passed"
                : alternate_status.message;
            last_error_ = alternate_previous_error;
            cross_backend_quantization_validation_ += ";" +
                std::string("alternate_backend=") + backend_label(alternate_backend) + ":" +
                (alternate_status.ok() ? "passed:" : "failed:") + alternate_detail;
        } else {
            cross_backend_quantization_validation_ += std::string(";alternate_backend=") +
                backend_label(alternate_backend) + ":unavailable:" + alternate_status.message;
        }

        engine_->shutdown();
        Status restore_status = engine_->initialize(backend_kind);
        if (!restore_status.ok()) {
            last_error_ = "Failed to restore the selected GPU backend after cross-backend GGUF validation: " +
                          restore_status.message;
            unload_locked();
            return {restore_status.code, last_error_};
        }
        last_error_ = primary_error;
    } else {
        cross_backend_quantization_validation_ = "not_required_no_packed_gguf_weights";
    }

    auto quantization_report = evaluate_model_quantization_backend(model, *quantizations_, backend_kind);
    if (!quantization_report.compatible) {
        last_error_ = "Model quantization is not compatible with the selected GPU backend";
        for (const auto& blocker : quantization_report.blockers) last_error_ += " | " + blocker;
        unload_locked();
        return {StatusCode::Unsupported, last_error_};
    }

    ModelGraphCoverageReport coverage;
    status = validate_model_plan_operator_coverage(plan_, *operators_, backend_kind, coverage);
    if (!status.ok() || !coverage.complete) {
        std::string first_backend_reason = status.ok()
            ? "Model operator coverage is incomplete" : status.message;
        for (const auto& blocker : coverage.unsupported_operators)
            first_backend_reason += " | " + blocker;
        for (const auto& blocker : plan_.blockers)
            first_backend_reason += " | " + blocker;

        const BackendKind alternate_backend = backend_kind == BackendKind::DawnWebGpu
            ? BackendKind::OpenGles : BackendKind::DawnWebGpu;
        engine_->shutdown();
        status = engine_->initialize(alternate_backend);
        if (status.ok()) {
            KernelValidationReport alternate_operator_report;
            status = validate_builtin_operator_kernels(
                *engine_->active_backend(), *operators_, alternate_operator_report);
            if (status.ok() && !alternate_operator_report.complete) {
                std::string detail = "Alternate backend failed baseline operator validation";
                for (const auto& item : alternate_operator_report.unavailable)
                    detail += " | unavailable=" + item;
                for (const auto& item : alternate_operator_report.rejected)
                    detail += " | rejected=" + item;
                status = {StatusCode::Unsupported, std::move(detail)};
            }
        }
        if (status.ok()) status = validate_quantization_kernels_for_active_backend();
        if (status.ok()) {
            quantization_report = evaluate_model_quantization_backend(
                model, *quantizations_, alternate_backend);
            if (!quantization_report.compatible) {
                std::string detail = "Alternate backend quantization support is incomplete";
                for (const auto& blocker : quantization_report.blockers)
                    detail += " | " + blocker;
                status = {StatusCode::Unsupported, std::move(detail)};
            }
        }
        ModelGraphCoverageReport alternate_coverage;
        if (status.ok()) {
            status = validate_model_plan_operator_coverage(
                plan_, *operators_, alternate_backend, alternate_coverage);
            if (status.ok() && !alternate_coverage.complete) {
                std::string detail = "Alternate backend model operator coverage is incomplete";
                for (const auto& blocker : alternate_coverage.unsupported_operators)
                    detail += " | " + blocker;
                status = {StatusCode::Unsupported, std::move(detail)};
            }
        }
        if (!status.ok()) {
            last_error_ = "Model is not supported by either GPU backend; primary backend: " +
                          first_backend_reason + "; alternate backend: " + status.message;
            unload_locked();
            return {status.code, last_error_};
        }
        backend_kind = alternate_backend;
        backend_fallback_reason_ = first_backend_reason;
        coverage = std::move(alternate_coverage);
    }

    const DeviceInfo* device = engine_->active_device();
    if (!device || !device->limits.max_buffer_size ||
        !device->limits.max_storage_buffer_binding_size) {
        last_error_ = "GPU backend did not expose required buffer limits";
        unload_locked();
        return {StatusCode::Unsupported, last_error_};
    }
    last_attempt_backend_ = backend_label(backend_kind);
    last_attempt_gpu_vendor_ = device->vendor;
    last_attempt_gpu_device_ = device->name;
    last_attempt_gpu_architecture_ = device->architecture;
    last_attempt_gpu_driver_ = device->driver;
    last_attempt_gpu_api_ = device->api_version;
    last_attempt_gpu_features_ = device->features;
    last_attempt_gpu_max_buffer_bytes_ = device->limits.max_buffer_size;
    last_attempt_gpu_max_storage_binding_bytes_ = device->limits.max_storage_buffer_binding_size;
    last_attempt_gpu_max_workgroup_invocations_ = device->limits.max_compute_workgroup_invocations;
    last_attempt_gpu_compute_available_ = device->compute_available;
    last_attempt_gpu_self_test_passed_ = device->compute_self_test_passed;
    last_attempt_architecture_ = plan_.architecture;
    last_attempt_weight_bytes_ = total_weight_bytes_;
    last_attempt_largest_weight_tensor_bytes_ = largest_weight_tensor_bytes_;
    last_attempt_peak_operation_weight_bytes_ = peak_operation_weight_bytes_;
    last_attempt_kv_cache_bytes_ = kv_cache_bytes_;
    const uint64_t max_single_buffer = std::min(device->limits.max_buffer_size,
                                                device->limits.max_storage_buffer_binding_size);
    for (const auto& tensor : model.tensors) {
        if (!tensor.byte_size_known || tensor.byte_size <= max_single_buffer) continue;
        bool referenced = false;
        bool all_uses_chunkable = true;
        uint64_t row_bytes = 0;
        const bool q6_k = tensor.type.name == "Q6_K" && tensor.type_id == 14 &&
                          tensor.dimensions.size() == 2 && tensor.dimensions[0] != 0 &&
                          tensor.dimensions[0] % 256 == 0;
        if (q6_k) row_bytes = (tensor.dimensions[0] / 256) * 210;
        for (const auto& operation : plan_.operations) {
            if (std::find(operation.weight_tensors.begin(), operation.weight_tensors.end(),
                          tensor.name) == operation.weight_tensors.end()) continue;
            referenced = true;
            const bool supported_q6_use = q6_k &&
                (operation.operator_name == "MatMul_Q6_K" ||
                 operation.operator_name == "Embedding_Q6_K") &&
                row_bytes != 0 && row_bytes <= max_single_buffer;
            if (!supported_q6_use) all_uses_chunkable = false;
        }
        if (!referenced || all_uses_chunkable) continue;
        const uint64_t padded_bytes = tensor.byte_size > UINT64_MAX - 3
            ? UINT64_MAX : (tensor.byte_size + 3) & ~uint64_t{3};
        last_failed_tensor_bytes_ = padded_bytes;
        last_failed_storage_binding_limit_bytes_ = max_single_buffer;
        last_failed_memory_budget_bytes_ = memory_budget_bytes;
        last_error_ = "A model tensor exceeds the backend storage-buffer limit and has no validated chunk-aware operator path; " +
            std::string("tensor=") + tensor.name +
            " | tensor_bytes=" + std::to_string(padded_bytes) +
            " | max_storage_binding_bytes=" + std::to_string(max_single_buffer);
        unload_locked();
        return {StatusCode::Unsupported, last_error_};
    }
    if (plan_.layer_count == 0 || plan_.layer_count > UINT32_MAX ||
        plan_.kv_heads == 0 || plan_.kv_heads > UINT32_MAX ||
        plan_.head_dimension == 0 || plan_.head_dimension > UINT32_MAX ||
        plan_.vocabulary_size == 0 || plan_.vocabulary_size > INT32_MAX ||
        std::find(plan_.retained_outputs.begin(), plan_.retained_outputs.end(), "logits") == plan_.retained_outputs.end() ||
        model.tokenizer.tokens.size() != plan_.vocabulary_size ||
        model.tokenizer.eos_token_id > INT32_MAX) {
        last_error_ = "Model plan is missing valid layer, KV-head, head-dimension or vocabulary metadata";
        unload_locked();
        return {StatusCode::Unsupported, last_error_};
    }

    context_length_ = context_length;
    memory_budget_bytes_ = memory_budget_bytes;
    ModelGpuMemoryEstimateInput estimate_input;
    estimate_input.total_weight_bytes = total_weight_bytes_;
    estimate_input.streaming_chunk_bytes = peak_operation_weight_bytes_;
    // Loading budgets for steady-state decode (one token). Prefill scratch
    // depends on the actual prompt length and is checked after tokenization.
    estimate_input.token_count = 1;
    estimate_input.context_length = context_length_;
    estimate_input.bytes_per_element = sizeof(float);
    estimate_input.safety_reserve_bytes = memory_budget_bytes_ / kGpuMemorySafetyReserveDivisor;
    estimate_input.device_budget_bytes = memory_budget_bytes_;
    estimate_input.stream_weights = true;
    memory_estimate_ = estimate_model_gpu_memory(plan_, estimate_input);
    if (!memory_estimate_.plan.fits) {
        last_error_ = "GPU memory preflight rejected the model";
        for (const auto& blocker : memory_estimate_.blockers) last_error_ += " | " + blocker;
        for (const auto& blocker : memory_estimate_.plan.blockers) last_error_ += " | " + blocker;
        const std::string failure = last_error_;
        const auto failed_estimate = memory_estimate_;
        last_failed_memory_budget_bytes_ = memory_budget_bytes_;
        last_failed_memory_required_bytes_ = failed_estimate.plan.required_bytes;
        last_failed_memory_usable_bytes_ = failed_estimate.plan.usable_budget_bytes;
        last_failed_memory_kv_bytes_ = failed_estimate.estimated_kv_cache_bytes;
        last_failed_memory_scratch_bytes_ = failed_estimate.estimated_scratch_bytes;
        const uint64_t failed_budget = memory_budget_bytes_;
        const uint32_t failed_context = context_length_;
        unload_locked();
        memory_estimate_ = failed_estimate;
        memory_budget_bytes_ = failed_budget;
        context_length_ = failed_context;
        last_error_ = failure;
        return {StatusCode::OutOfMemory, last_error_};
    }

    memory_ = std::make_unique<MemoryManager>(*engine_->active_backend(), memory_budget_bytes_);
    KvCacheConfig cache_config;
    cache_config.layers = static_cast<uint32_t>(plan_.layer_count);
    cache_config.kv_heads = static_cast<uint32_t>(plan_.kv_heads);
    cache_config.head_dimension = static_cast<uint32_t>(plan_.head_dimension);
    cache_config.maximum_sequence_length = context_length_;
    cache_config.data_type = DataType::F32;
    status = KvCache::estimate_bytes(cache_config, kv_cache_bytes_);
    if (!status.ok()) {
        last_error_ = "KV cache estimate failed: " + status.message;
        unload_locked();
        return status;
    }
    if (kv_cache_bytes_ > memory_budget_bytes_) {
        last_error_ = "KV cache alone exceeds the explicit GPU memory budget";
        unload_locked();
        return {StatusCode::OutOfMemory, last_error_};
    }
    status = KvCache::create(*engine_->active_backend(), *memory_, cache_config, kv_cache_);
    if (!status.ok()) {
        last_error_ = "KV cache allocation failed: " + status.message;
        unload_locked();
        return status;
    }

    context_.backend = engine_->active_backend();
    context_.kv_cache = kv_cache_.get();
    context_.context_length = context_length_;

    std::unordered_set<std::string> persistent_state_names;
    auto allocate_persistent_state = [&](const std::string& name,
                                         const std::vector<uint64_t>& shape) -> Status {
        if (name.empty() || !persistent_state_names.insert(name).second)
            return {StatusCode::InvalidArgument, "Hybrid state binding name is empty or duplicated"};
        TensorDesc descriptor;
        descriptor.shape = shape;
        descriptor.data_type = DataType::F32;
        descriptor.layout = TensorLayout::Contiguous;
        descriptor.memory = MemoryDomain::Device;
        descriptor.alignment = 4;
        uint64_t bytes = 0;
        Status result = Tensor::required_bytes(descriptor, bytes);
        if (!result.ok()) return result;
        if (bytes == 0 || bytes > SIZE_MAX)
            return {StatusCode::InvalidArgument, "Hybrid state tensor has an invalid byte size"};
        BufferDesc buffer_desc;
        buffer_desc.size_bytes = bytes;
        buffer_desc.alignment = 4;
        buffer_desc.memory = MemoryDomain::Device;
        buffer_desc.persistent = true;
        buffer_desc.label = "hybrid-state-" + name;
        MemoryLease lease;
        result = memory_->allocate(buffer_desc, lease);
        if (!result.ok()) return result;
        if (!lease || !lease.buffer() || lease.buffer()->size_bytes() < bytes)
            return {StatusCode::BackendFailure, "Hybrid state allocation returned an undersized buffer"};
        std::vector<uint8_t> zeros(static_cast<size_t>(bytes), 0);
        result = engine_->active_backend()->upload(lease.buffer(), 0, zeros.data(), bytes);
        if (!result.ok()) return result;
        Tensor tensor;
        result = Tensor::create(std::move(descriptor), lease.buffer(), 0, tensor);
        if (!result.ok()) return result;
        context_.tensors.emplace(name, std::move(tensor));
        context_.state_leases.push_back(std::move(lease));
        return Status::success();
    };
    for (const auto& operation : plan_.operations) {
        if (operation.operator_name == "SSMConv1D") {
            if (operation.inputs.size() != 2 || operation.output_shape.size() != 2)
                { last_error_ = "SSMConv1D state metadata is invalid"; unload_locked(); return {StatusCode::InvalidArgument, last_error_}; }
            const auto kernel = operation.integer_attributes.find("kernel_size");
            if (kernel == operation.integer_attributes.end() || kernel->second < 2 ||
                operation.output_shape[1] == 0)
                { last_error_ = "SSMConv1D state dimensions are invalid"; unload_locked(); return {StatusCode::InvalidArgument, last_error_}; }
            status = allocate_persistent_state(operation.inputs[1],
                {operation.output_shape[1], kernel->second - 1});
        } else if (operation.operator_name == "GatedDeltaNet") {
            if (operation.inputs.size() != 6)
                { last_error_ = "GatedDeltaNet state metadata is invalid"; unload_locked(); return {StatusCode::InvalidArgument, last_error_}; }
            const auto state_size = operation.integer_attributes.find("state_size");
            const auto value_heads = operation.integer_attributes.find("value_heads");
            if (state_size == operation.integer_attributes.end() ||
                value_heads == operation.integer_attributes.end() ||
                state_size->second == 0 || value_heads->second == 0)
                { last_error_ = "GatedDeltaNet recurrent-state dimensions are invalid"; unload_locked(); return {StatusCode::InvalidArgument, last_error_}; }
            status = allocate_persistent_state(operation.inputs[5],
                {value_heads->second, state_size->second, state_size->second});
        }
        if (!status.ok()) {
            last_error_ = "Hybrid persistent state allocation failed: " + status.message;
            unload_locked();
            return status;
        }
    }
    model_path_ = path;
    loaded_ = true;
    last_ttft_ms_ = 0;
    last_decode_tokens_per_second_ = 0.0;
    return Status::success();
}

Status GpuInferenceSession::run_model_step(uint64_t token_count) {
    if (!loaded_ || !engine_ || !engine_->active_backend() || !loader_ || !loader_->reader() ||
        !memory_ || !engine_->active_device() || !lowerings_ || !operators_)
        return {StatusCode::NotReady, "GPU inference session is not initialized"};
    ModelGraphCompileReport report;
    Status status = execute_model_execution_plan_streamed(plan_, *engine_->active_backend(),
        *lowerings_, *operators_, plan_.external_inputs, *loader_->reader(),
        *memory_, *engine_->active_device(), token_count, &cancelled_, context_, report);
    if (!status.ok()) {
        last_model_graph_validation_ = "failed:" + status.message;
        last_error_ = status.message;
        return status;
    }
    last_model_graph_validation_ = report.compiled_nodes == plan_.operations.size() &&
        report.errors.empty() ? "validated" : "partial";
    return Status::success();
}

Status GpuInferenceSession::read_last_token_logits(std::vector<float>& logits) {
    if (!engine_ || !engine_->active_backend())
        return {StatusCode::NotReady, "GPU backend is unavailable for logits readback"};
    const auto found = context_.tensors.find("logits");
    if (found == context_.tensors.end())
        return {StatusCode::NotReady, "Model plan did not retain a logits output tensor"};
    const Tensor& tensor = found->second;
    const auto& shape = tensor.descriptor().shape;
    if (tensor.descriptor().data_type != DataType::F32 || shape.size() != 2 ||
        shape[0] == 0 || shape[1] != plan_.vocabulary_size || shape[1] > UINT32_MAX)
        return {StatusCode::Unsupported, "Logits output must be contiguous F32 [tokens,vocabulary]"};
    const uint64_t vocabulary = shape[1];
    const uint64_t row_bytes = vocabulary * sizeof(float);
    if (row_bytes == 0 || shape[0] - 1 > UINT64_MAX / row_bytes)
        return {StatusCode::InvalidArgument, "Logits row offset overflowed"};
    const uint64_t row_offset = (shape[0] - 1) * row_bytes;
    if (tensor.byte_offset() > UINT64_MAX - row_offset)
        return {StatusCode::InvalidArgument, "Logits byte offset overflowed"};
    const uint64_t offset = tensor.byte_offset() + row_offset;
    if (offset > tensor.buffer()->size_bytes() ||
        row_bytes > tensor.buffer()->size_bytes() - offset)
        return {StatusCode::InvalidArgument, "Logits readback range exceeds its backing buffer"};
    logits.resize(static_cast<size_t>(vocabulary));
    return engine_->active_backend()->readback(tensor.buffer(), offset, logits.data(), row_bytes);
}

Status GpuInferenceSession::update_memory_budget(uint64_t allocation_cap_bytes) {
    if (allocation_cap_bytes == 0)
        return {StatusCode::InvalidArgument, "GPU engine allocation cap must be non-zero"};
    std::lock_guard<std::mutex> lock(mutex_);
    if (memory_) {
        Status status = memory_->update_budget(allocation_cap_bytes);
        if (!status.ok()) return status;
    }
    memory_budget_bytes_ = allocation_cap_bytes;
    return Status::success();
}

Status GpuInferenceSession::generate(const std::string& prompt, uint32_t max_new_tokens,
                                     SamplingConfig sampling, const TextCallback& on_text) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!loaded_ || !loader_ || !tokenizers_ || !kv_cache_)
        return {StatusCode::NotReady, "GPU inference session is not loaded"};
    if (prompt.empty() || max_new_tokens == 0)
        return {StatusCode::InvalidArgument, "Prompt and max_new_tokens must be non-empty"};
    if (cancelled_.load(std::memory_order_relaxed))
        return {StatusCode::Cancelled, "GPU generation was cancelled before it started"};

    const auto request_started = std::chrono::steady_clock::now();
    last_tokenization_ms_ = 0;
    last_prefill_ms_ = 0;
    last_prefill_readback_ms_ = 0;
    last_ttft_ms_ = 0;
    last_decode_ms_ = 0;
    last_tokenization_timing_valid_ = false;
    last_prefill_timing_valid_ = false;
    last_prefill_readback_timing_valid_ = false;
    last_ttft_timing_valid_ = false;
    last_decode_timing_valid_ = false;
    last_generation_stage_ = "tokenization";
    last_model_graph_validation_ = "not_run";
    last_prompt_tokens_ = 0;
    last_output_tokens_ = 0;
    last_decode_tokens_per_second_ = 0.0;
    last_generation_completed_ = false;

    TokenizerEngine tokenizer_engine(*tokenizers_);
    std::vector<uint32_t> prompt_tokens;
    const auto tokenization_started = std::chrono::steady_clock::now();
    Status status = tokenizer_engine.encode(loader_->reader()->info().tokenizer, prompt, prompt_tokens);
    const auto tokenization_finished = std::chrono::steady_clock::now();
    last_tokenization_ms_ = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        tokenization_finished - tokenization_started).count());
    last_tokenization_timing_valid_ = status.ok();
    if (!status.ok()) {
        last_generation_stage_ = "tokenization_failed";
        last_error_ = "Prompt tokenization failed: " + status.message;
        return status;
    }
    last_prompt_tokens_ = prompt_tokens.size();
    if (prompt_tokens.empty() || prompt_tokens.size() > context_length_ ||
        max_new_tokens > context_length_ - prompt_tokens.size()) {
        last_generation_stage_ = "prompt_context_validation_failed";
        last_error_ = "Prompt plus requested generation exceeds the validated context length";
        return {StatusCode::InvalidArgument, last_error_};
    }

    // Re-check peak memory for the actual prefill length. This avoids rejecting
    // a model based on a hypothetical full-context prompt while still refusing
    // a real prompt whose peak activation memory exceeds the explicit budget.
    ModelGpuMemoryEstimateInput prefill_input;
    prefill_input.total_weight_bytes = total_weight_bytes_;
    prefill_input.streaming_chunk_bytes = peak_operation_weight_bytes_;
    prefill_input.token_count = prompt_tokens.size();
    prefill_input.context_length = context_length_;
    prefill_input.bytes_per_element = sizeof(float);
    prefill_input.safety_reserve_bytes = memory_budget_bytes_ / kGpuMemorySafetyReserveDivisor;
    prefill_input.device_budget_bytes = memory_budget_bytes_;
    prefill_input.stream_weights = true;
    const auto prefill_estimate = estimate_model_gpu_memory(plan_, prefill_input);
    memory_estimate_ = prefill_estimate;
    if (!prefill_estimate.plan.fits) {
        last_generation_stage_ = "prefill_memory_preflight_failed";
        last_error_ = "GPU prefill memory preflight rejected the actual prompt length";
        for (const auto& blocker : prefill_estimate.blockers) last_error_ += " | " + blocker;
        for (const auto& blocker : prefill_estimate.plan.blockers) last_error_ += " | " + blocker;
        return {StatusCode::OutOfMemory, last_error_};
    }
    last_error_.clear();

    for (const auto& name : plan_.external_inputs) {
        const bool recurrent = name.size() >= 15 &&
            name.compare(name.size() - 15, 15, "recurrent_state") == 0;
        const bool convolution = name.size() >= 10 &&
            name.compare(name.size() - 10, 10, "conv_state") == 0;
        if (!recurrent && !convolution) continue;
        const auto found = context_.tensors.find(name);
        if (found == context_.tensors.end() || !found->second.buffer()) {
            last_error_ = "Hybrid state binding is missing: " + name;
            return {StatusCode::NotReady, last_error_};
        }
        const uint64_t bytes = found->second.byte_size();
        if (bytes == 0 || bytes > SIZE_MAX) {
            last_error_ = "Hybrid state buffer size is invalid: " + name;
            return {StatusCode::InvalidArgument, last_error_};
        }
        std::vector<uint8_t> zeros(static_cast<size_t>(bytes), 0);
        status = engine_->active_backend()->upload(found->second.buffer(),
            found->second.byte_offset(), zeros.data(), bytes);
        if (!status.ok()) {
            last_error_ = "Hybrid state reset failed: " + status.message;
            return status;
        }
    }
    for (uint32_t layer = 0; layer < plan_.layer_count; ++layer) {
        status = kv_cache_->reset_layer(layer);
        if (!status.ok()) {
            last_generation_stage_ = "kv_cache_reset_failed";
            last_error_ = "KV cache reset failed: " + status.message;
            return status;
        }
    }
    context_.token_ids = prompt_tokens;
    context_.past_length = 0;
    context_.sequence_length = prompt_tokens.size();
    context_.prefill = true;
    last_generation_stage_ = "prefill";
    const auto prefill_dispatch_started = std::chrono::steady_clock::now();
    status = run_model_step(prompt_tokens.size());
    const auto prefill_dispatch_finished = std::chrono::steady_clock::now();
    last_prefill_ms_ = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        prefill_dispatch_finished - prefill_dispatch_started).count());
    last_prefill_timing_valid_ = status.ok();
    if (!status.ok()) {
        last_generation_stage_ = "prefill_dispatch_failed";
        last_error_ = "Prefill dispatch failed: " + status.message;
        return status;
    }

    std::vector<float> logits;
    const auto prefill_readback_started = std::chrono::steady_clock::now();
    status = read_last_token_logits(logits);
    const auto prefill_readback_finished = std::chrono::steady_clock::now();
    last_prefill_readback_ms_ = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        prefill_readback_finished - prefill_readback_started).count());
    last_prefill_readback_timing_valid_ = status.ok();
    if (!status.ok()) {
        last_generation_stage_ = "prefill_logits_readback_failed";
        last_error_ = "Prefill logits readback failed: " + status.message;
        return status;
    }
    last_generation_stage_ = "first_token_sampling";

    Sampler sampler(sampling);
    std::vector<int32_t> history;
    history.reserve(prompt_tokens.size() + max_new_tokens);
    for (uint32_t token : prompt_tokens) {
        if (token > static_cast<uint32_t>(INT32_MAX) ||
            static_cast<uint64_t>(token) >= plan_.vocabulary_size) {
            last_error_ = "Tokenizer returned an ID outside the sampler's validated token range";
            return {StatusCode::InvalidArgument, last_error_};
        }
        history.push_back(static_cast<int32_t>(token));
    }
    StopController stop(static_cast<int32_t>(loader_->reader()->info().tokenizer.eos_token_id));
    std::vector<uint32_t> generated;
    generated.reserve(max_new_tokens);
    std::string previous_text;
    std::chrono::nanoseconds decode_compute_elapsed{0};
    bool first_token_ready = false;
    for (uint32_t index = 0; index < max_new_tokens; ++index) {
        if (cancelled_.load(std::memory_order_relaxed))
            return {StatusCode::Cancelled, "GPU generation was cancelled"};
        int32_t sampled = -1;
        status = sampler.sample(logits, history, sampled);
        if (!status.ok() || sampled < 0 ||
            static_cast<uint64_t>(sampled) >= plan_.vocabulary_size) {
            last_generation_stage_ = "sampling_failed";
            last_error_ = status.ok() ? "Sampler returned a token outside the model vocabulary" : status.message;
            return {status.ok() ? StatusCode::InvalidArgument : status.code, last_error_};
        }
        history.push_back(sampled);
        generated.push_back(static_cast<uint32_t>(sampled));
        last_output_tokens_ = generated.size();
        if (!first_token_ready) {
            const auto first_token_at = std::chrono::steady_clock::now();
            last_ttft_ms_ = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                first_token_at - request_started).count());
            last_ttft_timing_valid_ = true;
            first_token_ready = true;
            last_generation_stage_ = "decode";
        }
        const bool should_stop = stop.push(sampled);

        std::string decoded;
        status = tokenizer_engine.decode(loader_->reader()->info().tokenizer, generated, decoded);
        if (!status.ok()) {
            last_generation_stage_ = "token_decode_failed";
            last_error_ = "Generated token decoding failed: " + status.message;
            return status;
        }
        if (on_text && decoded.size() >= previous_text.size() &&
            decoded.compare(0, previous_text.size(), previous_text) == 0) {
            const std::string delta = decoded.substr(previous_text.size());
            if (!delta.empty()) on_text(delta);
        } else if (on_text && previous_text.empty() && !decoded.empty()) {
            on_text(decoded);
        }
        previous_text = std::move(decoded);
        if (should_stop || index + 1 == max_new_tokens) break;

        context_.token_ids = {static_cast<uint32_t>(sampled)};
        context_.past_length = prompt_tokens.size() + generated.size() - 1;
        context_.sequence_length = 1;
        context_.prefill = false;
        const auto decode_dispatch_started = std::chrono::steady_clock::now();
        status = run_model_step(1);
        if (!status.ok()) {
            last_decode_ms_ = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::milliseconds>(decode_compute_elapsed).count());
            last_generation_stage_ = "decode_dispatch_failed";
            last_error_ = "Decode dispatch failed: " + status.message;
            return status;
        }
        status = read_last_token_logits(logits);
        const auto decode_readback_finished = std::chrono::steady_clock::now();
        if (!status.ok()) {
            last_decode_ms_ = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::milliseconds>(decode_compute_elapsed +
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - decode_dispatch_started)).count());
            last_generation_stage_ = "decode_logits_readback_failed";
            last_error_ = "Decode logits readback failed: " + status.message;
            return status;
        }
        decode_compute_elapsed += std::chrono::duration_cast<std::chrono::nanoseconds>(
            decode_readback_finished - decode_dispatch_started);
    }

    last_decode_ms_ = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(decode_compute_elapsed).count());
    const uint64_t decode_tokens = generated.size() > 1 ? generated.size() - 1 : 0;
    last_decode_timing_valid_ = decode_tokens != 0;
    last_output_tokens_ = generated.size();
    last_decode_tokens_per_second_ = last_decode_ms_ && decode_tokens
        ? static_cast<double>(decode_tokens) * 1000.0 / static_cast<double>(last_decode_ms_) : 0.0;
    last_generation_completed_ = true;
    last_generation_stage_ = "completed";
    last_error_.clear();
    return Status::success();
}

Status GpuInferenceSession::count_tokens(const std::string& text, uint32_t& out_count) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!loaded_ || !loader_ || !tokenizers_)
        return {StatusCode::NotReady, "GPU inference session is not loaded"};
    TokenizerEngine tokenizer_engine(*tokenizers_);
    std::vector<uint32_t> tokens;
    Status status = tokenizer_engine.encode(loader_->reader()->info().tokenizer, text, tokens);
    if (!status.ok()) return status;
    if (tokens.size() > UINT32_MAX) return {StatusCode::InvalidArgument, "Token count exceeds uint32"};
    out_count = static_cast<uint32_t>(tokens.size());
    return Status::success();
}

std::string GpuInferenceSession::diagnostics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream report;
    const DeviceInfo* device = engine_ ? engine_->active_device() : nullptr;
    const std::string backend_display = device ? backend_name() :
        (last_attempt_backend_.empty() ? "Unknown" : last_attempt_backend_);
    const std::string gpu_vendor = device && !device->vendor.empty() ? device->vendor :
        (last_attempt_gpu_vendor_.empty() ? "unknown" : last_attempt_gpu_vendor_);
    const std::string gpu_device = device && !device->name.empty() ? device->name :
        (last_attempt_gpu_device_.empty() ? "unknown" : last_attempt_gpu_device_);
    const std::string gpu_architecture = device && !device->architecture.empty() ? device->architecture :
        (last_attempt_gpu_architecture_.empty() ? "unknown" : last_attempt_gpu_architecture_);
    const std::string gpu_driver = device && !device->driver.empty() ? device->driver :
        (last_attempt_gpu_driver_.empty() ? "unknown" : last_attempt_gpu_driver_);
    const std::string gpu_api = device && !device->api_version.empty() ? device->api_version :
        (last_attempt_gpu_api_.empty() ? "unknown" : last_attempt_gpu_api_);
    const auto& gpu_features = device ? device->features : last_attempt_gpu_features_;
    report << "runtime=AndroidGpuInferenceEngine"
           << ";status=" << (loaded_ ? "READY" : "UNLOADED")
           << ";runtime_version=0.1.0"
           << ";backend=" << backend_display
           << ";backend_fallback_reason=" << (backend_fallback_reason_.empty() ? "none" : backend_fallback_reason_)
           << ";gpu_vendor=" << gpu_vendor
           << ";gpu_device=" << gpu_device
           << ";gpu_architecture=" << gpu_architecture
           << ";gpu_driver=" << gpu_driver
           << ";gpu_api=" << gpu_api
           << ";gpu_compute_available=" << ((device ? device->compute_available : last_attempt_gpu_compute_available_) ? "true" : "false")
           << ";gpu_self_test=" << ((device ? device->compute_self_test_passed : last_attempt_gpu_self_test_passed_) ? "passed" : "unknown")
           << ";gpu_max_buffer_bytes=" << (device ? device->limits.max_buffer_size : last_attempt_gpu_max_buffer_bytes_)
           << ";gpu_max_storage_binding_bytes=" << (device ? device->limits.max_storage_buffer_binding_size : last_attempt_gpu_max_storage_binding_bytes_)
           << ";gpu_max_workgroup_invocations=" << (device ? device->limits.max_compute_workgroup_invocations : last_attempt_gpu_max_workgroup_invocations_)
           << ";architecture=" << (!plan_.architecture.empty() ? plan_.architecture :
                (last_attempt_architecture_.empty() ? "unknown" : last_attempt_architecture_))
           << ";device=" << gpu_device
           << ";model=" << (!model_path_.empty() ? model_path_ :
                (last_attempt_model_path_.empty() ? "unknown" : last_attempt_model_path_))
           << ";weight_bytes=" << (total_weight_bytes_ != 0 ? total_weight_bytes_ : last_attempt_weight_bytes_)
           << ";largest_weight_tensor_bytes=" << (largest_weight_tensor_bytes_ != 0 ? largest_weight_tensor_bytes_ : last_attempt_largest_weight_tensor_bytes_)
           << ";peak_operation_weight_bytes=" << (peak_operation_weight_bytes_ != 0 ? peak_operation_weight_bytes_ : last_attempt_peak_operation_weight_bytes_)
           << ";kv_cache_bytes=" << (kv_cache_bytes_ != 0 ? kv_cache_bytes_ : last_attempt_kv_cache_bytes_)
           << ";memory_budget_bytes=" << (memory_budget_bytes_ != 0
                ? memory_budget_bytes_ : (last_failed_memory_budget_bytes_ != 0
                    ? last_failed_memory_budget_bytes_ : last_attempt_memory_budget_bytes_))
           << ";memory_budget_kind=application_allocation_cap_not_vram"
           << ";memory_active_bytes=" << (memory_ ? std::to_string(memory_->allocated_bytes()) : "unknown")
           << ";memory_available_bytes=" << (memory_ ? std::to_string(memory_->available_bytes()) : "unknown")
           << ";memory_accounting_initialized=" << (memory_ ? "true" : "false")
           << ";memory_required_bytes=" << (memory_estimate_.plan.required_bytes != 0
                ? memory_estimate_.plan.required_bytes : last_failed_memory_required_bytes_)
           << ";memory_usable_budget_bytes=" << (memory_estimate_.plan.usable_budget_bytes != 0
                ? memory_estimate_.plan.usable_budget_bytes : last_failed_memory_usable_bytes_)
           << ";memory_estimated_kv_cache_bytes=" << (memory_estimate_.estimated_kv_cache_bytes != 0
                ? memory_estimate_.estimated_kv_cache_bytes : last_failed_memory_kv_bytes_)
           << ";memory_estimated_scratch_bytes=" << (memory_estimate_.estimated_scratch_bytes != 0
                ? memory_estimate_.estimated_scratch_bytes : last_failed_memory_scratch_bytes_)
           << ";last_failed_memory_budget_bytes=" << last_failed_memory_budget_bytes_
           << ";last_failed_tensor_bytes=" << last_failed_tensor_bytes_
           << ";last_failed_storage_binding_limit_bytes=" << last_failed_storage_binding_limit_bytes_
           << ";gpu_memory_total=unknown"
           << ";gpu_memory_free=unknown"
           << ";weight_streaming=operation-aware-chunked"
           << ";single_tensor_chunking=GGUF_BLOCK_AWARE_MATMUL_CHANNELS_AND_EMBEDDING_TOKEN_ROWS"
           << ";chunked_operator_support=F32_ELEMENTWISE,GGUF_BLOCK_MATMUL,GGUF_BLOCK_EMBEDDING"
           << ";backend_kernel_validation=runtime-numerical"
           << ";cross_backend_quantization_validation=" << cross_backend_quantization_validation_
           << ";model_reference_parity=UNVERIFIED"
           << ";model_graph_validation=" << last_model_graph_validation_
           << ";context_length=" << (context_length_ != 0 ? context_length_ : last_attempt_context_length_)
           << ";tokenization_ms=" << last_tokenization_ms_
           << ";tokenization_timing_valid=" << (last_tokenization_timing_valid_ ? "true" : "false")
           << ";prefill_ms=" << (last_prefill_ms_ + last_prefill_readback_ms_)
           << ";prefill_timing_valid=" << (last_prefill_timing_valid_ && last_prefill_readback_timing_valid_ ? "true" : "false")
           << ";prefill_compute_ms=" << last_prefill_ms_
           << ";prefill_compute_timing_valid=" << (last_prefill_timing_valid_ ? "true" : "false")
           << ";prefill_readback_ms=" << last_prefill_readback_ms_
           << ";prefill_readback_timing_valid=" << (last_prefill_readback_timing_valid_ ? "true" : "false")
           << ";ttft_ms=" << last_ttft_ms_
           << ";ttft_timing_valid=" << (last_ttft_timing_valid_ ? "true" : "false")
           << ";decode_ms=" << last_decode_ms_
           << ";decode_timing_valid=" << (last_decode_timing_valid_ ? "true" : "false")
           << ";generation_stage=" << last_generation_stage_
           << ";prompt_tokens=" << last_prompt_tokens_
           << ";output_tokens=" << last_output_tokens_
           << ";decode_tokens_per_second=" << last_decode_tokens_per_second_
           << ";generation_result=" << (last_generation_completed_ ? "COMPLETED" : "NOT_COMPLETED")
           << ";cpu_fallback_available=false"
           << ";gpu_feature_count=" << gpu_features.size()
           << ";gpu_features=";
    for (size_t i = 0; i < gpu_features.size(); ++i) {
        if (i != 0) report << ",";
        report << gpu_features[i];
    }
    report << ";last_error=" << (last_error_.empty() ? "none" : last_error_);
    return report.str();
}

} // namespace ai_chat::gpu_engine
