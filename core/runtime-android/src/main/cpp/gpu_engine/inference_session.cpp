#include "inference_session.h"

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
    last_ttft_ms_ = 0;
    last_decode_tokens_per_second_ = 0.0;
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

std::string GpuInferenceSession::backend_name() const {
    return engine_ ? backend_label(engine_->diagnostics().selected_backend) : "Unknown";
}

Status GpuInferenceSession::load_model(const std::string& path, uint32_t context_length,
                                       uint64_t memory_budget_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    unload_locked();
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
    const GgufReader& reader = *loader_->reader();
    const GgufFileInfo& model = reader.info();
    BackendKind backend_kind = engine_->active_backend()->kind();

    const auto tokenizer_preflight = evaluate_tokenizer_preflight(model.tokenizer, *tokenizers_);
    if (!tokenizer_preflight.ready) {
        last_error_ = "Tokenizer is not validated for this GGUF model";
        for (const auto& blocker : tokenizer_preflight.blockers) last_error_ += " | " + blocker;
        unload_locked();
        return {StatusCode::Unsupported, last_error_};
    }
    const auto architecture = architectures_->find(model.architecture);
    if (!architecture) {
        last_error_ = "Unsupported model architecture: " + model.architecture;
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
        last_error_ = "Weight type is not implemented by the current GPU loader: " +
                      tensor->type.name + " (" + name + ")";
        unload_locked();
        return {StatusCode::Unsupported, last_error_};
    }

    // We load only the current operation's weights, but all weights attached
    // to that operation coexist until its dispatch completes. Estimate that
    // peak set rather than only the largest individual tensor.
    peak_operation_weight_bytes_ = 0;
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
            const uint64_t padded_bytes = (tensor->byte_size + 3) & ~uint64_t{3};
            uint64_t next_operation_bytes = 0;
            if (add_overflow(operation_weight_bytes, padded_bytes, next_operation_bytes)) {
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
            if (tensor->type.name != "F32" &&
                !compatible_q4_0 && !compatible_q4_1 && !compatible_q8_0 &&
                !compatible_q5_0 && !compatible_q5_1) {
                last_error_ = "Quantized weight is attached to an operator without a matching packed kernel: " +
                              operation.operator_name + ":" + name;
                unload_locked();
                return {StatusCode::Unsupported, last_error_};
            }
        }
    }

    auto validate_quantization_kernels_for_active_backend = [&]() -> Status {
        const auto has_operator = [&](const std::string& name) {
            return std::any_of(plan_.operations.begin(), plan_.operations.end(),
                [&](const ModelPlanOperation& operation) { return operation.operator_name == name; });
        };
        for (uint32_t type_id : quantization_types) {
            Status validation_status = Status::success();
            std::string type_name;
            std::string matmul_name;
            std::string embedding_name;
            if (type_id == 2) { type_name="Q4_0"; matmul_name="MatMul_Q4_0"; embedding_name="Embedding_Q4_0"; }
            else if (type_id == 3) { type_name="Q4_1"; matmul_name="MatMul_Q4_1"; embedding_name="Embedding_Q4_1"; }
            else if (type_id == 6) { type_name="Q5_0"; matmul_name="MatMul_Q5_0"; embedding_name="Embedding_Q5_0"; }
            else if (type_id == 7) { type_name="Q5_1"; matmul_name="MatMul_Q5_1"; embedding_name="Embedding_Q5_1"; }
            else if (type_id == 8) { type_name="Q8_0"; matmul_name="MatMul_Q8_0"; embedding_name="Embedding_Q8_0"; }
            else return {StatusCode::Unsupported,"No GPU validation route exists for GGUF quantization type "+std::to_string(type_id)};

            if (has_operator(matmul_name) || type_id == 2 || type_id == 3 || type_id == 8) {
                if (type_id == 2) validation_status=validate_q4_0_matmul_kernel(*engine_->active_backend(),*operators_,*quantizations_);
                else if (type_id == 3) validation_status=validate_q4_1_matmul_kernel(*engine_->active_backend(),*operators_,*quantizations_);
                else if (type_id == 6) validation_status=validate_q5_0_matmul_kernel(*engine_->active_backend(),*operators_,*quantizations_);
                else if (type_id == 7) validation_status=validate_q5_1_matmul_kernel(*engine_->active_backend(),*operators_,*quantizations_);
                else validation_status=validate_q8_0_matmul_kernel(*engine_->active_backend(),*operators_,*quantizations_);
            }
            if (validation_status.ok() && has_operator(embedding_name)) {
                if (type_id == 2) validation_status=validate_q4_0_embedding_kernel(*engine_->active_backend(),*operators_);
                else if (type_id == 3) validation_status=validate_q4_1_embedding_kernel(*engine_->active_backend(),*operators_);
                else if (type_id == 6) validation_status=validate_q5_0_embedding_kernel(*engine_->active_backend(),*operators_,*quantizations_);
                else if (type_id == 7) validation_status=validate_q5_1_embedding_kernel(*engine_->active_backend(),*operators_,*quantizations_);
                else validation_status=validate_q8_0_embedding_kernel(*engine_->active_backend(),*operators_);
            }
            if (!validation_status.ok()) {
                last_error_ = "Quantized GPU kernel validation failed for " + type_name +
                              " on the active backend: " + validation_status.message;
                return validation_status;
            }
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
    const uint64_t max_single_buffer = std::min(device->limits.max_buffer_size,
                                                device->limits.max_storage_buffer_binding_size);
    const uint64_t padded_largest_weight_bytes =
        largest_weight_tensor_bytes_ > UINT64_MAX - 3 ? UINT64_MAX : (largest_weight_tensor_bytes_ + 3) & ~uint64_t{3};
    if (padded_largest_weight_bytes > max_single_buffer) {
        last_error_ = "A single model tensor exceeds the backend storage-buffer limit; multi-chunk operator execution is not implemented";
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
    estimate_input.token_count = context_length_;
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
        unload_locked();
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
        last_error_ = status.message;
        return status;
    }
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

Status GpuInferenceSession::generate(const std::string& prompt, uint32_t max_new_tokens,
                                     SamplingConfig sampling, const TextCallback& on_text) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!loaded_ || !loader_ || !tokenizers_ || !kv_cache_)
        return {StatusCode::NotReady, "GPU inference session is not loaded"};
    if (prompt.empty() || max_new_tokens == 0)
        return {StatusCode::InvalidArgument, "Prompt and max_new_tokens must be non-empty"};
    if (cancelled_.load(std::memory_order_relaxed))
        return {StatusCode::Cancelled, "GPU generation was cancelled before it started"};

    TokenizerEngine tokenizer_engine(*tokenizers_);
    std::vector<uint32_t> prompt_tokens;
    Status status = tokenizer_engine.encode(loader_->reader()->info().tokenizer, prompt, prompt_tokens);
    if (!status.ok()) {
        last_error_ = "Prompt tokenization failed: " + status.message;
        return status;
    }
    if (prompt_tokens.empty() || prompt_tokens.size() > context_length_ ||
        max_new_tokens > context_length_ - prompt_tokens.size()) {
        last_error_ = "Prompt plus requested generation exceeds the validated context length";
        return {StatusCode::InvalidArgument, last_error_};
    }

    for (uint32_t layer = 0; layer < plan_.layer_count; ++layer) {
        status = kv_cache_->reset_layer(layer);
        if (!status.ok()) {
            last_error_ = "KV cache reset failed: " + status.message;
            return status;
        }
    }
    context_.token_ids = prompt_tokens;
    context_.past_length = 0;
    context_.sequence_length = prompt_tokens.size();
    context_.prefill = true;
    const auto generation_started = std::chrono::steady_clock::now();
    status = run_model_step(prompt_tokens.size());
    if (!status.ok()) return status;

    std::vector<float> logits;
    status = read_last_token_logits(logits);
    if (!status.ok()) {
        last_error_ = "Prefill logits readback failed: " + status.message;
        return status;
    }
    const auto first_logits_ready = std::chrono::steady_clock::now();
    last_ttft_ms_ = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        first_logits_ready - generation_started).count());

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
    const auto decode_started = std::chrono::steady_clock::now();

    for (uint32_t index = 0; index < max_new_tokens; ++index) {
        if (cancelled_.load(std::memory_order_relaxed))
            return {StatusCode::Cancelled, "GPU generation was cancelled"};
        int32_t sampled = -1;
        status = sampler.sample(logits, history, sampled);
        if (!status.ok() || sampled < 0 ||
            static_cast<uint64_t>(sampled) >= plan_.vocabulary_size) {
            last_error_ = status.ok() ? "Sampler returned a token outside the model vocabulary" : status.message;
            return {status.ok() ? StatusCode::InvalidArgument : status.code, last_error_};
        }
        history.push_back(sampled);
        generated.push_back(static_cast<uint32_t>(sampled));
        const bool should_stop = stop.push(sampled);

        std::string decoded;
        status = tokenizer_engine.decode(loader_->reader()->info().tokenizer, generated, decoded);
        if (!status.ok()) {
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
        status = run_model_step(1);
        if (!status.ok()) return status;
        status = read_last_token_logits(logits);
        if (!status.ok()) {
            last_error_ = "Decode logits readback failed: " + status.message;
            return status;
        }
    }

    const auto generation_finished = std::chrono::steady_clock::now();
    const uint64_t decode_ms = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        generation_finished - decode_started).count());
    const uint64_t decode_tokens = generated.size() > 1 ? generated.size() - 1 : 0;
    last_decode_tokens_per_second_ = decode_ms && decode_tokens
        ? static_cast<double>(decode_tokens) * 1000.0 / static_cast<double>(decode_ms) : 0.0;
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
    report << "runtime=AIChatGpuEngine"
           << ";status=" << (loaded_ ? "READY" : "UNLOADED")
           << ";runtime_version=0.1.0"
           << ";backend=" << backend_name()
           << ";backend_fallback_reason=" << (backend_fallback_reason_.empty() ? "none" : backend_fallback_reason_)
           << ";gpu_vendor=" << (device && !device->vendor.empty() ? device->vendor : "unknown")
           << ";gpu_device=" << (device && !device->name.empty() ? device->name : "unknown")
           << ";gpu_architecture=" << (device && !device->architecture.empty() ? device->architecture : "unknown")
           << ";gpu_driver=" << (device && !device->driver.empty() ? device->driver : "unknown")
           << ";gpu_api=" << (device && !device->api_version.empty() ? device->api_version : "unknown")
           << ";gpu_compute_available=" << (device && device->compute_available ? "true" : "false")
           << ";gpu_self_test=" << (device && device->compute_self_test_passed ? "passed" : "unknown")
           << ";gpu_max_buffer_bytes=" << (device ? device->limits.max_buffer_size : 0)
           << ";gpu_max_storage_binding_bytes=" << (device ? device->limits.max_storage_buffer_binding_size : 0)
           << ";gpu_max_workgroup_invocations=" << (device ? device->limits.max_compute_workgroup_invocations : 0)
           << ";architecture=" << (plan_.architecture.empty() ? "unknown" : plan_.architecture)
           << ";device=" << (device && !device->name.empty() ? device->name : "unknown")
           << ";model=" << (model_path_.empty() ? "unknown" : model_path_)
           << ";weight_bytes=" << total_weight_bytes_
           << ";largest_weight_tensor_bytes=" << largest_weight_tensor_bytes_
           << ";peak_operation_weight_bytes=" << peak_operation_weight_bytes_
           << ";kv_cache_bytes=" << kv_cache_bytes_
           << ";memory_budget_bytes=" << memory_budget_bytes_
           << ";memory_active_bytes=" << (memory_ ? memory_->allocated_bytes() : 0)
           << ";memory_available_bytes=" << (memory_ ? memory_->available_bytes() : 0)
           << ";memory_required_bytes=" << memory_estimate_.plan.required_bytes
           << ";gpu_memory_total=unknown"
           << ";gpu_memory_free=unknown"
           << ";weight_streaming=per-operation"
           << ";single_tensor_chunking=UNSUPPORTED"
           << ";backend_kernel_validation=runtime-numerical"
           << ";model_reference_parity=UNVERIFIED"
           << ";model_graph_validation=UNVERIFIED"
           << ";context_length=" << context_length_
           << ";ttft_ms=" << last_ttft_ms_
           << ";decode_tokens_per_second=" << last_decode_tokens_per_second_
           << ";cpu_fallback_available=true"
           << ";gpu_feature_count=" << (device ? device->features.size() : 0)
           << ";gpu_features=";
    if (device) {
        for (size_t i = 0; i < device->features.size(); ++i) {
            if (i != 0) report << ",";
            report << device->features[i];
        }
    }
    report << ";last_error=" << (last_error_.empty() ? "none" : last_error_);
    return report.str();
}

} // namespace ai_chat::gpu_engine
