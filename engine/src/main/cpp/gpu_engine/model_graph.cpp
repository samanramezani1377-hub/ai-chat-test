#include "model_graph.h"
#include "kv_cache.h"
#include "execution_engine.h"
#include "memory_planner.h"
#include "weight_streamer.h"
#include "quantization.h"

#include <algorithm>
#include <unordered_set>
#include <utility>

namespace ai_chat::gpu_engine {

namespace {

void append_report_items(std::string& message, const char* label,
                         const std::vector<std::string>& items) {
    constexpr size_t kMaximumItems = 12;
    for (size_t index = 0; index < items.size() && index < kMaximumItems; ++index) {
        message += " | ";
        message += label;
        message += ":";
        message += items[index];
    }
    if (items.size() > kMaximumItems) {
        message += " | ";
        message += label;
        message += ":...(" + std::to_string(items.size() - kMaximumItems) + " more)";
    }
}

std::string describe_compile_failure(const ModelGraphCompileReport& report) {
    std::string message = "Model execution plan could not be fully lowered into a validated compute graph";
    message += " | compiled_nodes=" + std::to_string(report.compiled_nodes);
    append_report_items(message, "missing_lowering", report.missing_lowerings);
    append_report_items(message, "unresolved_input", report.unresolved_inputs);
    append_report_items(message, "missing_weight", report.missing_weight_tensors);
    append_report_items(message, "error", report.errors);
    return message;
}



Status execute_chunked_q6_k_matmul_if_needed(
        const ModelPlanOperation& operation, Backend& backend,
        const OperatorRegistry& operators, const GgufReader& reader,
        MemoryManager& memory, const DeviceInfo& device,
        ModelExecutionContext& context, bool& handled) {
    handled = false;
    if (operation.operator_name != "MatMul_Q6_K") return Status::success();
    if (operation.inputs.size() != 1 || operation.weight_tensors.size() != 1)
        return {StatusCode::InvalidArgument, "Chunked Q6_K MatMul requires one activation and one packed weight"};
    const auto weight = context.quantized_weights.find(operation.weight_tensors.front());
    if (weight == context.quantized_weights.end() || weight->second.file_type_id != 14)
        return {StatusCode::InvalidArgument, "Chunked Q6_K MatMul metadata binding is missing"};
    if (weight->second.buffer) return Status::success();
    handled = true;

    const Tensor* input = nullptr;
    const Tensor* output = nullptr;
    const auto input_it = context.tensors.find(operation.inputs.front());
    const auto output_it = context.tensors.find(operation.output);
    if (input_it == context.tensors.end() || output_it == context.tensors.end())
        return {StatusCode::InvalidArgument, "Chunked Q6_K MatMul activation or output is unbound"};
    input = &input_it->second;
    output = &output_it->second;
    if (input->descriptor().data_type != DataType::F32 ||
        output->descriptor().data_type != DataType::F32 ||
        input->descriptor().shape.size() != 2 ||
        (output->descriptor().shape.size() != 2 && output->descriptor().shape.size() != 3) ||
        weight->second.dimensions.size() != 2 ||
        input->descriptor().shape[1] != weight->second.dimensions[0])
        return {StatusCode::Unsupported, "Chunked Q6_K MatMul tensor shapes are incompatible"};

    const GgufTensorInfo* source = reader.find_tensor(operation.weight_tensors.front());
    if (!source || !source->byte_size_known || source->byte_size != weight->second.byte_size)
        return {StatusCode::InvalidArgument, "Chunked Q6_K MatMul GGUF source metadata is invalid"};
    const uint64_t inner = weight->second.dimensions[0];
    const uint64_t total_channels = weight->second.dimensions[1];
    if (inner == 0 || inner % 256 != 0 || total_channels == 0 ||
        total_channels > UINT32_MAX || input->descriptor().shape[0] == 0 ||
        output->element_count() / input->descriptor().shape[0] != total_channels)
        return {StatusCode::InvalidArgument, "Chunked Q6_K MatMul dimensions are not block/channel aligned"};
    const uint64_t row_bytes = (inner / 256) * 210;
    const uint64_t binding_limit = std::min(device.limits.max_storage_buffer_binding_size,
        device.limits.max_buffer_size == 0 ? device.limits.max_storage_buffer_binding_size
                                           : device.limits.max_buffer_size);
    if (row_bytes == 0 || binding_limit < row_bytes)
        return {StatusCode::Unsupported, "One Q6_K output-channel weight row exceeds the backend storage binding limit"};
    const uint64_t maximum_chunk_channels = binding_limit / row_bytes;
    if (maximum_chunk_channels == 0)
        return {StatusCode::Unsupported, "Backend cannot hold one complete Q6_K output-channel row"};

    ChunkExecutionConfig config;
    config.total_elements = total_channels;
    config.maximum_chunk_elements = maximum_chunk_channels;
    config.alignment_elements = 1;
    config.element_size_bytes = sizeof(float);
    config.output_base_offset_bytes = output->byte_offset();
    config.output_binding = 2;
    config.output_binding_is_chunk_slice = false;
    config.output_buffer = output->buffer();

    MemoryLease current_weight_lease;
    std::vector<uint8_t> staging;
    const auto build_dispatch = [&](const ChunkExecutionRange& range, ComputeDispatch& dispatch) -> Status {
        if (range.element_offset > UINT64_MAX / row_bytes ||
            range.element_count > UINT64_MAX / row_bytes)
            return {StatusCode::InvalidArgument, "Q6_K output-channel chunk byte range overflows"};
        const uint64_t source_offset = range.element_offset * row_bytes;
        const uint64_t chunk_bytes = range.element_count * row_bytes;
        if (source_offset > source->byte_size || chunk_bytes > source->byte_size - source_offset)
            return {StatusCode::InvalidArgument, "Q6_K output-channel chunk exceeds the GGUF tensor"};
        if (chunk_bytes > binding_limit || chunk_bytes > UINT64_MAX - 3)
            return {StatusCode::InvalidArgument, "Q6_K output-channel chunk exceeds backend binding capacity"};
        const uint64_t allocation_bytes = (chunk_bytes + 3) & ~uint64_t{3};

        current_weight_lease = MemoryLease{};
        BufferDesc allocation;
        allocation.size_bytes = allocation_bytes;
        allocation.alignment = std::max<uint64_t>(4, device.limits.min_storage_buffer_offset_alignment);
        allocation.memory = MemoryDomain::Device;
        allocation.persistent = false;
        allocation.label = "q6-k-matmul-channel-chunk:" + source->name;
        Status status = memory.allocate(allocation, current_weight_lease);
        if (!status.ok()) return status;
        if (!current_weight_lease || !current_weight_lease.buffer())
            return {StatusCode::BackendFailure, "Q6_K channel-chunk allocation returned an empty buffer"};

        staging.assign(static_cast<size_t>(allocation_bytes), 0);
        status = reader.read_tensor_chunk(*source, source_offset, staging.data(), chunk_bytes);
        if (!status.ok()) return status;
        status = backend.upload(current_weight_lease.buffer(), 0, staging.data(), allocation_bytes);
        if (!status.ok()) return status;

        if (backend.kind() == BackendKind::OpenGles)
            status = build_q6_k_matmul_chunk_gles_candidate(*input, current_weight_lease.buffer(),
                chunk_bytes, *output, static_cast<uint32_t>(range.element_offset),
                static_cast<uint32_t>(range.element_count), dispatch);
        else
            status = build_q6_k_matmul_chunk_candidate(*input, current_weight_lease.buffer(),
                chunk_bytes, *output, static_cast<uint32_t>(range.element_offset),
                static_cast<uint32_t>(range.element_count), dispatch);
        if (!status.ok()) return status;
        return operators.authorize_dispatch(dispatch.kernel_id, backend.kind(), dispatch);
    };
    return execute_chunked_compute(backend, config, build_dispatch, {});
}

bool is_chunkable_elementwise(const ModelPlanOperation& operation) {
    if (!operation.weight_tensors.empty() || operation.side_effect) return false;
    const auto& name = operation.operator_name;
    return name == "SiLU" || name == "Sigmoid" || name == "Softplus" ||
           name == "Add" || name == "Multiply";
}

Status execute_chunked_elementwise_if_needed(
        const ModelPlanOperation& operation, Backend& backend,
        const ModelLoweringRegistry& lowerings, const OperatorRegistry& operators,
        const DeviceInfo& device, ModelExecutionContext& context, bool& handled) {
    handled = false;
    if (!is_chunkable_elementwise(operation)) return Status::success();
    const size_t expected_inputs =
        (operation.operator_name == "Add" || operation.operator_name == "Multiply") ? 2u : 1u;
    if (operation.inputs.size() != expected_inputs)
        return {StatusCode::InvalidArgument, "Chunkable elementwise operation has an unexpected input count"};
    const auto output_it = context.tensors.find(operation.output);
    if (output_it == context.tensors.end())
        return {StatusCode::InvalidArgument, "Chunkable elementwise output binding is missing"};
    const Tensor& output = output_it->second;
    if (output.descriptor().data_type != DataType::F32 ||
        output.descriptor().layout != TensorLayout::Contiguous || !output.buffer())
        return {StatusCode::Unsupported, "Chunked elementwise execution currently requires contiguous F32 output"};

    std::vector<const Tensor*> inputs;
    inputs.reserve(expected_inputs);
    bool oversized = output.byte_size() > device.limits.max_storage_buffer_binding_size;
    for (const auto& name : operation.inputs) {
        const auto input_it = context.tensors.find(name);
        if (input_it == context.tensors.end())
            return {StatusCode::InvalidArgument, "Chunkable elementwise input binding is missing: " + name};
        const Tensor& input = input_it->second;
        if (input.descriptor().data_type != DataType::F32 ||
            input.descriptor().layout != TensorLayout::Contiguous || !input.buffer() ||
            input.element_count() != output.element_count() ||
            input.descriptor().shape != output.descriptor().shape)
            return {StatusCode::Unsupported, "Chunked elementwise inputs must match the contiguous F32 output shape"};
        oversized = oversized || input.byte_size() > device.limits.max_storage_buffer_binding_size;
        inputs.push_back(&input);
    }
    if (!oversized) return Status::success();
    handled = true;

    const uint64_t binding_limit = device.limits.max_storage_buffer_binding_size;
    if (binding_limit < sizeof(float))
        return {StatusCode::Unsupported, "Backend storage binding limit is smaller than one F32 element"};
    const uint64_t alignment_bytes = std::max<uint64_t>(4, device.limits.min_storage_buffer_offset_alignment);
    const uint64_t alignment_elements = (alignment_bytes + sizeof(float) - 1) / sizeof(float);
    const uint64_t maximum_chunk_elements = binding_limit / sizeof(float);
    if (maximum_chunk_elements < alignment_elements)
        return {StatusCode::Unsupported, "Backend binding limit cannot hold one aligned F32 chunk"};

    const auto* lowering = lowerings.find(backend.kind(), operation.operator_name);
    if (!lowering)
        return {StatusCode::Unsupported, "No backend lowering exists for chunked elementwise operation: " + operation.operator_name};

    ChunkExecutionConfig config;
    config.total_elements = output.element_count();
    config.maximum_chunk_elements = maximum_chunk_elements;
    config.alignment_elements = alignment_elements;
    config.element_size_bytes = sizeof(float);
    config.output_base_offset_bytes = output.byte_offset();
    config.output_binding = expected_inputs == 1 ? 1u : 2u;
    config.output_buffer = output.buffer();

    const auto build_dispatch = [&](const ChunkExecutionRange& range, ComputeDispatch& dispatch) -> Status {
        if (range.byte_offset > UINT64_MAX - output.byte_offset())
            return {StatusCode::InvalidArgument, "Chunked output byte offset overflows"};
        ModelExecutionContext chunk_context = context;
        for (size_t index = 0; index < inputs.size(); ++index) {
            const Tensor& source = *inputs[index];
            if (range.byte_offset > UINT64_MAX - source.byte_offset())
                return {StatusCode::InvalidArgument, "Chunked input byte offset overflows"};
            TensorDesc descriptor = source.descriptor();
            descriptor.shape = {range.element_count};
            descriptor.strides_bytes.clear();
            Tensor view;
            Status result = Tensor::create(std::move(descriptor), source.buffer(),
                source.byte_offset() + range.byte_offset, view);
            if (!result.ok()) return result;
            chunk_context.tensors[operation.inputs[index]] = std::move(view);
        }
        TensorDesc output_descriptor = output.descriptor();
        output_descriptor.shape = {range.element_count};
        output_descriptor.strides_bytes.clear();
        Tensor output_view;
        Status result = Tensor::create(std::move(output_descriptor), output.buffer(),
            output.byte_offset() + range.byte_offset, output_view);
        if (!result.ok()) return result;
        chunk_context.tensors[operation.output] = std::move(output_view);

        result = (*lowering)(operation, chunk_context, dispatch);
        if (!result.ok()) return result;
        return operators.authorize_dispatch(dispatch.kernel_id, backend.kind(), dispatch);
    };
    return execute_chunked_compute(backend, config, build_dispatch, {});
}

} // namespace

std::string ModelLoweringRegistry::make_key(BackendKind backend, const std::string& name) {
    return std::to_string(static_cast<unsigned>(backend)) + ":" + name;
}

Status ModelLoweringRegistry::register_lowering(BackendKind backend,
                                                const std::string& operator_name,
                                                Lowering lowering) {
    if (backend == BackendKind::Unknown || operator_name.empty() || !lowering)
        return {StatusCode::InvalidArgument, "Model lowering requires a backend, operator name and callback"};
    const std::string key = make_key(backend, operator_name);
    if (lowerings_.find(key) != lowerings_.end())
        return {StatusCode::InvalidArgument, "Model lowering is already registered for this backend/operator"};
    lowerings_.emplace(key, std::move(lowering));
    return Status::success();
}

const ModelLoweringRegistry::Lowering* ModelLoweringRegistry::find(
        BackendKind backend, const std::string& operator_name) const {
    const auto found = lowerings_.find(make_key(backend, operator_name));
    return found == lowerings_.end() ? nullptr : &found->second;
}

Status compile_model_execution_plan(ModelExecutionPlan& plan, BackendKind backend,
                                    const ModelLoweringRegistry& lowerings,
                                    const OperatorRegistry& operators,
                                    const std::vector<std::string>& external_inputs,
                                    const ModelExecutionContext& context,
                                    ComputeGraph& graph, ModelGraphCompileReport& report) {
    report = ModelGraphCompileReport{};
    plan.graph_compiled = false;
    if (!plan.structurally_validated || !plan.operator_coverage_validated)
        return {StatusCode::NotReady, "Model graph compilation requires structural and operator-coverage validation"};
    if (backend == BackendKind::Unknown || graph.node_count() != 0)
        return {StatusCode::InvalidArgument, "Model graph compilation requires a backend and an empty destination graph"};

    const std::unordered_set<std::string> external(external_inputs.begin(), external_inputs.end());
    std::unordered_map<std::string, size_t> producers;
    std::unordered_set<std::string> operation_ids;
    ComputeGraph candidate;

    for (const auto& operation : plan.operations) {
        if (operation.id.empty() || operation.operator_name.empty() || operation.output.empty()) {
            report.errors.emplace_back("MODEL_OPERATION_HAS_EMPTY_ID_OPERATOR_OR_OUTPUT");
            continue;
        }
        if (!operation_ids.insert(operation.id).second) {
            report.errors.emplace_back("DUPLICATE_MODEL_OPERATION_ID:" + operation.id);
            continue;
        }
        bool inputs_ready = true;
        std::vector<size_t> dependencies;
        for (const auto& input : operation.inputs) {
            const auto producer = producers.find(input);
            if (producer != producers.end()) {
                if (std::find(dependencies.begin(), dependencies.end(), producer->second) == dependencies.end())
                    dependencies.push_back(producer->second);
            } else if (external.find(input) == external.end()) {
                report.unresolved_inputs.push_back(operation.id + ":" + input);
                inputs_ready = false;
            }
            const bool token_ids_external = input == "token_ids" &&
                external.find(input) != external.end() && !context.token_ids.empty();
            if (context.tensors.find(input) == context.tensors.end() && !token_ids_external) {
                report.unresolved_inputs.push_back(operation.id + ":UNBOUND:" + input);
                inputs_ready = false;
            }
        }
        for (const auto& control_input : operation.control_inputs) {
            const auto producer = producers.find(control_input);
            if (producer == producers.end()) {
                report.unresolved_inputs.push_back(operation.id + ":CONTROL:" + control_input);
                inputs_ready = false;
            } else if (std::find(dependencies.begin(), dependencies.end(), producer->second) == dependencies.end()) {
                dependencies.push_back(producer->second);
            }
        }
        for (const auto& weight : operation.weight_tensors) {
            const bool dense_bound = context.tensors.find(weight) != context.tensors.end();
            const bool quantized_bound = context.quantized_weights.find(weight) != context.quantized_weights.end();
            if (!dense_bound && !quantized_bound) {
                report.missing_weight_tensors.push_back(operation.id + ":" + weight);
                inputs_ready = false;
            }
        }
        if (operation.side_effect) {
            if (operation.output.empty()) {
                report.errors.push_back(operation.id + ":SIDE_EFFECT_OUTPUT_ID_REQUIRED");
                inputs_ready = false;
            }
        } else if (context.tensors.find(operation.output) == context.tensors.end()) {
            report.errors.push_back(operation.id + ":UNBOUND_OUTPUT:" + operation.output);
            inputs_ready = false;
        }
        if (!inputs_ready) continue;

        const auto* lowering = lowerings.find(backend, operation.operator_name);
        if (!lowering) {
            report.missing_lowerings.push_back(operation.operator_name);
            continue;
        }
        ComputeDispatch dispatch;
        Status status = (*lowering)(operation, context, dispatch);
        if (!status.ok()) {
            report.errors.push_back(operation.id + ":" + status.message);
            continue;
        }
        if (dispatch.backend != backend || (!dispatch.copy_only && dispatch.kernel_id.empty())) {
            report.errors.push_back(operation.id + ":LOWERING_RETURNED_WRONG_BACKEND_OR_EMPTY_KERNEL_ID");
            continue;
        }
        if (!dispatch.copy_only) {
            status = operators.authorize_dispatch(dispatch.kernel_id, backend, dispatch);
            if (!status.ok()) {
                report.errors.push_back(operation.id + ":" + status.message);
                continue;
            }
        }
        GraphNode node;
        node.name = operation.id;
        node.dependencies = std::move(dependencies);
        node.dispatch = std::move(dispatch);
        size_t index = 0;
        status = candidate.add_node(std::move(node), operators, index);
        if (!status.ok()) {
            report.errors.push_back(operation.id + ":" + status.message);
            continue;
        }
        producers[operation.output] = index;
    }

    if (!report.errors.empty() || !report.missing_lowerings.empty() ||
        !report.unresolved_inputs.empty() || !report.missing_weight_tensors.empty() ||
        candidate.node_count() != plan.operations.size()) {
        report.compiled_nodes = candidate.node_count();
        return {StatusCode::Unsupported, describe_compile_failure(report)};
    }
    graph = std::move(candidate);
    report.compiled = true;
    report.compiled_nodes = graph.node_count();
    plan.graph_compiled = true;
    plan.gpu_ready = false; // Graph compilation is not execution proof.
    return Status::success();
}


Status execute_model_execution_plan(ModelExecutionPlan& plan, Backend& backend,
                                    const ModelLoweringRegistry& lowerings,
                                    const OperatorRegistry& operators,
                                    const std::vector<std::string>& external_inputs,
                                    const ModelExecutionContext& context,
                                    const std::atomic<bool>* cancellation,
                                    ModelGraphCompileReport& report) {
    report = ModelGraphCompileReport{};
    plan.execution_validated = false;
    plan.gpu_ready = false;
    ComputeGraph graph;
    Status status = compile_model_execution_plan(plan, backend.kind(), lowerings,
                                                 operators, external_inputs, context,
                                                 graph, report);
    if (!status.ok()) return status;
    ExecutionScheduler scheduler;
    status = scheduler.execute(backend, graph, cancellation);
    if (!status.ok()) {
        report.errors.push_back("GRAPH_EXECUTION_FAILED:" + status.message);
        return status;
    }
    for (const auto& operation : plan.operations) {
        if (operation.operator_name != "KVCacheAppend") continue;
        if (!context.kv_cache) {
            report.errors.emplace_back("KV_CACHE_COMMIT_CONTEXT_MISSING");
            return {StatusCode::InvalidArgument, "KV cache context disappeared before commit"};
        }
        const auto layer_attr = operation.integer_attributes.find("layer_index");
        if (layer_attr == operation.integer_attributes.end() || layer_attr->second > UINT32_MAX ||
            operation.inputs.size() != 2) {
            report.errors.push_back(operation.id + ":KV_CACHE_COMMIT_METADATA_INVALID");
            return {StatusCode::InvalidArgument, "KV cache append commit metadata is invalid"};
        }
        const auto keys = context.tensors.find(operation.inputs[0]);
        if (keys == context.tensors.end() || keys->second.descriptor().shape.empty() ||
            keys->second.descriptor().shape[0] > UINT32_MAX) {
            report.errors.push_back(operation.id + ":KV_CACHE_COMMIT_TENSOR_MISSING");
            return {StatusCode::InvalidArgument, "KV cache append source tensor is missing at commit"};
        }
        status = context.kv_cache->commit_device_append(
            static_cast<uint32_t>(layer_attr->second),
            static_cast<uint32_t>(context.past_length),
            static_cast<uint32_t>(keys->second.descriptor().shape[0]));
        if (!status.ok()) {
            report.errors.push_back(operation.id + ":KV_CACHE_COMMIT_FAILED:" + status.message);
            return status;
        }
    }
    report.executed = true;
    plan.execution_validated = true;
    // One successful graph run is not a model-family or quantization certification.
    // The caller must still perform model-level parity and lifecycle qualification.
    plan.gpu_ready = false;
    return Status::success();
}



Status execute_model_execution_plan_streamed(ModelExecutionPlan& plan, Backend& backend,
                                    const ModelLoweringRegistry& lowerings,
                                    const OperatorRegistry& operators,
                                    const std::vector<std::string>& external_inputs,
                                    const GgufReader& reader, MemoryManager& memory,
                                    const DeviceInfo& device, uint64_t token_count,
                                    const std::atomic<bool>* cancellation,
                                    ModelExecutionContext& context,
                                    ModelGraphCompileReport& report) {
    report = ModelGraphCompileReport{};
    plan.graph_compiled = false;
    plan.execution_validated = false;
    plan.gpu_ready = false;
    if (!plan.structurally_validated || !plan.operator_coverage_validated ||
        plan.operations.empty() || backend.kind() == BackendKind::Unknown ||
        !backend.healthy() || token_count == 0)
        return {StatusCode::NotReady, "Streamed execution requires a non-empty validated plan, healthy backend and non-zero token count"};
    context.backend = &backend;
    if (context.token_ids.size() != token_count)
        return {StatusCode::InvalidArgument, "Streamed execution token count does not match the bound token IDs"};

    std::unordered_map<std::string, size_t> producer_indices;
    for (size_t index = 0; index < plan.operations.size(); ++index) {
        const auto& operation = plan.operations[index];
        if (operation.output.empty() || !producer_indices.emplace(operation.output, index).second)
            return {StatusCode::InvalidArgument, "Streamed model plan has an empty or duplicate output name"};
    }
    for (size_t index = 0; index < plan.operations.size(); ++index) {
        const auto& operation = plan.operations[index];
        for (const auto& input : operation.inputs) {
            const auto producer = producer_indices.find(input);
            if (producer != producer_indices.end() && producer->second >= index)
                return {StatusCode::InvalidArgument, "Streamed model plan is not in dependency order: " + operation.id};
        }
        for (const auto& control : operation.control_inputs) {
            const auto producer = producer_indices.find(control);
            if (producer == producer_indices.end() || producer->second >= index)
                return {StatusCode::InvalidArgument, "Streamed model plan has an unresolved or forward control dependency: " + operation.id};
        }
    }

    Status status = allocate_model_activation_tensors(plan, memory, token_count, context);
    if (!status.ok()) return status;
    const bool has_attention = std::any_of(plan.operations.begin(), plan.operations.end(),
        [](const ModelPlanOperation& operation) { return operation.operator_name == "Attention"; });
    const bool has_kv_append = std::any_of(plan.operations.begin(), plan.operations.end(),
        [](const ModelPlanOperation& operation) { return operation.operator_name == "KVCacheAppend"; });
    if ((has_attention || has_kv_append) &&
        (!context.kv_cache || context.sequence_length != token_count))
        return {StatusCode::NotReady, "Streamed attention execution requires a KV cache and a sequence length matching the invocation"};
    if (has_attention) {
        status = bind_model_kv_cache_views(plan, *context.kv_cache, context.past_length,
                                           context.sequence_length, context);
        if (!status.ok()) return status;
    }
    for (const auto& operation : plan.operations) {
        for (const auto& weight : operation.weight_tensors) {
            if (context.tensors.find(weight) != context.tensors.end() ||
                context.quantized_weights.find(weight) != context.quantized_weights.end())
                return {StatusCode::InvalidArgument, "Streamed execution requires model weights to be loaded on demand, not pre-bound"};
        }
    }

    ExecutionScheduler scheduler;
    for (const auto& original_operation : plan.operations) {
        if (cancellation && cancellation->load(std::memory_order_relaxed)) {
            report.errors.emplace_back("STREAMED_EXECUTION_CANCELLED");
            return {StatusCode::Cancelled, "Streamed model execution was cancelled"};
        }
        if (!backend.healthy()) {
            report.errors.emplace_back("STREAMED_BACKEND_DEVICE_LOST");
            return {StatusCode::DeviceLost, "GPU backend became unavailable during streamed execution"};
        }

        ModelPlanOperation operation = original_operation;
        // Operations are executed serially in the already-validated plan order;
        // their control dependencies are therefore satisfied by prior submits.
        operation.control_inputs.clear();
        ModelExecutionPlan single;
        single.structurally_validated = true;
        single.operator_coverage_validated = true;
        single.external_inputs = external_inputs;
        single.operations.push_back(operation);

        ModelExecutionContext operation_context;
        operation_context.tensors = context.tensors;
        operation_context.backend = &backend;
        operation_context.token_ids = context.token_ids;
        operation_context.kv_cache = context.kv_cache;
        operation_context.past_length = context.past_length;
        operation_context.sequence_length = context.sequence_length;
        operation_context.context_length = context.context_length;
        operation_context.prefill = context.prefill;

        // Only the final prompt position is consumed by the sampler. Slice the
        // final hidden-state row before output projection, avoiding [tokens,vocab]
        // logits storage during prefill.
        if (operation.output == "logits" && context.prefill &&
            context.token_ids.size() > 1 && !operation.inputs.empty()) {
            auto hidden = operation_context.tensors.find(operation.inputs.front());
            if (hidden == operation_context.tensors.end() ||
                hidden->second.descriptor().data_type != DataType::F32 ||
                hidden->second.descriptor().shape.size() != 2 ||
                hidden->second.descriptor().shape[0] != token_count) {
                report.errors.push_back(operation.id + ":LAST_TOKEN_LOGITS_INPUT_INVALID");
                return {StatusCode::InvalidArgument, "Final hidden-state tensor cannot be sliced for last-token logits"};
            }
            const Tensor& full_hidden = hidden->second;
            if (full_hidden.descriptor().shape[1] == 0 ||
                full_hidden.descriptor().shape[1] > UINT64_MAX / sizeof(float))
                return {StatusCode::InvalidArgument, "Final hidden-state row size overflowed"};
            const uint64_t row_bytes = full_hidden.descriptor().shape[1] * sizeof(float);
            if (token_count - 1 > UINT64_MAX / row_bytes)
                return {StatusCode::InvalidArgument, "Final hidden-state row offset overflowed"};
            const uint64_t row_offset = (token_count - 1) * row_bytes;
            if (full_hidden.byte_offset() > UINT64_MAX - row_offset)
                return {StatusCode::InvalidArgument, "Final hidden-state byte offset overflowed"};
            TensorDesc last_row_desc = full_hidden.descriptor();
            last_row_desc.shape[0] = 1;
            last_row_desc.strides_bytes.clear();
            Tensor last_hidden_row;
            status = Tensor::create(last_row_desc, full_hidden.buffer(),
                full_hidden.byte_offset() + row_offset, last_hidden_row);
            if (!status.ok()) return status;
            hidden->second = std::move(last_hidden_row);
        }

        if (!operation.weight_tensors.empty()) {
            status = load_model_execution_weights(reader, single, backend, memory,
                                                  device, operation_context);
            if (!status.ok()) {
                report.errors.push_back(operation.id + ":WEIGHT_STREAM_FAILED:" + status.message);
                for (const auto& name : operation.weight_tensors) {
                    operation_context.tensors.erase(name);
                    operation_context.quantized_weights.erase(name);
                }
                operation_context.weight_leases.clear();
                return status;
            }
        }

        bool chunked_q6_matmul = false;
        status = execute_chunked_q6_k_matmul_if_needed(operation, backend, operators,
            reader, memory, device, operation_context, chunked_q6_matmul);
        if (!status.ok()) {
            report.errors.push_back(operation.id + ":CHUNKED_Q6_K_MATMUL_FAILED:" + status.message);
            for (const auto& name : operation.weight_tensors) {
                operation_context.tensors.erase(name);
                operation_context.quantized_weights.erase(name);
            }
            operation_context.weight_leases.clear();
            return status;
        }
        if (chunked_q6_matmul) {
            ++report.compiled_nodes;
            for (const auto& name : operation.weight_tensors) {
                operation_context.tensors.erase(name);
                operation_context.quantized_weights.erase(name);
            }
            operation_context.weight_leases.clear();
            continue;
        }

        bool chunked_elementwise = false;
        status = execute_chunked_elementwise_if_needed(operation, backend, lowerings,
            operators, device, operation_context, chunked_elementwise);
        if (!status.ok()) {
            report.errors.push_back(operation.id + ":CHUNKED_ELEMENTWISE_FAILED:" + status.message);
            for (const auto& name : operation.weight_tensors) {
                operation_context.tensors.erase(name);
                operation_context.quantized_weights.erase(name);
            }
            operation_context.weight_leases.clear();
            return status;
        }
        if (chunked_elementwise) {
            ++report.compiled_nodes;
            for (const auto& name : operation.weight_tensors) {
                operation_context.tensors.erase(name);
                operation_context.quantized_weights.erase(name);
            }
            operation_context.weight_leases.clear();
            continue;
        }

        {
            ComputeGraph graph;
            ModelGraphCompileReport compile_report;
            std::vector<std::string> operation_external_inputs = external_inputs;
            operation_external_inputs.reserve(operation_external_inputs.size() +
                                               operation_context.tensors.size() + 1);
            for (const auto& binding : operation_context.tensors)
                operation_external_inputs.push_back(binding.first);
            if (!operation_context.token_ids.empty())
                operation_external_inputs.emplace_back("token_ids");
            status = compile_model_execution_plan(single, backend.kind(), lowerings,
                operators, operation_external_inputs, operation_context, graph, compile_report);
            report.missing_lowerings.insert(report.missing_lowerings.end(),
                compile_report.missing_lowerings.begin(), compile_report.missing_lowerings.end());
            report.unresolved_inputs.insert(report.unresolved_inputs.end(),
                compile_report.unresolved_inputs.begin(), compile_report.unresolved_inputs.end());
            report.missing_weight_tensors.insert(report.missing_weight_tensors.end(),
                compile_report.missing_weight_tensors.begin(), compile_report.missing_weight_tensors.end());
            report.errors.insert(report.errors.end(), compile_report.errors.begin(), compile_report.errors.end());
            if (!status.ok()) {
                for (const auto& name : operation.weight_tensors) {
                    operation_context.tensors.erase(name);
                    operation_context.quantized_weights.erase(name);
                }
                operation_context.weight_leases.clear();
                return status;
            }
            status = scheduler.execute(backend, graph, cancellation);
            if (!status.ok()) {
                report.errors.push_back(operation.id + ":STREAMED_DISPATCH_FAILED:" + status.message);
                for (const auto& name : operation.weight_tensors) {
                    operation_context.tensors.erase(name);
                    operation_context.quantized_weights.erase(name);
                }
                operation_context.weight_leases.clear();
                return status;
            }
            ++report.compiled_nodes;
        }

        // Drop every temporary weight reference before releasing its lease so
        // MemoryManager can reclaim it without aliasing a still-live Tensor.
        for (const auto& name : operation.weight_tensors) {
            operation_context.tensors.erase(name);
            operation_context.quantized_weights.erase(name);
        }
        operation_context.weight_leases.clear();
    }

    for (const auto& operation : plan.operations) {
        if (operation.operator_name != "KVCacheAppend") continue;
        if (!context.kv_cache || operation.inputs.size() != 2) {
            report.errors.push_back(operation.id + ":KV_CACHE_COMMIT_CONTEXT_INVALID");
            return {StatusCode::InvalidArgument, "KV cache append commit context is invalid"};
        }
        const auto layer = operation.integer_attributes.find("layer_index");
        const auto keys = context.tensors.find(operation.inputs[0]);
        if (layer == operation.integer_attributes.end() || layer->second > UINT32_MAX ||
            keys == context.tensors.end() || keys->second.descriptor().shape.empty() ||
            keys->second.descriptor().shape[0] > UINT32_MAX) {
            report.errors.push_back(operation.id + ":KV_CACHE_COMMIT_METADATA_INVALID");
            return {StatusCode::InvalidArgument, "KV cache append commit metadata is invalid"};
        }
        status = context.kv_cache->commit_device_append(static_cast<uint32_t>(layer->second),
            static_cast<uint32_t>(context.past_length),
            static_cast<uint32_t>(keys->second.descriptor().shape[0]));
        if (!status.ok()) {
            report.errors.push_back(operation.id + ":KV_CACHE_COMMIT_FAILED:" + status.message);
            return status;
        }
    }

    report.compiled = true;
    report.executed = true;
    plan.graph_compiled = true;
    plan.execution_validated = true;
    // Streaming is a valid execution strategy, not a model-family certification.
    plan.gpu_ready = false;
    return Status::success();
}

Status bind_model_kv_cache_views(const ModelExecutionPlan& plan, KvCache& cache,
                                 uint64_t past_length, uint64_t sequence_length,
                                 ModelExecutionContext& context) {
    if (sequence_length == 0 || past_length > UINT32_MAX ||
        sequence_length > UINT32_MAX - past_length)
        return {StatusCode::InvalidArgument, "KV cache view sequence range is invalid"};
    const uint32_t total_length = static_cast<uint32_t>(past_length + sequence_length);
    std::vector<std::pair<std::string, Tensor>> pending_views;
    for (const auto& operation : plan.operations) {
        if (operation.operator_name != "Attention") continue;
        if (operation.inputs.size() != 3)
            return {StatusCode::InvalidArgument, "Attention plan must reference Q and cache K/V tensor names"};
        const auto layer_attr = operation.integer_attributes.find("layer_index");
        if (layer_attr == operation.integer_attributes.end() || layer_attr->second > UINT32_MAX)
            return {StatusCode::InvalidArgument, "Attention plan is missing a valid KV cache layer index"};
        Tensor keys, values;
        Status status = cache.key_tensor_for_length(static_cast<uint32_t>(layer_attr->second),
                                                    total_length, keys);
        if (!status.ok()) return status;
        status = cache.value_tensor_for_length(static_cast<uint32_t>(layer_attr->second),
                                               total_length, values);
        if (!status.ok()) return status;
        pending_views.emplace_back(operation.inputs[1], std::move(keys));
        pending_views.emplace_back(operation.inputs[2], std::move(values));
    }

    // Rebind only cache-view names after all new views have been created
    // successfully. This supports repeated prefill/decode calls with a growing
    // valid prefix without disturbing weights or activation allocations.
    for (const auto& entry : pending_views) context.tensors.erase(entry.first);
    for (auto& entry : pending_views)
        context.tensors.emplace(std::move(entry.first), std::move(entry.second));
    context.kv_cache = &cache;
    context.past_length = past_length;
    context.sequence_length = sequence_length;
    return Status::success();
}

Status validate_model_plan_operator_coverage(ModelExecutionPlan& plan,
                                             const OperatorRegistry& registry,
                                             BackendKind backend,
                                             ModelGraphCoverageReport& report) {
    report = ModelGraphCoverageReport{};
    if (!plan.structurally_validated)
        return {StatusCode::InvalidArgument, "Cannot validate operator coverage for a structurally invalid model plan"};
    std::unordered_set<std::string> seen;
    for (const auto& operation : plan.operations) {
        if (!seen.insert(operation.operator_name).second) continue;
        if (operation.operator_name == "KVCacheAppend") {
            report.supported_operators.push_back(operation.operator_name);
            continue;
        }
        if (registry.has_validated_kernel(operation.operator_name, backend))
            report.supported_operators.push_back(operation.operator_name);
        else
            report.unsupported_operators.push_back(operation.operator_name);
    }
    std::sort(report.supported_operators.begin(), report.supported_operators.end());
    std::sort(report.unsupported_operators.begin(), report.unsupported_operators.end());
    report.complete = report.unsupported_operators.empty();
    const std::string blocker = "OPERATOR_KERNEL_COVERAGE_NOT_VALIDATED";
    auto found = std::find(plan.blockers.begin(), plan.blockers.end(), blocker);
    if (report.complete) {
        if (found != plan.blockers.end()) plan.blockers.erase(found);
    } else if (found == plan.blockers.end()) {
        plan.blockers.push_back(blocker);
    }
    plan.operator_coverage_validated = report.complete;
    plan.gpu_ready = plan.structurally_validated && plan.operator_coverage_validated &&
                     plan.execution_validated && plan.blockers.empty();
    return Status::success();
}

} // namespace ai_chat::gpu_engine
