#include "model_graph.h"
#include "execution_engine.h"

#include <algorithm>
#include <unordered_set>
#include <utility>

namespace ai_chat::gpu_engine {


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
            if (context.tensors.find(weight) == context.tensors.end()) {
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
        return {StatusCode::Unsupported, "Model execution plan could not be fully lowered into a validated compute graph"};
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



Status bind_model_kv_cache_views(const ModelExecutionPlan& plan, KvCache& cache,
                                 uint64_t past_length, uint64_t sequence_length,
                                 ModelExecutionContext& context) {
    if (sequence_length == 0 || past_length > UINT32_MAX ||
        sequence_length > UINT32_MAX - past_length)
        return {StatusCode::InvalidArgument, "KV cache view sequence range is invalid"};
    const uint32_t total_length = static_cast<uint32_t>(past_length + sequence_length);
    for (const auto& operation : plan.operations) {
        if (operation.operator_name != "Attention") continue;
        if (operation.inputs.size() != 3)
            return {StatusCode::InvalidArgument, "Attention plan must reference Q and cache K/V tensor names"};
        const auto layer_attr = operation.integer_attributes.find("layer_index");
        if (layer_attr == operation.integer_attributes.end() || layer_attr->second > UINT32_MAX)
            return {StatusCode::InvalidArgument, "Attention plan is missing a valid KV cache layer index"};
        if (context.tensors.find(operation.inputs[1]) != context.tensors.end() ||
            context.tensors.find(operation.inputs[2]) != context.tensors.end())
            return {StatusCode::InvalidArgument, "KV cache key/value tensor binding already exists"};
        Tensor keys, values;
        Status status = cache.key_tensor_for_length(static_cast<uint32_t>(layer_attr->second),
                                                    total_length, keys);
        if (!status.ok()) return status;
        status = cache.value_tensor_for_length(static_cast<uint32_t>(layer_attr->second),
                                               total_length, values);
        if (!status.ok()) return status;
        context.tensors.emplace(operation.inputs[1], std::move(keys));
        context.tensors.emplace(operation.inputs[2], std::move(values));
    }
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
