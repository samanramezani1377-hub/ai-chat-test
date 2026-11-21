#include "model_lowerings.h"
#include "operators.h"

#include <cmath>
#include <limits>
#include <utility>

namespace ai_chat::gpu_engine {
namespace {

Status lookup_tensor(const ModelExecutionContext& context, const std::string& name,
                     const Tensor*& tensor) {
    const auto found = context.tensors.find(name);
    if (found == context.tensors.end())
        return {StatusCode::InvalidArgument, "Model lowering tensor binding is missing: " + name};
    tensor = &found->second;
    return Status::success();
}

Status lookup_output(const ModelPlanOperation& operation,
                     const ModelExecutionContext& context, const Tensor*& output) {
    return lookup_tensor(context, operation.output, output);
}

Status lookup_operand(const ModelPlanOperation& operation,
                      const ModelExecutionContext& context, size_t index,
                      const Tensor*& tensor) {
    if (index < operation.inputs.size())
        return lookup_tensor(context, operation.inputs[index], tensor);
    index -= operation.inputs.size();
    if (index < operation.weight_tensors.size())
        return lookup_tensor(context, operation.weight_tensors[index], tensor);
    return {StatusCode::InvalidArgument, "Model lowering is missing operand " + std::to_string(index)};
}

Status required_float(const ModelPlanOperation& operation, const char* key, float& value) {
    const auto found = operation.float_attributes.find(key);
    if (found == operation.float_attributes.end() || !std::isfinite(found->second) ||
        found->second > std::numeric_limits<float>::max() ||
        found->second < -std::numeric_limits<float>::max())
        return {StatusCode::InvalidArgument, std::string("Model lowering requires finite float attribute: ") + key};
    value = static_cast<float>(found->second);
    return Status::success();
}

Status required_integer(const ModelPlanOperation& operation, const char* key, uint32_t& value) {
    const auto found = operation.integer_attributes.find(key);
    if (found == operation.integer_attributes.end() || found->second > UINT32_MAX)
        return {StatusCode::InvalidArgument, std::string("Model lowering requires uint32 attribute: ") + key};
    value = static_cast<uint32_t>(found->second);
    return Status::success();
}

Status required_boolean(const ModelPlanOperation& operation, const char* key, bool& value) {
    const auto found = operation.bool_attributes.find(key);
    if (found == operation.bool_attributes.end())
        return {StatusCode::InvalidArgument, std::string("Model lowering requires boolean attribute: ") + key};
    value = found->second;
    return Status::success();
}

Status register_binary(BackendKind backend, ModelLoweringRegistry& registry,
                       const std::string& name, bool multiply) {
    return registry.register_lowering(backend, name,
        [backend, multiply](const ModelPlanOperation& operation,
                            const ModelExecutionContext& context, ComputeDispatch& dispatch) {
            if (operation.inputs.size() + operation.weight_tensors.size() != 2)
                return Status{StatusCode::InvalidArgument, "Binary lowering requires exactly two bound operands"};
            const Tensor *lhs = nullptr, *rhs = nullptr, *output = nullptr;
            Status status = lookup_operand(operation, context, 0, lhs);
            if (!status.ok()) return status;
            status = lookup_operand(operation, context, 1, rhs);
            if (!status.ok()) return status;
            status = lookup_output(operation, context, output);
            if (!status.ok()) return status;
            if (backend == BackendKind::OpenGles) {
                return multiply ? build_multiply_f32_gles_candidate(*lhs, *rhs, *output, dispatch)
                                : build_add_f32_gles_candidate(*lhs, *rhs, *output, dispatch);
            }
            return multiply ? build_multiply_f32_candidate(*lhs, *rhs, *output, dispatch)
                            : build_add_f32_candidate(*lhs, *rhs, *output, dispatch);
        });
}

} // namespace

Status register_builtin_model_lowerings(BackendKind backend, ModelLoweringRegistry& registry) {
    if (backend != BackendKind::DawnWebGpu && backend != BackendKind::OpenGles)
        return {StatusCode::Unsupported, "Built-in model lowerings are not defined for this backend"};

    Status status = register_binary(backend, registry, "Add", false);
    if (!status.ok()) return status;
    status = register_binary(backend, registry, "Multiply", true);
    if (!status.ok()) return status;

    status = registry.register_lowering(backend, "MatMul",
        [backend](const ModelPlanOperation& operation, const ModelExecutionContext& context,
                  ComputeDispatch& dispatch) {
            if (operation.inputs.size() + operation.weight_tensors.size() != 2)
                return Status{StatusCode::InvalidArgument, "MatMul lowering requires two bound operands"};
            const Tensor *lhs = nullptr, *rhs = nullptr, *output = nullptr;
            Status result = lookup_operand(operation, context, 0, lhs);
            if (!result.ok()) return result;
            result = lookup_operand(operation, context, 1, rhs);
            if (!result.ok()) return result;
            result = lookup_output(operation, context, output);
            if (!result.ok()) return result;
            return backend == BackendKind::OpenGles
                ? build_matmul_f32_gles_candidate(*lhs, *rhs, *output, dispatch)
                : build_matmul_f32_candidate(*lhs, *rhs, *output, dispatch);
        });
    if (!status.ok()) return status;

    status = registry.register_lowering(backend, "RMSNorm",
        [backend](const ModelPlanOperation& operation, const ModelExecutionContext& context,
                  ComputeDispatch& dispatch) {
            if (operation.inputs.size() != 1 || operation.weight_tensors.size() != 1)
                return Status{StatusCode::InvalidArgument, "RMSNorm requires one activation and one weight tensor"};
            const Tensor *input = nullptr, *weight = nullptr, *output = nullptr;
            Status result = lookup_operand(operation, context, 0, input);
            if (!result.ok()) return result;
            result = lookup_operand(operation, context, 1, weight);
            if (!result.ok()) return result;
            result = lookup_output(operation, context, output);
            if (!result.ok()) return result;
            float epsilon = 0.0f;
            result = required_float(operation, "epsilon", epsilon);
            if (!result.ok()) return result;
            return backend == BackendKind::OpenGles
                ? build_rms_norm_f32_gles_candidate(*input, *weight, *output, epsilon, dispatch)
                : build_rms_norm_f32_candidate(*input, *weight, *output, epsilon, dispatch);
        });
    if (!status.ok()) return status;

    for (const char* unary_name : {"SiLU", "Softmax"}) {
        status = registry.register_lowering(backend, unary_name,
            [backend, unary_name](const ModelPlanOperation& operation,
                                 const ModelExecutionContext& context, ComputeDispatch& dispatch) {
                if (operation.inputs.size() != 1 || !operation.weight_tensors.empty())
                    return Status{StatusCode::InvalidArgument, "Unary lowering requires exactly one activation input"};
                const Tensor *input = nullptr, *output = nullptr;
                Status result = lookup_operand(operation, context, 0, input);
                if (!result.ok()) return result;
                result = lookup_output(operation, context, output);
                if (!result.ok()) return result;
                if (std::string(unary_name) == "SiLU")
                    return backend == BackendKind::OpenGles
                        ? build_silu_f32_gles_candidate(*input, *output, dispatch)
                        : build_silu_f32_candidate(*input, *output, dispatch);
                return backend == BackendKind::OpenGles
                    ? build_softmax_f32_gles_candidate(*input, *output, dispatch)
                    : build_softmax_f32_candidate(*input, *output, dispatch);
            });
        if (!status.ok()) return status;
    }

    status = registry.register_lowering(backend, "RoPE",
        [backend](const ModelPlanOperation& operation, const ModelExecutionContext& context,
                  ComputeDispatch& dispatch) {
            if (operation.inputs.size() != 1 || !operation.weight_tensors.empty() ||
                context.past_length > UINT32_MAX)
                return Status{StatusCode::InvalidArgument, "RoPE requires one activation and a uint32 position"};
            const Tensor *input = nullptr, *output = nullptr;
            Status result = lookup_operand(operation, context, 0, input);
            if (!result.ok()) return result;
            result = lookup_output(operation, context, output);
            if (!result.ok()) return result;
            RopeConfig config;
            result = required_integer(operation, "rotary_dimensions", config.rotary_dimensions);
            if (!result.ok()) return result;
            result = required_float(operation, "theta", config.theta);
            if (!result.ok()) return result;
            result = required_boolean(operation, "neox_style", config.neox_style);
            if (!result.ok()) return result;
            config.start_position = static_cast<uint32_t>(context.past_length);
            return backend == BackendKind::OpenGles
                ? build_rope_f32_gles_candidate(*input, *output, config, dispatch)
                : build_rope_f32_candidate(*input, *output, config, dispatch);
        });
    if (!status.ok()) return status;

    // Embedding needs a backend-owned token-id upload and lifetime contract.
    // Attention needs the KV-cache append/read contract. Both intentionally stay
    // unregistered until those resource paths are implemented and validated.
    return Status::success();
}

} // namespace ai_chat::gpu_engine
