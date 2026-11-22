#include "model_lowerings.h"
#include "kv_cache.h"
#include "operators.h"
#include "quantization.h"

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


Status register_q5_matmul_lowering(BackendKind backend, ModelLoweringRegistry& registry, bool q5_1) {
    const std::string type = q5_1 ? "Q5_1" : "Q5_0";
    const std::string op = q5_1 ? "MatMul_Q5_1" : "MatMul_Q5_0";
    const uint32_t type_id = q5_1 ? 7u : 6u;
    return registry.register_lowering(backend, op,
        [backend, type, type_id, q5_1](const ModelPlanOperation& operation,
                                      const ModelExecutionContext& context, ComputeDispatch& dispatch) {
            if (operation.inputs.size() != 1 || operation.weight_tensors.size() != 1)
                return Status{StatusCode::InvalidArgument, type + " MatMul requires one activation and one quantized weight"};
            const Tensor *input = nullptr, *output = nullptr;
            Status result = lookup_operand(operation, context, 0, input);
            if (!result.ok()) return result;
            result = lookup_output(operation, context, output);
            if (!result.ok()) return result;
            const auto weight = context.quantized_weights.find(operation.weight_tensors.front());
            if (weight == context.quantized_weights.end() || !weight->second.buffer ||
                weight->second.byte_size == 0 || weight->second.file_type_id != type_id)
                return Status{StatusCode::InvalidArgument, type + " weight binding is missing or has the wrong file type"};
            if (input->descriptor().shape.size() != 2 ||
                (output->descriptor().shape.size() != 2 && output->descriptor().shape.size() != 3) ||
                input->descriptor().shape[0] == 0)
                return Status{StatusCode::Unsupported, type + " MatMul requires non-empty [M,K] activations and [M,N] or [M,H,D] output"};
            const uint64_t output_channels = output->element_count() / input->descriptor().shape[0];
            if (weight->second.dimensions.size() != 2 ||
                weight->second.dimensions[0] != input->descriptor().shape[1] ||
                weight->second.dimensions[1] != output_channels)
                return Status{StatusCode::InvalidArgument, type + " weight dimensions do not match activation and output shapes"};
            if (backend == BackendKind::OpenGles)
                return q5_1 ? build_q5_1_matmul_gles_candidate(*input, weight->second.buffer, weight->second.byte_size, *output, dispatch)
                            : build_q5_0_matmul_gles_candidate(*input, weight->second.buffer, weight->second.byte_size, *output, dispatch);
            return q5_1 ? build_q5_1_matmul_candidate(*input, weight->second.buffer, weight->second.byte_size, *output, dispatch)
                        : build_q5_0_matmul_candidate(*input, weight->second.buffer, weight->second.byte_size, *output, dispatch);
        });
}


Status register_q5_embedding_lowering(BackendKind backend,ModelLoweringRegistry& registry,bool q5_1){
    const std::string type=q5_1?"Q5_1":"Q5_0",op=q5_1?"Embedding_Q5_1":"Embedding_Q5_0";
    const uint32_t type_id=q5_1?7u:6u;
    return registry.register_lowering(backend,op,[backend,type,type_id,q5_1](
        const ModelPlanOperation& operation,const ModelExecutionContext& context,ComputeDispatch& dispatch){
        if(operation.inputs.size()!=1||operation.inputs[0]!="token_ids"||
           operation.weight_tensors.size()!=1||!context.backend||context.token_ids.empty()||
           context.token_ids.size()>UINT32_MAX)
            return Status{StatusCode::InvalidArgument,type+" Embedding requires token IDs, one quantized weight and an active backend"};
        const Tensor* output=nullptr;Status result=lookup_output(operation,context,output);if(!result.ok())return result;
        const auto weight=context.quantized_weights.find(operation.weight_tensors.front());
        if(weight==context.quantized_weights.end()||!weight->second.buffer||weight->second.byte_size==0||
           weight->second.file_type_id!=type_id||weight->second.dimensions.size()!=2||
           weight->second.dimensions[0]>UINT32_MAX||weight->second.dimensions[1]>UINT32_MAX)
            return Status{StatusCode::InvalidArgument,type+" embedding weight binding is missing or has invalid metadata"};
        const uint32_t dimension=static_cast<uint32_t>(weight->second.dimensions[0]);
        const uint32_t vocabulary=static_cast<uint32_t>(weight->second.dimensions[1]);
        if(output->descriptor().shape.size()!=2||output->descriptor().shape[0]!=context.token_ids.size()||
           output->descriptor().shape[1]!=dimension)
            return Status{StatusCode::InvalidArgument,type+" embedding output shape does not match tokens and weight dimensions"};
        BufferDesc token_desc;
        token_desc.size_bytes=context.token_ids.size()*sizeof(uint32_t);
        token_desc.alignment=alignof(uint32_t);
        token_desc.memory=MemoryDomain::Device;
        token_desc.persistent=false;
        token_desc.label="q5-model-token-ids";
        std::shared_ptr<Buffer> token_buffer;
        result=context.backend->allocate(token_desc,token_buffer);if(!result.ok())return result;
        if(!token_buffer||token_buffer->size_bytes()<token_desc.size_bytes)
            return Status{StatusCode::BackendFailure,"Backend returned an undersized Q5 token-ID buffer"};
        result=context.backend->upload(token_buffer,0,context.token_ids.data(),token_desc.size_bytes);if(!result.ok())return result;
        return backend==BackendKind::OpenGles
            ?(q5_1?build_q5_1_embedding_gles_candidate(weight->second.buffer,weight->second.byte_size,vocabulary,dimension,token_buffer,context.token_ids,*output,dispatch)
                  :build_q5_0_embedding_gles_candidate(weight->second.buffer,weight->second.byte_size,vocabulary,dimension,token_buffer,context.token_ids,*output,dispatch))
            :(q5_1?build_q5_1_embedding_candidate(weight->second.buffer,weight->second.byte_size,vocabulary,dimension,token_buffer,context.token_ids,*output,dispatch)
                  :build_q5_0_embedding_candidate(weight->second.buffer,weight->second.byte_size,vocabulary,dimension,token_buffer,context.token_ids,*output,dispatch));
    });
}

Status register_fp16_matmul_lowering(BackendKind backend,ModelLoweringRegistry&registry,bool bf16){
    const std::string type=bf16?"BF16":"F16",op=bf16?"MatMul_BF16":"MatMul_F16";
    const uint32_t type_id=bf16?30u:1u;
    return registry.register_lowering(backend,op,[backend,type,type_id,bf16](
        const ModelPlanOperation&operation,const ModelExecutionContext&context,ComputeDispatch&dispatch){
        if(operation.inputs.size()!=1||operation.weight_tensors.size()!=1)
            return Status{StatusCode::InvalidArgument,type+" MatMul requires one activation and one packed weight"};
        const Tensor*input=nullptr,*output=nullptr;Status result=lookup_operand(operation,context,0,input);if(!result.ok())return result;
        result=lookup_output(operation,context,output);if(!result.ok())return result;
        const auto weight=context.quantized_weights.find(operation.weight_tensors.front());
        if(weight==context.quantized_weights.end()||!weight->second.buffer||weight->second.byte_size==0||
           weight->second.file_type_id!=type_id||weight->second.dimensions.size()!=2)
            return Status{StatusCode::InvalidArgument,type+" MatMul weight binding is missing or invalid"};
        if(input->descriptor().shape.size()!=2||input->descriptor().shape[0]==0||
           (output->descriptor().shape.size()!=2&&output->descriptor().shape.size()!=3))
            return Status{StatusCode::Unsupported,type+" MatMul activation/output shapes are unsupported"};
        const uint64_t channels=output->element_count()/input->descriptor().shape[0];
        if(weight->second.dimensions[0]!=input->descriptor().shape[1]||weight->second.dimensions[1]!=channels)
            return Status{StatusCode::InvalidArgument,type+" MatMul weight dimensions do not match activation/output"};
        if(backend==BackendKind::OpenGles)
            return bf16?build_bf16_matmul_gles_candidate(*input,weight->second.buffer,weight->second.byte_size,*output,dispatch)
                       :build_f16_matmul_gles_candidate(*input,weight->second.buffer,weight->second.byte_size,*output,dispatch);
        return bf16?build_bf16_matmul_candidate(*input,weight->second.buffer,weight->second.byte_size,*output,dispatch)
                   :build_f16_matmul_candidate(*input,weight->second.buffer,weight->second.byte_size,*output,dispatch);
    });
}
Status register_fp16_embedding_lowering(BackendKind backend,ModelLoweringRegistry&registry,bool bf16){
    const std::string type=bf16?"BF16":"F16",op=bf16?"Embedding_BF16":"Embedding_F16";
    const uint32_t type_id=bf16?30u:1u;
    return registry.register_lowering(backend,op,[backend,type,type_id,bf16](
        const ModelPlanOperation&operation,const ModelExecutionContext&context,ComputeDispatch&dispatch){
        if(operation.inputs.size()!=1||operation.inputs[0]!="token_ids"||operation.weight_tensors.size()!=1||
           !context.backend||context.token_ids.empty()||context.token_ids.size()>UINT32_MAX)
            return Status{StatusCode::InvalidArgument,type+" Embedding requires token IDs, one packed weight and an active backend"};
        const Tensor*output=nullptr;Status result=lookup_output(operation,context,output);if(!result.ok())return result;
        const auto weight=context.quantized_weights.find(operation.weight_tensors.front());
        if(weight==context.quantized_weights.end()||!weight->second.buffer||weight->second.byte_size==0||
           weight->second.file_type_id!=type_id||weight->second.dimensions.size()!=2||
           weight->second.dimensions[0]>UINT32_MAX||weight->second.dimensions[1]>UINT32_MAX)
            return Status{StatusCode::InvalidArgument,type+" Embedding weight binding is missing or invalid"};
        const uint32_t dimension=static_cast<uint32_t>(weight->second.dimensions[0]);
        const uint32_t vocabulary=static_cast<uint32_t>(weight->second.dimensions[1]);
        if(output->descriptor().shape.size()!=2||output->descriptor().shape[0]!=context.token_ids.size()||
           output->descriptor().shape[1]!=dimension)
            return Status{StatusCode::InvalidArgument,type+" Embedding output shape is inconsistent"};
        BufferDesc token_desc;token_desc.size_bytes=context.token_ids.size()*sizeof(uint32_t);
        token_desc.alignment=alignof(uint32_t);token_desc.memory=MemoryDomain::Device;
        token_desc.persistent=false;token_desc.label="fp16-model-token-ids";
        std::shared_ptr<Buffer> token_buffer;result=context.backend->allocate(token_desc,token_buffer);if(!result.ok())return result;
        if(!token_buffer||token_buffer->size_bytes()<token_desc.size_bytes)
            return Status{StatusCode::BackendFailure,"Backend returned undersized 16-bit token-ID buffer"};
        result=context.backend->upload(token_buffer,0,context.token_ids.data(),token_desc.size_bytes);if(!result.ok())return result;
        if(backend==BackendKind::OpenGles)
            return bf16?build_bf16_embedding_gles_candidate(weight->second.buffer,weight->second.byte_size,vocabulary,dimension,token_buffer,context.token_ids,*output,dispatch)
                       :build_f16_embedding_gles_candidate(weight->second.buffer,weight->second.byte_size,vocabulary,dimension,token_buffer,context.token_ids,*output,dispatch);
        return bf16?build_bf16_embedding_candidate(weight->second.buffer,weight->second.byte_size,vocabulary,dimension,token_buffer,context.token_ids,*output,dispatch)
                   :build_f16_embedding_candidate(weight->second.buffer,weight->second.byte_size,vocabulary,dimension,token_buffer,context.token_ids,*output,dispatch);
    });
}


Status register_q6_k_matmul_lowering(BackendKind backend,ModelLoweringRegistry&registry){
    return registry.register_lowering(backend,"MatMul_Q6_K",[backend](
        const ModelPlanOperation&operation,const ModelExecutionContext&context,ComputeDispatch&dispatch){
        if(operation.inputs.size()!=1||operation.weight_tensors.size()!=1)
            return Status{StatusCode::InvalidArgument,"Q6_K MatMul requires one activation and one quantized weight"};
        const Tensor*input=nullptr,*output=nullptr;Status result=lookup_operand(operation,context,0,input);if(!result.ok())return result;
        result=lookup_output(operation,context,output);if(!result.ok())return result;
        const auto weight=context.quantized_weights.find(operation.weight_tensors.front());
        if(weight==context.quantized_weights.end()||!weight->second.buffer||weight->second.byte_size==0||
           weight->second.file_type_id!=14||weight->second.dimensions.size()!=2)
            return Status{StatusCode::InvalidArgument,"Q6_K MatMul weight binding is missing or invalid"};
        if(input->descriptor().shape.size()!=2||input->descriptor().shape[0]==0||
           (output->descriptor().shape.size()!=2&&output->descriptor().shape.size()!=3))
            return Status{StatusCode::Unsupported,"Q6_K MatMul activation/output shapes are unsupported"};
        const uint64_t channels=output->element_count()/input->descriptor().shape[0];
        if(weight->second.dimensions[0]!=input->descriptor().shape[1]||weight->second.dimensions[1]!=channels)
            return Status{StatusCode::InvalidArgument,"Q6_K weight dimensions do not match activation/output"};
        return backend==BackendKind::OpenGles
            ?build_q6_k_matmul_gles_candidate(*input,weight->second.buffer,weight->second.byte_size,*output,dispatch)
            :build_q6_k_matmul_candidate(*input,weight->second.buffer,weight->second.byte_size,*output,dispatch);
    });
}


Status register_k4_matmul_lowering(BackendKind backend,ModelLoweringRegistry&registry,bool q5){
    const std::string type=q5?"Q5_K":"Q4_K",op=q5?"MatMul_Q5_K":"MatMul_Q4_K";
    const uint32_t type_id=q5?13u:12u;
    return registry.register_lowering(backend,op,[backend,type,type_id,q5](
        const ModelPlanOperation&operation,const ModelExecutionContext&context,ComputeDispatch&dispatch){
        if(operation.inputs.size()!=1||operation.weight_tensors.size()!=1)
            return Status{StatusCode::InvalidArgument,type+" MatMul requires one activation and one quantized weight"};
        const Tensor*input=nullptr,*output=nullptr;Status result=lookup_operand(operation,context,0,input);if(!result.ok())return result;
        result=lookup_output(operation,context,output);if(!result.ok())return result;
        const auto weight=context.quantized_weights.find(operation.weight_tensors.front());
        if(weight==context.quantized_weights.end()||!weight->second.buffer||weight->second.byte_size==0||
           weight->second.file_type_id!=type_id||weight->second.dimensions.size()!=2)
            return Status{StatusCode::InvalidArgument,type+" MatMul weight binding is missing or invalid"};
        if(input->descriptor().shape.size()!=2||input->descriptor().shape[0]==0||
           (output->descriptor().shape.size()!=2&&output->descriptor().shape.size()!=3))
            return Status{StatusCode::Unsupported,type+" MatMul activation/output shapes are unsupported"};
        const uint64_t channels=output->element_count()/input->descriptor().shape[0];
        if(weight->second.dimensions[0]!=input->descriptor().shape[1]||weight->second.dimensions[1]!=channels)
            return Status{StatusCode::InvalidArgument,type+" weight dimensions do not match activation/output"};
        if(backend==BackendKind::OpenGles)
            return q5?build_q5_k_matmul_gles_candidate(*input,weight->second.buffer,weight->second.byte_size,*output,dispatch)
                     :build_q4_k_matmul_gles_candidate(*input,weight->second.buffer,weight->second.byte_size,*output,dispatch);
        return q5?build_q5_k_matmul_candidate(*input,weight->second.buffer,weight->second.byte_size,*output,dispatch)
                 :build_q4_k_matmul_candidate(*input,weight->second.buffer,weight->second.byte_size,*output,dispatch);
    });
}


Status register_k23_matmul_lowering(BackendKind backend,ModelLoweringRegistry&registry,bool q3){
    const std::string type=q3?"Q3_K":"Q2_K",op=q3?"MatMul_Q3_K":"MatMul_Q2_K";
    const uint32_t type_id=q3?11u:10u;
    return registry.register_lowering(backend,op,[backend,type,type_id,q3](
        const ModelPlanOperation&operation,const ModelExecutionContext&context,ComputeDispatch&dispatch){
        if(operation.inputs.size()!=1||operation.weight_tensors.size()!=1)
            return Status{StatusCode::InvalidArgument,type+" MatMul requires one activation and one quantized weight"};
        const Tensor*input=nullptr,*output=nullptr;Status result=lookup_operand(operation,context,0,input);if(!result.ok())return result;
        result=lookup_output(operation,context,output);if(!result.ok())return result;
        const auto weight=context.quantized_weights.find(operation.weight_tensors.front());
        if(weight==context.quantized_weights.end()||!weight->second.buffer||weight->second.byte_size==0||
           weight->second.file_type_id!=type_id||weight->second.dimensions.size()!=2)
            return Status{StatusCode::InvalidArgument,type+" MatMul weight binding is missing or invalid"};
        if(input->descriptor().shape.size()!=2||input->descriptor().shape[0]==0||
           (output->descriptor().shape.size()!=2&&output->descriptor().shape.size()!=3))
            return Status{StatusCode::Unsupported,type+" MatMul activation/output shapes are unsupported"};
        const uint64_t channels=output->element_count()/input->descriptor().shape[0];
        if(weight->second.dimensions[0]!=input->descriptor().shape[1]||weight->second.dimensions[1]!=channels)
            return Status{StatusCode::InvalidArgument,type+" weight dimensions do not match activation/output"};
        if(backend==BackendKind::OpenGles)
            return q3?build_q3_k_matmul_gles_candidate(*input,weight->second.buffer,weight->second.byte_size,*output,dispatch)
                     :build_q2_k_matmul_gles_candidate(*input,weight->second.buffer,weight->second.byte_size,*output,dispatch);
        return q3?build_q3_k_matmul_candidate(*input,weight->second.buffer,weight->second.byte_size,*output,dispatch)
                 :build_q2_k_matmul_candidate(*input,weight->second.buffer,weight->second.byte_size,*output,dispatch);
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

    status = registry.register_lowering(backend, "MatMul_Q8_0",
        [backend](const ModelPlanOperation& operation, const ModelExecutionContext& context,
                  ComputeDispatch& dispatch) {
            if (operation.inputs.size() != 1 || operation.weight_tensors.size() != 1)
                return Status{StatusCode::InvalidArgument, "Q8_0 MatMul requires one activation and one quantized weight"};
            const Tensor *input = nullptr, *output = nullptr;
            Status result = lookup_operand(operation, context, 0, input);
            if (!result.ok()) return result;
            result = lookup_output(operation, context, output);
            if (!result.ok()) return result;
            const auto weight = context.quantized_weights.find(operation.weight_tensors.front());
            if (weight == context.quantized_weights.end() || !weight->second.buffer ||
                weight->second.byte_size == 0 ||
                GgufReader::tensor_type_info(weight->second.file_type_id).name != "Q8_0")
                return Status{StatusCode::InvalidArgument, "Q8_0 weight binding is missing or has the wrong file type"};
            if (input->descriptor().shape.size() != 2 ||
                (output->descriptor().shape.size() != 2 && output->descriptor().shape.size() != 3))
                return Status{StatusCode::Unsupported, "Q8_0 MatMul requires [M,K] activations and [M,N] or [M,H,D] output"};
            const uint64_t output_channels = output->element_count() / input->descriptor().shape[0];
            if (weight->second.dimensions.size() != 2 ||
                weight->second.dimensions[0] != input->descriptor().shape[1] ||
                weight->second.dimensions[1] != output_channels)
                return Status{StatusCode::InvalidArgument, "Q8_0 weight tensor dimensions do not match activation and output shapes"};
            return backend == BackendKind::OpenGles
                ? build_q8_0_matmul_gles_candidate(*input, weight->second.buffer,
                    weight->second.byte_size, *output, dispatch)
                : build_q8_0_matmul_candidate(*input, weight->second.buffer,
                    weight->second.byte_size, *output, dispatch);
        });
    if (!status.ok()) return status;
    status = registry.register_lowering(backend, "MatMul_Q4_0",
        [backend](const ModelPlanOperation& operation, const ModelExecutionContext& context,
                  ComputeDispatch& dispatch) {
            if (operation.inputs.size() != 1 || operation.weight_tensors.size() != 1)
                return Status{StatusCode::InvalidArgument, "Q4_0 MatMul requires one activation and one quantized weight"};
            const Tensor *input = nullptr, *output = nullptr;
            Status result = lookup_operand(operation, context, 0, input);
            if (!result.ok()) return result;
            result = lookup_output(operation, context, output);
            if (!result.ok()) return result;
            const auto weight = context.quantized_weights.find(operation.weight_tensors.front());
            if (weight == context.quantized_weights.end() || !weight->second.buffer ||
                weight->second.byte_size == 0 ||
                GgufReader::tensor_type_info(weight->second.file_type_id).name != "Q4_0")
                return Status{StatusCode::InvalidArgument, "Q4_0 weight binding is missing or has the wrong file type"};
            if (input->descriptor().shape.size() != 2 ||
                (output->descriptor().shape.size() != 2 && output->descriptor().shape.size() != 3))
                return Status{StatusCode::Unsupported, "Q4_0 MatMul requires [M,K] activations and [M,N] or [M,H,D] output"};
            const uint64_t output_channels = output->element_count() / input->descriptor().shape[0];
            if (weight->second.dimensions.size() != 2 ||
                weight->second.dimensions[0] != input->descriptor().shape[1] ||
                weight->second.dimensions[1] != output_channels)
                return Status{StatusCode::InvalidArgument, "Q4_0 weight tensor dimensions do not match activation and output shapes"};
            return backend == BackendKind::OpenGles
                ? build_q4_0_matmul_gles_candidate(*input, weight->second.buffer,
                    weight->second.byte_size, *output, dispatch)
                : build_q4_0_matmul_candidate(*input, weight->second.buffer,
                    weight->second.byte_size, *output, dispatch);
        });
    if (!status.ok()) return status;
    status = registry.register_lowering(backend, "MatMul_Q4_1",
        [backend](const ModelPlanOperation& operation, const ModelExecutionContext& context,
                  ComputeDispatch& dispatch) {
            if (operation.inputs.size() != 1 || operation.weight_tensors.size() != 1)
                return Status{StatusCode::InvalidArgument, "Q4_1 MatMul requires one activation and one quantized weight"};
            const Tensor *input = nullptr, *output = nullptr;
            Status result = lookup_operand(operation, context, 0, input);
            if (!result.ok()) return result;
            result = lookup_output(operation, context, output);
            if (!result.ok()) return result;
            const auto weight = context.quantized_weights.find(operation.weight_tensors.front());
            if (weight == context.quantized_weights.end() || !weight->second.buffer ||
                weight->second.byte_size == 0 ||
                GgufReader::tensor_type_info(weight->second.file_type_id).name != "Q4_1")
                return Status{StatusCode::InvalidArgument, "Q4_1 weight binding is missing or has the wrong file type"};
            if (input->descriptor().shape.size() != 2 ||
                (output->descriptor().shape.size() != 2 && output->descriptor().shape.size() != 3))
                return Status{StatusCode::Unsupported, "Q4_1 MatMul requires [M,K] activations and [M,N] or [M,H,D] output"};
            const uint64_t output_channels = output->element_count() / input->descriptor().shape[0];
            if (weight->second.dimensions.size() != 2 ||
                weight->second.dimensions[0] != input->descriptor().shape[1] ||
                weight->second.dimensions[1] != output_channels)
                return Status{StatusCode::InvalidArgument, "Q4_1 weight tensor dimensions do not match activation and output shapes"};
            return backend == BackendKind::OpenGles
                ? build_q4_1_matmul_gles_candidate(*input, weight->second.buffer,
                    weight->second.byte_size, *output, dispatch)
                : build_q4_1_matmul_candidate(*input, weight->second.buffer,
                    weight->second.byte_size, *output, dispatch);
        });
    if (!status.ok()) return status;
    status = register_q5_matmul_lowering(backend, registry, false);
    if (!status.ok()) return status;
    status = register_q5_matmul_lowering(backend, registry, true);
    if (!status.ok()) return status;
    status = register_q5_embedding_lowering(backend, registry, false);
    if (!status.ok()) return status;
    status = register_q5_embedding_lowering(backend, registry, true);
    if (!status.ok()) return status;
    status = register_fp16_matmul_lowering(backend, registry, false);
    if (!status.ok()) return status;
    status = register_fp16_matmul_lowering(backend, registry, true);
    if (!status.ok()) return status;
    status = register_fp16_embedding_lowering(backend, registry, false);
    if (!status.ok()) return status;
    status = register_fp16_embedding_lowering(backend, registry, true);
    if (!status.ok()) return status;
    status = register_q6_k_matmul_lowering(backend, registry);
    if (!status.ok()) return status;
    status = register_k4_matmul_lowering(backend, registry, false);
    if (!status.ok()) return status;
    status = register_k4_matmul_lowering(backend, registry, true);
    if (!status.ok()) return status;
    status = register_k23_matmul_lowering(backend, registry, false);
    if (!status.ok()) return status;
    status = register_k23_matmul_lowering(backend, registry, true);
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

    for (const char* unary_name : {"SiLU", "Softmax", "Sigmoid", "Softplus"}) {
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
                if (std::string(unary_name) == "Softmax")
                    return backend == BackendKind::OpenGles
                        ? build_softmax_f32_gles_candidate(*input, *output, dispatch)
                        : build_softmax_f32_candidate(*input, *output, dispatch);
                if (std::string(unary_name) == "Sigmoid")
                    return backend == BackendKind::OpenGles
                        ? build_sigmoid_f32_gles_candidate(*input, *output, dispatch)
                        : build_sigmoid_f32_candidate(*input, *output, dispatch);
                return backend == BackendKind::OpenGles
                    ? build_softplus_f32_gles_candidate(*input, *output, dispatch)
                    : build_softplus_f32_candidate(*input, *output, dispatch);
            });
        if (!status.ok()) return status;
    }

    status = registry.register_lowering(backend, "Split",
        [backend](const ModelPlanOperation& operation, const ModelExecutionContext& context,
                  ComputeDispatch& dispatch) {
            if (operation.inputs.size() != 1 || !operation.weight_tensors.empty())
                return Status{StatusCode::InvalidArgument, "Split requires one activation and no weights"};
            const Tensor *input = nullptr, *output = nullptr;
            Status result = lookup_operand(operation, context, 0, input);
            if (!result.ok()) return result;
            result = lookup_output(operation, context, output);
            if (!result.ok()) return result;
            uint32_t axis = 0, offset = 0, length = 0;
            result = required_integer(operation, "axis", axis);
            if (!result.ok()) return result;
            result = required_integer(operation, "offset", offset);
            if (!result.ok()) return result;
            result = required_integer(operation, "length", length);
            if (!result.ok()) return result;
            if (axis >= output->descriptor().shape.size() ||
                output->descriptor().shape[axis] != length)
                return Status{StatusCode::InvalidArgument, "Split output length does not match its declared slice"};
            return backend == BackendKind::OpenGles
                ? build_split_f32_gles_candidate(*input, *output, axis, offset, dispatch)
                : build_split_f32_candidate(*input, *output, axis, offset, dispatch);
        });
    if (!status.ok()) return status;

    status = registry.register_lowering(backend, "GdnL2Norm",
        [backend](const ModelPlanOperation& operation, const ModelExecutionContext& context,
                  ComputeDispatch& dispatch) {
            if (operation.inputs.size() != 1 || !operation.weight_tensors.empty())
                return Status{StatusCode::InvalidArgument, "GdnL2Norm requires one activation and no weights"};
            const Tensor *input = nullptr, *output = nullptr;
            Status result = lookup_operand(operation, context, 0, input);
            if (!result.ok()) return result;
            result = lookup_output(operation, context, output);
            if (!result.ok()) return result;
            uint32_t state_size = 0, group_count = 0;
            float epsilon = 0.0f;
            result = required_integer(operation, "state_size", state_size);
            if (!result.ok()) return result;
            result = required_integer(operation, "group_count", group_count);
            if (!result.ok()) return result;
            result = required_float(operation, "epsilon", epsilon);
            if (!result.ok()) return result;
            return backend == BackendKind::OpenGles
                ? build_gdn_l2_norm_f32_gles_candidate(*input, *output, state_size,
                    group_count, epsilon, dispatch)
                : build_gdn_l2_norm_f32_candidate(*input, *output, state_size,
                    group_count, epsilon, dispatch);
        });
    if (!status.ok()) return status;

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


    status = registry.register_lowering(backend, "Embedding",
        [backend](const ModelPlanOperation& operation, const ModelExecutionContext& context,
                  ComputeDispatch& dispatch) {
            if (operation.inputs.size() != 1 || operation.inputs[0] != "token_ids" ||
                operation.weight_tensors.size() != 1 || !context.backend ||
                context.token_ids.empty() || context.token_ids.size() > UINT32_MAX)
                return Status{StatusCode::InvalidArgument, "Embedding requires token IDs, one weight tensor and an active backend"};
            const Tensor *weights = nullptr, *output = nullptr;
            Status result = lookup_operand(operation, context, 1, weights);
            if (!result.ok()) return result;
            result = lookup_output(operation, context, output);
            if (!result.ok()) return result;
            if (context.token_ids.size() > UINT64_MAX / sizeof(uint32_t))
                return Status{StatusCode::InvalidArgument, "Embedding token buffer size overflows"};
            BufferDesc token_desc;
            token_desc.size_bytes = context.token_ids.size() * sizeof(uint32_t);
            token_desc.alignment = alignof(uint32_t);
            token_desc.memory = MemoryDomain::Device;
            token_desc.persistent = false;
            token_desc.label = "model-token-ids";
            std::shared_ptr<Buffer> token_buffer;
            result = context.backend->allocate(token_desc, token_buffer);
            if (!result.ok()) return result;
            if (!token_buffer || token_buffer->size_bytes() < token_desc.size_bytes)
                return Status{StatusCode::BackendFailure, "Backend returned an undersized token-ID buffer"};
            result = context.backend->upload(token_buffer, 0, context.token_ids.data(), token_desc.size_bytes);
            if (!result.ok()) return result;
            return backend == BackendKind::OpenGles
                ? build_embedding_f32_gles_candidate(*weights, token_buffer, context.token_ids, *output, dispatch)
                : build_embedding_f32_candidate(*weights, token_buffer, context.token_ids, *output, dispatch);
        });
    if (!status.ok()) return status;

    status = registry.register_lowering(backend, "Embedding_Q4_0",
        [backend](const ModelPlanOperation& operation, const ModelExecutionContext& context,
                  ComputeDispatch& dispatch) {
            if (operation.inputs.size() != 1 || operation.inputs[0] != "token_ids" ||
                operation.weight_tensors.size() != 1 || !context.backend ||
                context.token_ids.empty() || context.token_ids.size() > UINT32_MAX)
                return Status{StatusCode::InvalidArgument, "Q4_0 Embedding requires token IDs, one quantized weight and an active backend"};
            const Tensor* output = nullptr;
            Status result = lookup_output(operation, context, output);
            if (!result.ok()) return result;
            const auto weight = context.quantized_weights.find(operation.weight_tensors.front());
            if (weight == context.quantized_weights.end() || !weight->second.buffer ||
                weight->second.byte_size == 0 || weight->second.file_type_id != 2 ||
                weight->second.dimensions.size() != 2 ||
                weight->second.dimensions[0] > UINT32_MAX || weight->second.dimensions[1] > UINT32_MAX)
                return Status{StatusCode::InvalidArgument, "Q4_0 embedding weight binding is missing or has invalid metadata"};
            const uint64_t embedding_dimension = weight->second.dimensions[0];
            const uint64_t vocabulary_size = weight->second.dimensions[1];
            if (output->descriptor().shape.size() != 2 ||
                output->descriptor().shape[0] != context.token_ids.size() ||
                output->descriptor().shape[1] != embedding_dimension)
                return Status{StatusCode::InvalidArgument, "Q4_0 embedding output shape does not match tokens and weight dimensions"};
            BufferDesc token_desc;
            token_desc.size_bytes = context.token_ids.size() * sizeof(uint32_t);
            token_desc.alignment = alignof(uint32_t);
            token_desc.memory = MemoryDomain::Device;
            token_desc.persistent = false;
            token_desc.label = "q4-0-model-token-ids";
            std::shared_ptr<Buffer> token_buffer;
            result = context.backend->allocate(token_desc, token_buffer);
            if (!result.ok()) return result;
            if (!token_buffer || token_buffer->size_bytes() < token_desc.size_bytes)
                return Status{StatusCode::BackendFailure, "Backend returned an undersized Q4_0 token-ID buffer"};
            result = context.backend->upload(token_buffer, 0, context.token_ids.data(), token_desc.size_bytes);
            if (!result.ok()) return result;
            return backend == BackendKind::OpenGles
                ? build_q4_0_embedding_gles_candidate(weight->second.buffer, weight->second.byte_size,
                    static_cast<uint32_t>(vocabulary_size), static_cast<uint32_t>(embedding_dimension),
                    token_buffer, context.token_ids, *output, dispatch)
                : build_q4_0_embedding_candidate(weight->second.buffer, weight->second.byte_size,
                    static_cast<uint32_t>(vocabulary_size), static_cast<uint32_t>(embedding_dimension),
                    token_buffer, context.token_ids, *output, dispatch);
        });
    if (!status.ok()) return status;
    status = registry.register_lowering(backend, "Embedding_Q8_0",
        [backend](const ModelPlanOperation& operation, const ModelExecutionContext& context,
                  ComputeDispatch& dispatch) {
            if (operation.inputs.size() != 1 || operation.inputs[0] != "token_ids" ||
                operation.weight_tensors.size() != 1 || !context.backend ||
                context.token_ids.empty() || context.token_ids.size() > UINT32_MAX)
                return Status{StatusCode::InvalidArgument, "Q8_0 Embedding requires token IDs, one quantized weight and an active backend"};
            const Tensor* output = nullptr;
            Status result = lookup_output(operation, context, output);
            if (!result.ok()) return result;
            const auto weight = context.quantized_weights.find(operation.weight_tensors.front());
            if (weight == context.quantized_weights.end() || !weight->second.buffer ||
                weight->second.byte_size == 0 || weight->second.file_type_id != 8 ||
                weight->second.dimensions.size() != 2 ||
                weight->second.dimensions[0] > UINT32_MAX || weight->second.dimensions[1] > UINT32_MAX)
                return Status{StatusCode::InvalidArgument, "Q8_0 embedding weight binding is missing or has invalid metadata"};
            const uint64_t embedding_dimension = weight->second.dimensions[0];
            const uint64_t vocabulary_size = weight->second.dimensions[1];
            if (output->descriptor().shape.size() != 2 ||
                output->descriptor().shape[0] != context.token_ids.size() ||
                output->descriptor().shape[1] != embedding_dimension)
                return Status{StatusCode::InvalidArgument, "Q8_0 embedding output shape does not match tokens and weight dimensions"};
            BufferDesc token_desc;
            token_desc.size_bytes = context.token_ids.size() * sizeof(uint32_t);
            token_desc.alignment = alignof(uint32_t);
            token_desc.memory = MemoryDomain::Device;
            token_desc.persistent = false;
            token_desc.label = "q8-0-model-token-ids";
            std::shared_ptr<Buffer> token_buffer;
            result = context.backend->allocate(token_desc, token_buffer);
            if (!result.ok()) return result;
            if (!token_buffer || token_buffer->size_bytes() < token_desc.size_bytes)
                return Status{StatusCode::BackendFailure, "Backend returned an undersized Q8_0 token-ID buffer"};
            result = context.backend->upload(token_buffer, 0, context.token_ids.data(), token_desc.size_bytes);
            if (!result.ok()) return result;
            return backend == BackendKind::OpenGles
                ? build_q8_0_embedding_gles_candidate(weight->second.buffer, weight->second.byte_size,
                    static_cast<uint32_t>(vocabulary_size), static_cast<uint32_t>(embedding_dimension),
                    token_buffer, context.token_ids, *output, dispatch)
                : build_q8_0_embedding_candidate(weight->second.buffer, weight->second.byte_size,
                    static_cast<uint32_t>(vocabulary_size), static_cast<uint32_t>(embedding_dimension),
                    token_buffer, context.token_ids, *output, dispatch);
        });
    if (!status.ok()) return status;
    status = registry.register_lowering(backend, "Embedding_Q4_1",
        [backend](const ModelPlanOperation& operation, const ModelExecutionContext& context,
                  ComputeDispatch& dispatch) {
            if (operation.inputs.size() != 1 || operation.inputs[0] != "token_ids" ||
                operation.weight_tensors.size() != 1 || !context.backend ||
                context.token_ids.empty() || context.token_ids.size() > UINT32_MAX)
                return Status{StatusCode::InvalidArgument, "Q4_1 Embedding requires token IDs, one quantized weight and an active backend"};
            const Tensor* output = nullptr;
            Status result = lookup_output(operation, context, output);
            if (!result.ok()) return result;
            const auto weight = context.quantized_weights.find(operation.weight_tensors.front());
            if (weight == context.quantized_weights.end() || !weight->second.buffer ||
                weight->second.byte_size == 0 || weight->second.file_type_id != 3 ||
                weight->second.dimensions.size() != 2 ||
                weight->second.dimensions[0] > UINT32_MAX || weight->second.dimensions[1] > UINT32_MAX)
                return Status{StatusCode::InvalidArgument, "Q4_1 embedding weight binding is missing or has invalid metadata"};
            const uint64_t embedding_dimension = weight->second.dimensions[0];
            const uint64_t vocabulary_size = weight->second.dimensions[1];
            if (output->descriptor().shape.size() != 2 ||
                output->descriptor().shape[0] != context.token_ids.size() ||
                output->descriptor().shape[1] != embedding_dimension)
                return Status{StatusCode::InvalidArgument, "Q4_1 embedding output shape does not match tokens and weight dimensions"};
            BufferDesc token_desc;
            token_desc.size_bytes = context.token_ids.size() * sizeof(uint32_t);
            token_desc.alignment = alignof(uint32_t);
            token_desc.memory = MemoryDomain::Device;
            token_desc.persistent = false;
            token_desc.label = "q4-1-model-token-ids";
            std::shared_ptr<Buffer> token_buffer;
            result = context.backend->allocate(token_desc, token_buffer);
            if (!result.ok()) return result;
            if (!token_buffer || token_buffer->size_bytes() < token_desc.size_bytes)
                return Status{StatusCode::BackendFailure, "Backend returned an undersized Q4_1 token-ID buffer"};
            result = context.backend->upload(token_buffer, 0, context.token_ids.data(), token_desc.size_bytes);
            if (!result.ok()) return result;
            return backend == BackendKind::OpenGles
                ? build_q4_1_embedding_gles_candidate(weight->second.buffer, weight->second.byte_size,
                    static_cast<uint32_t>(vocabulary_size), static_cast<uint32_t>(embedding_dimension),
                    token_buffer, context.token_ids, *output, dispatch)
                : build_q4_1_embedding_candidate(weight->second.buffer, weight->second.byte_size,
                    static_cast<uint32_t>(vocabulary_size), static_cast<uint32_t>(embedding_dimension),
                    token_buffer, context.token_ids, *output, dispatch);
        });
    if (!status.ok()) return status;

    status = registry.register_lowering(backend, "KVCacheAppend",
        [backend](const ModelPlanOperation& operation, const ModelExecutionContext& context,
                  ComputeDispatch& dispatch) {
            if (!context.kv_cache || context.past_length > UINT32_MAX ||
                operation.inputs.size() != 2 || !operation.weight_tensors.empty())
                return Status{StatusCode::InvalidArgument, "KVCacheAppend requires a cache and K/V activation tensors"};
            uint32_t layer = 0;
            Status result = required_integer(operation, "layer_index", layer);
            if (!result.ok()) return result;
            const Tensor *keys = nullptr, *values = nullptr;
            result = lookup_operand(operation, context, 0, keys);
            if (!result.ok()) return result;
            result = lookup_operand(operation, context, 1, values);
            if (!result.ok()) return result;
            if (keys->descriptor().shape.empty() ||
                keys->descriptor().shape[0] > UINT32_MAX ||
                (context.sequence_length != 0 &&
                 keys->descriptor().shape[0] != context.sequence_length))
                return Status{StatusCode::InvalidArgument, "KVCacheAppend token count does not match the execution context"};
            std::vector<BufferCopy> copies;
            result = context.kv_cache->prepare_device_append(
                layer, static_cast<uint32_t>(context.past_length),
                static_cast<uint32_t>(keys->descriptor().shape[0]), *keys, *values, copies);
            if (!result.ok()) return result;
            dispatch = ComputeDispatch{};
            dispatch.label = "kv-cache-append-device-copy";
            dispatch.kernel_id = "internal-kv-cache-append";
            dispatch.backend = backend;
            dispatch.copy_only = true;
            dispatch.buffer_copies = std::move(copies);
            return Status::success();
        });
    if (!status.ok()) return status;

    status = registry.register_lowering(backend, "Attention",
        [backend](const ModelPlanOperation& operation, const ModelExecutionContext& context,
                  ComputeDispatch& dispatch) {
            if (operation.inputs.size() != 3 || !operation.weight_tensors.empty() ||
                context.past_length > UINT32_MAX)
                return Status{StatusCode::InvalidArgument, "Attention requires Q and valid-prefix K/V cache tensors"};
            const Tensor *query = nullptr, *key = nullptr, *value = nullptr, *output = nullptr;
            Status result = lookup_operand(operation, context, 0, query);
            if (!result.ok()) return result;
            result = lookup_operand(operation, context, 1, key);
            if (!result.ok()) return result;
            result = lookup_operand(operation, context, 2, value);
            if (!result.ok()) return result;
            result = lookup_output(operation, context, output);
            if (!result.ok()) return result;
            return backend == BackendKind::OpenGles
                ? build_attention_f32_gles_candidate(*query, *key, *value, *output,
                    static_cast<uint32_t>(context.past_length), dispatch)
                : build_attention_f32_candidate(*query, *key, *value, *output,
                    static_cast<uint32_t>(context.past_length), dispatch);
        });
    if (!status.ok()) return status;

    // Attention remains unregistered until KV-cache append/read semantics are
    // sequenced with the graph and validated on each backend.
    return Status::success();
}

} // namespace ai_chat::gpu_engine
