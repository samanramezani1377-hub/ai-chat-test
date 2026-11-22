#include "operators.h"

#include <limits>
#include <string>
#include <utility>

namespace ai_chat::gpu_engine {
namespace {

constexpr char kSplitGlsl[] = R"glsl(#version 310 es
precision highp float;
precision highp int;
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer Input { float input_values[]; };
layout(std430, binding = 1) writeonly buffer Output { float output_values[]; };
const uint TOTAL_ELEMENTS = 1u;
const uint INPUT_AXIS_DIM = 1u;
const uint OUTPUT_AXIS_DIM = 1u;
const uint INNER_SIZE = 1u;
const uint OFFSET = 0u;
void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= TOTAL_ELEMENTS) return;
    uint outer_index = index / (OUTPUT_AXIS_DIM * INNER_SIZE);
    uint axis_index = (index / INNER_SIZE) % OUTPUT_AXIS_DIM;
    uint inner_index = index % INNER_SIZE;
    uint input_index = outer_index * (INPUT_AXIS_DIM * INNER_SIZE) +
                       (axis_index + OFFSET) * INNER_SIZE + inner_index;
    output_values[index] = input_values[input_index];
}
)glsl";

constexpr char kSplitWgsl[] = R"wgsl(
@group(0) @binding(0) var<storage, read> input_values: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_values: array<f32>;
const TOTAL_ELEMENTS: u32 = 1u;
const INPUT_AXIS_DIM: u32 = 1u;
const OUTPUT_AXIS_DIM: u32 = 1u;
const INNER_SIZE: u32 = 1u;
const OFFSET: u32 = 0u;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= TOTAL_ELEMENTS) { return; }
    let outer_index = index / (OUTPUT_AXIS_DIM * INNER_SIZE);
    let axis_index = (index / INNER_SIZE) % OUTPUT_AXIS_DIM;
    let inner_index = index % INNER_SIZE;
    let input_index = outer_index * (INPUT_AXIS_DIM * INNER_SIZE) +
                      (axis_index + OFFSET) * INNER_SIZE + inner_index;
    output_values[index] = input_values[input_index];
}
)wgsl";

bool contiguous_f32(const Tensor& tensor) {
    return tensor.descriptor().data_type == DataType::F32 &&
           tensor.descriptor().layout == TensorLayout::Contiguous &&
           static_cast<bool>(tensor.buffer());
}

Status build_split_impl(const Tensor& input, const Tensor& output,
                        uint32_t axis, uint32_t offset, bool gles,
                        ComputeDispatch& dispatch) {
    dispatch = ComputeDispatch{};
    if (!contiguous_f32(input) || !contiguous_f32(output))
        return {StatusCode::Unsupported, "Split requires contiguous F32 tensors"};
    const auto& in_shape = input.descriptor().shape;
    const auto& out_shape = output.descriptor().shape;
    if (in_shape.empty() || in_shape.size() != out_shape.size() ||
        in_shape.size() > 8 || axis >= in_shape.size() ||
        input.element_count() == 0 || output.element_count() == 0 ||
        input.element_count() > UINT32_MAX || output.element_count() > UINT32_MAX ||
        !input.buffer() || !output.buffer() || input.buffer().get() == output.buffer().get())
        return {StatusCode::InvalidArgument, "Split rank, size, axis or buffer is invalid"};

    uint64_t inner = 1;
    for (size_t i = 0; i < in_shape.size(); ++i) {
        if (in_shape[i] == 0 || out_shape[i] == 0)
            return {StatusCode::InvalidArgument, "Split dimensions must be non-zero"};
        if (i != axis && in_shape[i] != out_shape[i])
            return {StatusCode::InvalidArgument, "Split may change only the selected axis dimension"};
        if (i > axis) {
            if (inner > UINT32_MAX / in_shape[i])
                return {StatusCode::InvalidArgument, "Split inner dimension overflows"};
            inner *= in_shape[i];
        } else if (i == axis) {
            if (out_shape[i] > in_shape[i] || offset > in_shape[i] - out_shape[i])
                return {StatusCode::InvalidArgument, "Split offset and length exceed the source axis"};
        } else if (in_shape[i] != out_shape[i]) {
            return {StatusCode::InvalidArgument, "Split may change only the selected axis dimension"};
        }
        if (i == axis && out_shape[i] > in_shape[i])
            return {StatusCode::InvalidArgument, "Split output axis is larger than the source axis"};
    }
    uint64_t outer = 1;
    for (size_t i = 0; i < axis; ++i) {
        if (outer > UINT32_MAX / in_shape[i])
            return {StatusCode::InvalidArgument, "Split outer dimension overflows"};
        outer *= in_shape[i];
    }
    if (outer * in_shape[axis] * inner != input.element_count() ||
        outer * out_shape[axis] * inner != output.element_count())
        return {StatusCode::InvalidArgument, "Split shape products do not match tensor storage"};

    dispatch.backend = gles ? BackendKind::OpenGles : BackendKind::DawnWebGpu;
    dispatch.kernel_id = gles ? "gles-split-f32-v1" : "dawn-split-f32-v1";
    dispatch.label = gles ? "gles-split-f32-unverified" : "dawn-split-f32-unverified";
    dispatch.workgroup_size_x = 64;
    if (gles) {
        dispatch.glsl_source = kSplitGlsl;        const std::pair<std::string, std::string> constants[] = {
            {"const uint TOTAL_ELEMENTS = 1u;", "const uint TOTAL_ELEMENTS = " + std::to_string(output.element_count()) + "u;"},
            {"const uint INPUT_AXIS_DIM = 1u;", "const uint INPUT_AXIS_DIM = " + std::to_string(in_shape[axis]) + "u;"},
            {"const uint OUTPUT_AXIS_DIM = 1u;", "const uint OUTPUT_AXIS_DIM = " + std::to_string(out_shape[axis]) + "u;"},
            {"const uint INNER_SIZE = 1u;", "const uint INNER_SIZE = " + std::to_string(inner) + "u;"},
            {"const uint OFFSET = 0u;", "const uint OFFSET = " + std::to_string(offset) + "u;"}
        };
        for (const auto& item : constants) {
            const size_t pos = dispatch.glsl_source.find(item.first);
            if (pos == std::string::npos)
                return {StatusCode::BackendFailure, "Split GLSL specialization constant is missing"};
            dispatch.glsl_source.replace(pos, item.first.size(), item.second);
        }
    } else {
        dispatch.wgsl_source = kSplitWgsl;
        const std::pair<std::string, std::string> constants[] = {
            {"const TOTAL_ELEMENTS: u32 = 1u;", "const TOTAL_ELEMENTS: u32 = " + std::to_string(output.element_count()) + "u;"},
            {"const INPUT_AXIS_DIM: u32 = 1u;", "const INPUT_AXIS_DIM: u32 = " + std::to_string(in_shape[axis]) + "u;"},
            {"const OUTPUT_AXIS_DIM: u32 = 1u;", "const OUTPUT_AXIS_DIM: u32 = " + std::to_string(out_shape[axis]) + "u;"},
            {"const INNER_SIZE: u32 = 1u;", "const INNER_SIZE: u32 = " + std::to_string(inner) + "u;"},
            {"const OFFSET: u32 = 0u;", "const OFFSET: u32 = " + std::to_string(offset) + "u;"}
        };
        for (const auto& item : constants) {
            const size_t pos = dispatch.wgsl_source.find(item.first);
            if (pos == std::string::npos)
                return {StatusCode::BackendFailure, "Split WGSL specialization constant is missing"};
            dispatch.wgsl_source.replace(pos, item.first.size(), item.second);
        }
    }
    dispatch.workgroups_x = static_cast<uint32_t>((output.element_count() + 63) / 64);
    dispatch.bindings = {
        {0, input.buffer(), input.byte_offset(), input.byte_size()},
        {1, output.buffer(), output.byte_offset(), output.byte_size()}
    };
    return Status::success();
}

} // namespace

Status build_split_f32_candidate(const Tensor& input, const Tensor& output,
                                 uint32_t axis, uint32_t offset,
                                 ComputeDispatch& dispatch) {
    return build_split_impl(input, output, axis, offset, false, dispatch);
}

Status build_split_f32_gles_candidate(const Tensor& input, const Tensor& output,
                                      uint32_t axis, uint32_t offset,
                                      ComputeDispatch& dispatch) {
    return build_split_impl(input, output, axis, offset, true, dispatch);
}

Status reference_split_f32(const float* input, float* output,
                           const std::vector<uint64_t>& input_shape,
                           const std::vector<uint64_t>& output_shape,
                           uint32_t axis, uint32_t offset) {
    if (!input || !output || input_shape.empty() || input_shape.size() != output_shape.size() ||
        axis >= input_shape.size())
        return {StatusCode::InvalidArgument, "Split reference arguments are invalid"};
    uint64_t inner = 1, outer = 1, input_elements = 1, output_elements = 1;
    for (size_t i = 0; i < input_shape.size(); ++i) {
        if (input_shape[i] == 0 || output_shape[i] == 0)
            return {StatusCode::InvalidArgument, "Split reference dimensions must be non-zero"};
        if (i > axis) inner *= input_shape[i];
        else if (i < axis) outer *= input_shape[i];
        else if (output_shape[i] > input_shape[i] || offset > input_shape[i] - output_shape[i])
            return {StatusCode::InvalidArgument, "Split reference range exceeds source axis"};
        if (i != axis && input_shape[i] != output_shape[i])
            return {StatusCode::InvalidArgument, "Split reference may change only the selected axis"};
        input_elements *= input_shape[i];
        output_elements *= output_shape[i];
    }
    if (outer * input_shape[axis] * inner != input_elements ||
        outer * output_shape[axis] * inner != output_elements)
        return {StatusCode::InvalidArgument, "Split reference shape product mismatch"};
    for (uint64_t i = 0; i < output_elements; ++i) {
        const uint64_t outer_index = i / (output_shape[axis] * inner);
        const uint64_t axis_index = (i / inner) % output_shape[axis];
        const uint64_t inner_index = i % inner;
        const uint64_t source = outer_index * (input_shape[axis] * inner) +
            (axis_index + offset) * inner + inner_index;
        if (source >= input_elements)
            return {StatusCode::InvalidArgument, "Split reference index exceeds source"};
        output[i] = input[source];
    }
    return Status::success();
}

Status register_split_operator_candidate(OperatorRegistry& registry, BackendKind backend) {
    KernelDefinition definition;
    definition.operator_name = "Split";
    definition.backend = backend;
    definition.input_type = DataType::F32;
    definition.output_type = DataType::F32;
    definition.min_input_count = 1;
    definition.max_input_count = 1;
    definition.min_rank = 1;
    definition.max_rank = 8;
    definition.validation = KernelValidation::Unverified;
    if (backend == BackendKind::OpenGles) {
        definition.kernel_id = "gles-split-f32-v1";
        definition.glsl_source = kSplitGlsl;
    } else if (backend == BackendKind::DawnWebGpu) {
        definition.kernel_id = "dawn-split-f32-v1";
        definition.wgsl_source = kSplitWgsl;
    } else {
        return {StatusCode::Unsupported, "Split candidate backend is unsupported"};
    }
    return registry.register_kernel(std::move(definition));
}

} // namespace ai_chat::gpu_engine
