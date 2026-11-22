#include "operators.h"

#include <cmath>
#include <string>
#include <utility>

namespace ai_chat::gpu_engine {
namespace {

constexpr char kGdnL2NormWgsl[] = R"wgsl(
@group(0) @binding(0) var<storage, read> input_values: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_values: array<f32>;
const STATE_SIZE: u32 = 1u;
const GROUP_COUNT: u32 = 1u;
const TOTAL_GROUPS: u32 = 1u;
const EPSILON: f32 = 0.00001;
var<workgroup> partial: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let task = wid.x;
    if (task >= TOTAL_GROUPS) { return; }
    let lane = lid.x;
    let token = task / GROUP_COUNT;
    let group = task % GROUP_COUNT;
    let base = token * GROUP_COUNT * STATE_SIZE + group * STATE_SIZE;
    var sum = 0.0;
    for (var d = lane; d < STATE_SIZE; d = d + 64u) {
        let x = input_values[base + d];
        sum = sum + x * x;
    }
    partial[lane] = sum;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) { partial[lane] = partial[lane] + partial[lane + stride]; }
        workgroupBarrier();
    }
    let inverse_norm = inverseSqrt(partial[0] + EPSILON);
    for (var d = lane; d < STATE_SIZE; d = d + 64u) {
        output_values[base + d] = input_values[base + d] * inverse_norm;
    }
}
)wgsl";

constexpr char kGdnL2NormGlsl[] = R"glsl(#version 310 es
precision highp float;
precision highp int;
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer Input { float input_values[]; };
layout(std430, binding = 1) writeonly buffer Output { float output_values[]; };
const uint STATE_SIZE = 1u;
const uint GROUP_COUNT = 1u;
const uint TOTAL_GROUPS = 1u;
const float EPSILON = 0.00001;
shared float partial[64];
void main() {
    uint task = gl_WorkGroupID.x;
    if (task >= TOTAL_GROUPS) return;
    uint lane = gl_LocalInvocationID.x;
    uint token = task / GROUP_COUNT;
    uint group = task % GROUP_COUNT;
    uint base = token * GROUP_COUNT * STATE_SIZE + group * STATE_SIZE;
    float sum = 0.0;
    for (uint d = lane; d < STATE_SIZE; d += 64u) {
        float x = input_values[base + d];
        sum += x * x;
    }
    partial[lane] = sum;
    barrier();
    for (uint stride = 32u; stride > 0u; stride >>= 1u) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        barrier();
    }
    float inverse_norm = inversesqrt(partial[0] + EPSILON);
    for (uint d = lane; d < STATE_SIZE; d += 64u)
        output_values[base + d] = input_values[base + d] * inverse_norm;
}
)glsl";

bool contiguous_f32(const Tensor& tensor) {
    return tensor.descriptor().data_type == DataType::F32 &&
           tensor.descriptor().layout == TensorLayout::Contiguous &&
           static_cast<bool>(tensor.buffer());
}

Status build_gdn_l2_norm_impl(const Tensor& input, const Tensor& output,
                              uint32_t state_size, uint32_t group_count,
                              float epsilon, bool gles,
                              ComputeDispatch& dispatch) {
    dispatch = ComputeDispatch{};
    if (!contiguous_f32(input) || !contiguous_f32(output))
        return {StatusCode::Unsupported, "GDN L2 normalization requires contiguous F32 tensors"};
    const auto& shape = input.descriptor().shape;
    if (shape.size() != 2 || output.descriptor().shape != shape ||
        shape[0] == 0 || shape[1] == 0 || state_size == 0 || group_count == 0 ||
        static_cast<uint64_t>(state_size) * group_count != shape[1] ||
        shape[0] > UINT32_MAX / group_count ||
        input.element_count() > UINT32_MAX || !std::isfinite(epsilon) || epsilon <= 0.0f ||
        !input.buffer() || !output.buffer() || input.buffer().get() == output.buffer().get())
        return {StatusCode::InvalidArgument, "GDN L2 normalization dimensions or epsilon are invalid"};

    dispatch.backend = gles ? BackendKind::OpenGles : BackendKind::DawnWebGpu;
    dispatch.kernel_id = gles ? "gles-gdn-l2-norm-f32-v1" : "dawn-gdn-l2-norm-f32-v1";
    dispatch.label = gles ? "gles-gdn-l2-norm-f32-unverified" : "dawn-gdn-l2-norm-f32-unverified";
    dispatch.workgroup_size_x = 64;
    std::string& shader = gles ? dispatch.glsl_source : dispatch.wgsl_source;
    shader = gles ? kGdnL2NormGlsl : kGdnL2NormWgsl;
    const std::pair<std::string, std::string> constants[] = {
        {gles ? "const uint STATE_SIZE = 1u;" : "const STATE_SIZE: u32 = 1u;",
         std::string(gles ? "const uint STATE_SIZE = " : "const STATE_SIZE: u32 = ") + std::to_string(state_size) + "u;"},
        {gles ? "const uint GROUP_COUNT = 1u;" : "const GROUP_COUNT: u32 = 1u;",
         std::string(gles ? "const uint GROUP_COUNT = " : "const GROUP_COUNT: u32 = ") + std::to_string(group_count) + "u;"},
        {gles ? "const uint TOTAL_GROUPS = 1u;" : "const TOTAL_GROUPS: u32 = 1u;",
         std::string(gles ? "const uint TOTAL_GROUPS = " : "const TOTAL_GROUPS: u32 = ") + std::to_string(shape[0] * group_count) + "u;"}
    };
    for (const auto& item : constants) {
        const size_t pos = shader.find(item.first);
        if (pos == std::string::npos)
            return {StatusCode::BackendFailure, "GDN L2 normalization shader constant is missing"};
        shader.replace(pos, item.first.size(), item.second);
    }
    const std::string epsilon_key = gles ? "const float EPSILON = 0.00001;" : "const EPSILON: f32 = 0.00001;";
    const std::string epsilon_value = std::string(gles ? "const float EPSILON = " : "const EPSILON: f32 = ") +
        std::to_string(epsilon) + ";";
    const size_t epsilon_pos = shader.find(epsilon_key);
    if (epsilon_pos == std::string::npos)
        return {StatusCode::BackendFailure, "GDN L2 normalization epsilon placeholder is missing"};
    shader.replace(epsilon_pos, epsilon_key.size(), epsilon_value);
    dispatch.workgroups_x = static_cast<uint32_t>(shape[0] * group_count);
    dispatch.bindings = {
        {0, input.buffer(), input.byte_offset(), input.byte_size()},
        {1, output.buffer(), output.byte_offset(), output.byte_size()}
    };
    return Status::success();
}

} // namespace

Status build_gdn_l2_norm_f32_candidate(const Tensor& input, const Tensor& output,
                                       uint32_t state_size, uint32_t group_count,
                                       float epsilon, ComputeDispatch& dispatch) {
    return build_gdn_l2_norm_impl(input, output, state_size, group_count, epsilon, false, dispatch);
}
Status build_gdn_l2_norm_f32_gles_candidate(const Tensor& input, const Tensor& output,
                                            uint32_t state_size, uint32_t group_count,
                                            float epsilon, ComputeDispatch& dispatch) {
    return build_gdn_l2_norm_impl(input, output, state_size, group_count, epsilon, true, dispatch);
}
Status reference_gdn_l2_norm_f32(const float* input, float* output,
                                 uint32_t tokens, uint32_t state_size,
                                 uint32_t group_count, float epsilon) {
    if ((!input || !output) && tokens != 0)
        return {StatusCode::InvalidArgument, "GDN L2 reference received a null pointer"};
    if (tokens == 0 || state_size == 0 || group_count == 0 ||
        !std::isfinite(epsilon) || epsilon <= 0.0f)
        return {StatusCode::InvalidArgument, "GDN L2 reference dimensions or epsilon are invalid"};
    const uint64_t width = static_cast<uint64_t>(state_size) * group_count;
    for (uint32_t token = 0; token < tokens; ++token) {
        for (uint32_t group = 0; group < group_count; ++group) {
            const uint64_t base = static_cast<uint64_t>(token) * width +
                                  static_cast<uint64_t>(group) * state_size;
            float sum = 0.0f;
            for (uint32_t d = 0; d < state_size; ++d) {
                const float value = input[base + d];
                sum += value * value;
            }
            const float inverse_norm = 1.0f / std::sqrt(sum + epsilon);
            for (uint32_t d = 0; d < state_size; ++d)
                output[base + d] = input[base + d] * inverse_norm;
        }
    }
    return Status::success();
}

Status register_gdn_l2_norm_operator_candidate(OperatorRegistry& registry, BackendKind backend) {
    KernelDefinition definition;
    definition.operator_name = "GdnL2Norm";
    definition.backend = backend;
    definition.input_type = DataType::F32;
    definition.output_type = DataType::F32;
    definition.min_input_count = 1;
    definition.max_input_count = 1;
    definition.min_rank = 2;
    definition.max_rank = 2;
    definition.validation = KernelValidation::Unverified;
    if (backend == BackendKind::OpenGles) {
        definition.kernel_id = "gles-gdn-l2-norm-f32-v1";
        definition.glsl_source = kGdnL2NormGlsl;
    } else if (backend == BackendKind::DawnWebGpu) {
        definition.kernel_id = "dawn-gdn-l2-norm-f32-v1";
        definition.wgsl_source = kGdnL2NormWgsl;
    } else {
        return {StatusCode::Unsupported, "GDN L2 normalization backend is unsupported"};
    }
    return registry.register_kernel(std::move(definition));
}

} // namespace ai_chat::gpu_engine
