#include "operators.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace ai_chat::gpu_engine {
namespace {

constexpr char kSsmConvWgsl[] = R"wgsl(
@group(0) @binding(0) var<storage, read> input_values: array<f32>;
@group(0) @binding(1) var<storage, read_write> conv_state: array<f32>;
@group(0) @binding(2) var<storage, read> conv_weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_values: array<f32>;
const TOKENS: u32 = 1u;
const CHANNELS: u32 = 1u;
const KERNEL_SIZE: u32 = 2u;
const WORKGROUPS_X: u32 = 1u;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let channel = (wid.y * WORKGROUPS_X + wid.x) * 64u + lid.x;
    if (channel >= CHANNELS) { return; }
    let history = KERNEL_SIZE - 1u;
    for (var token = 0u; token < TOKENS; token = token + 1u) {
        var sum = 0.0;
        for (var tap = 0u; tap < KERNEL_SIZE; tap = tap + 1u) {
            let source_index = token + tap;
            var sample_value = 0.0;
            if (source_index < history) {
                sample_value = conv_state[channel * history + source_index];
            } else {
                sample_value = input_values[(source_index - history) * CHANNELS + channel];
            }
            sum = sum + sample_value * conv_weights[channel * KERNEL_SIZE + tap];
        }
        output_values[token * CHANNELS + channel] = sum;
    }
    for (var slot = 0u; slot < history; slot = slot + 1u) {
        let source_index = TOKENS + slot;
        if (source_index < history) {
            conv_state[channel * history + slot] = conv_state[channel * history + source_index];
        } else {
            conv_state[channel * history + slot] =
                input_values[(source_index - history) * CHANNELS + channel];
        }
    }
}
)wgsl";

constexpr char kSsmConvGlsl[] = R"glsl(#version 310 es
precision highp float;
precision highp int;
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer Input { float input_values[]; };
layout(std430, binding = 1) buffer ConvState { float conv_state[]; };
layout(std430, binding = 2) readonly buffer ConvWeights { float conv_weights[]; };
layout(std430, binding = 3) writeonly buffer Output { float output_values[]; };
const uint TOKENS = 1u;
const uint CHANNELS = 1u;
const uint KERNEL_SIZE = 2u;
const uint WORKGROUPS_X = 1u;
void main() {
    uint channel = (gl_WorkGroupID.y * WORKGROUPS_X + gl_WorkGroupID.x) * 64u + gl_LocalInvocationID.x;
    if (channel >= CHANNELS) return;
    uint history = KERNEL_SIZE - 1u;
    for (uint token = 0u; token < TOKENS; ++token) {
        float sum = 0.0;
        for (uint tap = 0u; tap < KERNEL_SIZE; ++tap) {
            uint source_index = token + tap;
            float sample_value = source_index < history
                ? conv_state[channel * history + source_index]
                : input_values[(source_index - history) * CHANNELS + channel];
            sum += sample_value * conv_weights[channel * KERNEL_SIZE + tap];
        }
        output_values[token * CHANNELS + channel] = sum;
    }
    for (uint slot = 0u; slot < history; ++slot) {
        uint source_index = TOKENS + slot;
        conv_state[channel * history + slot] = source_index < history
            ? conv_state[channel * history + source_index]
            : input_values[(source_index - history) * CHANNELS + channel];
    }
}
)glsl";

constexpr char kGatedDeltaNetWgsl[] = R"wgsl(
@group(0) @binding(0) var<storage, read> q_values: array<f32>;
@group(0) @binding(1) var<storage, read> k_values: array<f32>;
@group(0) @binding(2) var<storage, read> v_values: array<f32>;
@group(0) @binding(3) var<storage, read> gate_values: array<f32>;
@group(0) @binding(4) var<storage, read> beta_values: array<f32>;
@group(0) @binding(5) var<storage, read_write> recurrent_state: array<f32>;
@group(0) @binding(6) var<storage, read_write> output_values: array<f32>;
const TOKENS: u32 = 1u;
const STATE_SIZE: u32 = 1u;
const VALUE_HEADS: u32 = 1u;
const KEY_HEADS: u32 = 1u;
const WORKGROUPS_X: u32 = 1u;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let task = (wid.y * WORKGROUPS_X + wid.x) * 64u + lid.x;
    let tasks = VALUE_HEADS * STATE_SIZE;
    if (task >= tasks) { return; }
    let head = task / STATE_SIZE;
    let column = task % STATE_SIZE;
    let key_head = head % KEY_HEADS;
    let state_base = head * STATE_SIZE * STATE_SIZE;
    for (var token = 0u; token < TOKENS; token = token + 1u) {
        let decay = exp(gate_values[token * VALUE_HEADS + head]);
        let beta = beta_values[token * VALUE_HEADS + head];
        var kv = 0.0;
        for (var row = 0u; row < STATE_SIZE; row = row + 1u) {
            let index = state_base + row * STATE_SIZE + column;
            recurrent_state[index] = recurrent_state[index] * decay;
            kv = kv + k_values[token * KEY_HEADS * STATE_SIZE + key_head * STATE_SIZE + row] * recurrent_state[index];
        }
        let delta = (v_values[token * VALUE_HEADS * STATE_SIZE + head * STATE_SIZE + column] - kv) * beta;
        var result = 0.0;
        for (var row = 0u; row < STATE_SIZE; row = row + 1u) {
            let index = state_base + row * STATE_SIZE + column;
            recurrent_state[index] = recurrent_state[index] +
                k_values[token * KEY_HEADS * STATE_SIZE + key_head * STATE_SIZE + row] * delta;
            result = result + q_values[token * KEY_HEADS * STATE_SIZE + key_head * STATE_SIZE + row] * recurrent_state[index];
        }
        output_values[token * VALUE_HEADS * STATE_SIZE + head * STATE_SIZE + column] = result * inverseSqrt(f32(STATE_SIZE));
    }
}
)wgsl";

constexpr char kGatedDeltaNetGlsl[] = R"glsl(#version 310 es
precision highp float;
precision highp int;
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer Query { float q_values[]; };
layout(std430, binding = 1) readonly buffer Key { float k_values[]; };
layout(std430, binding = 2) readonly buffer Value { float v_values[]; };
layout(std430, binding = 3) readonly buffer Gate { float gate_values[]; };
layout(std430, binding = 4) readonly buffer Beta { float beta_values[]; };
layout(std430, binding = 5) buffer RecurrentState { float recurrent_state[]; };
layout(std430, binding = 6) writeonly buffer Output { float output_values[]; };
const uint TOKENS = 1u;
const uint STATE_SIZE = 1u;
const uint VALUE_HEADS = 1u;
const uint KEY_HEADS = 1u;
const uint WORKGROUPS_X = 1u;
void main() {
    uint task = (gl_WorkGroupID.y * WORKGROUPS_X + gl_WorkGroupID.x) * 64u + gl_LocalInvocationID.x;
    uint tasks = VALUE_HEADS * STATE_SIZE;
    if (task >= tasks) return;
    uint head = task / STATE_SIZE;
    uint column = task % STATE_SIZE;
    uint key_head = head % KEY_HEADS;
    uint state_base = head * STATE_SIZE * STATE_SIZE;
    for (uint token = 0u; token < TOKENS; ++token) {
        float decay = exp(gate_values[token * VALUE_HEADS + head]);
        float beta = beta_values[token * VALUE_HEADS + head];
        float kv = 0.0;
        for (uint row = 0u; row < STATE_SIZE; ++row) {
            uint index = state_base + row * STATE_SIZE + column;
            recurrent_state[index] *= decay;
            kv += k_values[token * KEY_HEADS * STATE_SIZE + key_head * STATE_SIZE + row] * recurrent_state[index];
        }
        float delta = (v_values[token * VALUE_HEADS * STATE_SIZE + head * STATE_SIZE + column] - kv) * beta;
        float result = 0.0;
        for (uint row = 0u; row < STATE_SIZE; ++row) {
            uint index = state_base + row * STATE_SIZE + column;
            recurrent_state[index] += k_values[token * KEY_HEADS * STATE_SIZE + key_head * STATE_SIZE + row] * delta;
            result += q_values[token * KEY_HEADS * STATE_SIZE + key_head * STATE_SIZE + row] * recurrent_state[index];
        }
        output_values[token * VALUE_HEADS * STATE_SIZE + head * STATE_SIZE + column] = result * inversesqrt(float(STATE_SIZE));
    }
}
)glsl";

bool contiguous_f32(const Tensor& tensor) {
    return tensor.descriptor().data_type == DataType::F32 &&
           tensor.descriptor().layout == TensorLayout::Contiguous &&
           static_cast<bool>(tensor.buffer());
}

bool same_shape(const Tensor& tensor, const std::vector<uint64_t>& shape) {
    return tensor.descriptor().shape == shape;
}

Status specialize(std::string& shader, bool gles,
                  uint32_t tokens, uint32_t channels,
                  uint32_t kernel_size, uint32_t workgroups_x) {
    const std::pair<std::string, std::string> values[] = {
        {gles ? "const uint TOKENS = 1u;" : "const TOKENS: u32 = 1u;",
         std::string(gles ? "const uint TOKENS = " : "const TOKENS: u32 = ") + std::to_string(tokens) + "u;"},
        {gles ? "const uint CHANNELS = 1u;" : "const CHANNELS: u32 = 1u;",
         std::string(gles ? "const uint CHANNELS = " : "const CHANNELS: u32 = ") + std::to_string(channels) + "u;"},
        {gles ? "const uint KERNEL_SIZE = 2u;" : "const KERNEL_SIZE: u32 = 2u;",
         std::string(gles ? "const uint KERNEL_SIZE = " : "const KERNEL_SIZE: u32 = ") + std::to_string(kernel_size) + "u;"},
        {gles ? "const uint WORKGROUPS_X = 1u;" : "const WORKGROUPS_X: u32 = 1u;",
         std::string(gles ? "const uint WORKGROUPS_X = " : "const WORKGROUPS_X: u32 = ") + std::to_string(workgroups_x) + "u;"}
    };
    for (const auto& item : values) {
        const size_t position = shader.find(item.first);
        if (position == std::string::npos)
            return {StatusCode::BackendFailure, "SSMConv1D shader specialization constant is missing"};
        shader.replace(position, item.first.size(), item.second);
    }
    return Status::success();
}

Status specialize_gdn(std::string& shader, bool gles, uint32_t tokens,
                      uint32_t state_size, uint32_t value_heads,
                      uint32_t key_heads, uint32_t workgroups_x) {
    const std::pair<std::string, std::string> values[] = {
        {gles ? "const uint TOKENS = 1u;" : "const TOKENS: u32 = 1u;",
         std::string(gles ? "const uint TOKENS = " : "const TOKENS: u32 = ") + std::to_string(tokens) + "u;"},
        {gles ? "const uint STATE_SIZE = 1u;" : "const STATE_SIZE: u32 = 1u;",
         std::string(gles ? "const uint STATE_SIZE = " : "const STATE_SIZE: u32 = ") + std::to_string(state_size) + "u;"},
        {gles ? "const uint VALUE_HEADS = 1u;" : "const VALUE_HEADS: u32 = 1u;",
         std::string(gles ? "const uint VALUE_HEADS = " : "const VALUE_HEADS: u32 = ") + std::to_string(value_heads) + "u;"},
        {gles ? "const uint KEY_HEADS = 1u;" : "const KEY_HEADS: u32 = 1u;",
         std::string(gles ? "const uint KEY_HEADS = " : "const KEY_HEADS: u32 = ") + std::to_string(key_heads) + "u;"},
        {gles ? "const uint WORKGROUPS_X = 1u;" : "const WORKGROUPS_X: u32 = 1u;",
         std::string(gles ? "const uint WORKGROUPS_X = " : "const WORKGROUPS_X: u32 = ") + std::to_string(workgroups_x) + "u;"}
    };
    for (const auto& item : values) {
        const size_t position = shader.find(item.first);
        if (position == std::string::npos)
            return {StatusCode::BackendFailure, "GatedDeltaNet shader specialization constant is missing"};
        shader.replace(position, item.first.size(), item.second);
    }
    return Status::success();
}

Status build_ssm_conv_impl(const Tensor& input, const Tensor& state,
                           const Tensor& weights, const Tensor& output,
                           uint32_t kernel_size, bool gles,
                           ComputeDispatch& dispatch) {
    dispatch = ComputeDispatch{};
    if (!contiguous_f32(input) || !contiguous_f32(state) ||
        !contiguous_f32(weights) || !contiguous_f32(output))
        return {StatusCode::Unsupported, "SSMConv1D currently requires contiguous F32 tensors and weights"};
    const auto& in = input.descriptor().shape;
    const auto& st = state.descriptor().shape;
    const auto& wt = weights.descriptor().shape;
    if (in.size() != 2 || st.size() != 2 || wt.size() != 2 ||
        !same_shape(output, in) || in[0] == 0 || in[0] > UINT32_MAX ||
        in[1] == 0 || in[1] > UINT32_MAX || kernel_size < 2 ||
        st[0] != in[1] || st[1] != kernel_size - 1 ||
        wt[0] != kernel_size || wt[1] != in[1] ||
        input.buffer().get() == state.buffer().get() ||
        output.buffer().get() == state.buffer().get() ||
        output.buffer().get() == input.buffer().get() ||
        output.buffer().get() == weights.buffer().get())
        return {StatusCode::InvalidArgument, "SSMConv1D tensor dimensions or aliases are invalid"};
    const uint64_t groups = (in[1] + 63) / 64;
    const uint32_t workgroups_x = static_cast<uint32_t>(std::min<uint64_t>(groups, 65535));
    const uint64_t workgroups_y = (groups + workgroups_x - 1) / workgroups_x;
    if (workgroups_y > 65535)
        return {StatusCode::Unsupported, "SSMConv1D dispatch exceeds workgroup limits"};
    dispatch.backend = gles ? BackendKind::OpenGles : BackendKind::DawnWebGpu;
    dispatch.kernel_id = gles ? "gles-ssm-conv1d-f32-v1" : "dawn-ssm-conv1d-f32-v1";
    dispatch.label = dispatch.kernel_id;
    dispatch.workgroup_size_x = 64;
    dispatch.workgroups_x = workgroups_x;
    dispatch.workgroups_y = static_cast<uint32_t>(workgroups_y);
    std::string& shader = gles ? dispatch.glsl_source : dispatch.wgsl_source;
    shader = gles ? kSsmConvGlsl : kSsmConvWgsl;
    Status status = specialize(shader, gles, static_cast<uint32_t>(in[0]),
        static_cast<uint32_t>(in[1]), kernel_size, workgroups_x);
    if (!status.ok()) return status;
    dispatch.bindings = {
        {0, input.buffer(), input.byte_offset(), input.byte_size()},
        {1, state.buffer(), state.byte_offset(), state.byte_size()},
        {2, weights.buffer(), weights.byte_offset(), weights.byte_size()},
        {3, output.buffer(), output.byte_offset(), output.byte_size()}
    };
    return Status::success();
}

Status build_gdn_impl(const Tensor& q, const Tensor& k, const Tensor& v,
                      const Tensor& gate, const Tensor& beta, const Tensor& state,
                      const Tensor& output, uint32_t state_size,
                      uint32_t value_heads, uint32_t key_heads, bool gles,
                      ComputeDispatch& dispatch) {
    dispatch = ComputeDispatch{};
    if (!contiguous_f32(q) || !contiguous_f32(k) || !contiguous_f32(v) ||
        !contiguous_f32(gate) || !contiguous_f32(beta) ||
        !contiguous_f32(state) || !contiguous_f32(output))
        return {StatusCode::Unsupported, "GatedDeltaNet currently requires contiguous F32 tensors"};
    const auto& qs = q.descriptor().shape;
    const auto& ks = k.descriptor().shape;
    const auto& vs = v.descriptor().shape;
    const auto& gs = gate.descriptor().shape;
    const auto& bs = beta.descriptor().shape;
    const auto& ss = state.descriptor().shape;
    const uint64_t tokens = qs.size() == 2 ? qs[0] : 0;
    if (qs.size() != 2 || ks != qs || vs.size() != 2 ||
        gs.size() != 2 || bs != gs || ss.size() != 3 ||
        tokens == 0 || tokens > UINT32_MAX || state_size == 0 ||
        value_heads == 0 || key_heads == 0 || value_heads % key_heads != 0 ||
        qs[1] != static_cast<uint64_t>(key_heads) * state_size ||
        ks[1] != qs[1] || vs[0] != tokens ||
        vs[1] != static_cast<uint64_t>(value_heads) * state_size ||
        gs[0] != tokens || gs[1] != value_heads ||
        ss[0] != value_heads || ss[1] != state_size || ss[2] != state_size ||
        !same_shape(output, vs) ||
        q.buffer().get() == state.buffer().get() ||
        k.buffer().get() == state.buffer().get() ||
        v.buffer().get() == state.buffer().get() ||
        output.buffer().get() == state.buffer().get() ||
        output.buffer().get() == q.buffer().get() ||
        output.buffer().get() == k.buffer().get() ||
        output.buffer().get() == v.buffer().get())
        return {StatusCode::InvalidArgument, "GatedDeltaNet tensor dimensions or aliases are invalid"};
    const uint64_t tasks = static_cast<uint64_t>(value_heads) * state_size;
    const uint64_t groups = (tasks + 63) / 64;
    const uint32_t workgroups_x = static_cast<uint32_t>(std::min<uint64_t>(groups, 65535));
    const uint64_t workgroups_y = (groups + workgroups_x - 1) / workgroups_x;
    if (workgroups_y > 65535)
        return {StatusCode::Unsupported, "GatedDeltaNet dispatch exceeds workgroup limits"};
    dispatch.backend = gles ? BackendKind::OpenGles : BackendKind::DawnWebGpu;
    dispatch.kernel_id = gles ? "gles-gated-delta-net-f32-v1" : "dawn-gated-delta-net-f32-v1";
    dispatch.label = dispatch.kernel_id;
    dispatch.workgroup_size_x = 64;
    dispatch.workgroups_x = workgroups_x;
    dispatch.workgroups_y = static_cast<uint32_t>(workgroups_y);
    std::string& shader = gles ? dispatch.glsl_source : dispatch.wgsl_source;
    shader = gles ? kGatedDeltaNetGlsl : kGatedDeltaNetWgsl;
    Status status = specialize_gdn(shader, gles, static_cast<uint32_t>(tokens),
        state_size, value_heads, key_heads, workgroups_x);
    if (!status.ok()) return status;
    dispatch.bindings = {
        {0, q.buffer(), q.byte_offset(), q.byte_size()},
        {1, k.buffer(), k.byte_offset(), k.byte_size()},
        {2, v.buffer(), v.byte_offset(), v.byte_size()},
        {3, gate.buffer(), gate.byte_offset(), gate.byte_size()},
        {4, beta.buffer(), beta.byte_offset(), beta.byte_size()},
        {5, state.buffer(), state.byte_offset(), state.byte_size()},
        {6, output.buffer(), output.byte_offset(), output.byte_size()}
    };
    return Status::success();
}

} // namespace

Status build_ssm_conv1d_f32_candidate(const Tensor& input, const Tensor& state,
                                     const Tensor& weights, const Tensor& output,
                                     uint32_t kernel_size, ComputeDispatch& dispatch) {
    return build_ssm_conv_impl(input, state, weights, output, kernel_size, false, dispatch);
}
Status build_ssm_conv1d_f32_gles_candidate(const Tensor& input, const Tensor& state,
                                           const Tensor& weights, const Tensor& output,
                                           uint32_t kernel_size, ComputeDispatch& dispatch) {
    return build_ssm_conv_impl(input, state, weights, output, kernel_size, true, dispatch);
}
Status build_gated_delta_net_f32_candidate(const Tensor& q, const Tensor& k, const Tensor& v,
                                           const Tensor& gate, const Tensor& beta,
                                           const Tensor& state, const Tensor& output,
                                           uint32_t state_size, uint32_t value_heads,
                                           uint32_t key_heads, ComputeDispatch& dispatch) {
    return build_gdn_impl(q, k, v, gate, beta, state, output, state_size,
                          value_heads, key_heads, false, dispatch);
}
Status build_gated_delta_net_f32_gles_candidate(const Tensor& q, const Tensor& k, const Tensor& v,
                                                const Tensor& gate, const Tensor& beta,
                                                const Tensor& state, const Tensor& output,
                                                uint32_t state_size, uint32_t value_heads,
                                                uint32_t key_heads, ComputeDispatch& dispatch) {
    return build_gdn_impl(q, k, v, gate, beta, state, output, state_size,
                          value_heads, key_heads, true, dispatch);
}

Status reference_ssm_conv1d_f32(const float* input, float* state, const float* weights,
                                float* output, uint32_t tokens, uint32_t channels,
                                uint32_t kernel_size) {
    if (!input || !state || !weights || !output || tokens == 0 || channels == 0 ||
        kernel_size < 2)
        return {StatusCode::InvalidArgument, "SSMConv1D reference received invalid pointers or dimensions"};
    const uint32_t history = kernel_size - 1;
    std::vector<float> sequence(static_cast<size_t>(history + tokens) * channels);
    for (uint32_t channel = 0; channel < channels; ++channel) {
        for (uint32_t slot = 0; slot < history; ++slot)
            sequence[static_cast<size_t>(slot) * channels + channel] =
                state[static_cast<size_t>(channel) * history + slot];
        for (uint32_t token = 0; token < tokens; ++token)
            sequence[static_cast<size_t>(history + token) * channels + channel] =
                input[static_cast<size_t>(token) * channels + channel];
    }
    for (uint32_t token = 0; token < tokens; ++token)
        for (uint32_t channel = 0; channel < channels; ++channel) {
            float sum = 0.0f;
            for (uint32_t tap = 0; tap < kernel_size; ++tap)
                sum += sequence[static_cast<size_t>(token + tap) * channels + channel] *
                       weights[static_cast<size_t>(channel) * kernel_size + tap];
            output[static_cast<size_t>(token) * channels + channel] = sum;
        }
    for (uint32_t channel = 0; channel < channels; ++channel)
        for (uint32_t slot = 0; slot < history; ++slot)
            state[static_cast<size_t>(channel) * history + slot] =
                sequence[static_cast<size_t>(tokens + slot) * channels + channel];
    return Status::success();
}

Status reference_gated_delta_net_f32(const float* q, const float* k, const float* v,
                                     const float* gate, const float* beta,
                                     float* state, float* output,
                                     uint32_t tokens, uint32_t state_size,
                                     uint32_t value_heads, uint32_t key_heads) {
    if (!q || !k || !v || !gate || !beta || !state || !output ||
        tokens == 0 || state_size == 0 || value_heads == 0 || key_heads == 0 ||
        value_heads % key_heads != 0)
        return {StatusCode::InvalidArgument, "GatedDeltaNet reference received invalid pointers or dimensions"};
    for (uint32_t token = 0; token < tokens; ++token)
        for (uint32_t head = 0; head < value_heads; ++head) {
            const uint32_t key_head = head % key_heads;
            const float decay = std::exp(gate[static_cast<size_t>(token) * value_heads + head]);
            const float rate = beta[static_cast<size_t>(token) * value_heads + head];
            float* head_state = state + static_cast<size_t>(head) * state_size * state_size;
            for (uint32_t column = 0; column < state_size; ++column) {
                float kv = 0.0f;
                for (uint32_t row = 0; row < state_size; ++row) {
                    const size_t state_index = static_cast<size_t>(row) * state_size + column;
                    head_state[state_index] *= decay;
                    kv += k[(static_cast<size_t>(token) * key_heads + key_head) * state_size + row] *
                          head_state[state_index];
                }
                const float delta =
                    (v[(static_cast<size_t>(token) * value_heads + head) * state_size + column] - kv) * rate;
                float result = 0.0f;
                for (uint32_t row = 0; row < state_size; ++row) {
                    const size_t state_index = static_cast<size_t>(row) * state_size + column;
                    head_state[state_index] +=
                        k[(static_cast<size_t>(token) * key_heads + key_head) * state_size + row] * delta;
                    result += q[(static_cast<size_t>(token) * key_heads + key_head) * state_size + row] *
                              head_state[state_index];
                }
                output[(static_cast<size_t>(token) * value_heads + head) * state_size + column] =
                    result / std::sqrt(static_cast<float>(state_size));
            }
        }
    return Status::success();
}

Status register_ssm_conv1d_operator_candidate(OperatorRegistry& registry, BackendKind backend) {
    KernelDefinition definition;
    definition.operator_name = "SSMConv1D";
    definition.backend = backend;
    definition.input_type = DataType::F32;
    definition.output_type = DataType::F32;
    definition.min_input_count = 3;
    definition.max_input_count = 3;
    definition.min_rank = 2;
    definition.max_rank = 2;
    definition.validation = KernelValidation::Unverified;
    if (backend == BackendKind::OpenGles) {
        definition.kernel_id = "gles-ssm-conv1d-f32-v1";
        definition.glsl_source = kSsmConvGlsl;
    } else if (backend == BackendKind::DawnWebGpu) {
        definition.kernel_id = "dawn-ssm-conv1d-f32-v1";
        definition.wgsl_source = kSsmConvWgsl;
    } else return {StatusCode::Unsupported, "SSMConv1D backend is unsupported"};
    return registry.register_kernel(std::move(definition));
}

Status register_gated_delta_net_operator_candidate(OperatorRegistry& registry, BackendKind backend) {
    KernelDefinition definition;
    definition.operator_name = "GatedDeltaNet";
    definition.backend = backend;
    definition.input_type = DataType::F32;
    definition.output_type = DataType::F32;
    definition.min_input_count = 6;
    definition.max_input_count = 6;
    definition.min_rank = 2;
    definition.max_rank = 3;
    definition.validation = KernelValidation::Unverified;
    if (backend == BackendKind::OpenGles) {
        definition.kernel_id = "gles-gated-delta-net-f32-v1";
        definition.glsl_source = kGatedDeltaNetGlsl;
    } else if (backend == BackendKind::DawnWebGpu) {
        definition.kernel_id = "dawn-gated-delta-net-f32-v1";
        definition.wgsl_source = kGatedDeltaNetWgsl;
    } else return {StatusCode::Unsupported, "GatedDeltaNet backend is unsupported"};
    return registry.register_kernel(std::move(definition));
}

} // namespace ai_chat::gpu_engine
