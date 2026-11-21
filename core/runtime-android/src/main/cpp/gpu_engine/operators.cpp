#include "operators.h"
#include "quantization.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <iomanip>
#include <sstream>
#include <utility>

namespace ai_chat::gpu_engine {
namespace {

constexpr char kAttentionF32Wgsl[] = R"wgsl(
@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> key: array<f32>;
@group(0) @binding(2) var<storage, read> value: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
const TOKENS: u32 = 1u;
const QUERY_HEADS: u32 = 1u;
const KV_HEADS: u32 = 1u;
const HEAD_DIM: u32 = 2u;
const PAST_LENGTH: u32 = 0u;
const SEQUENCE_LENGTH: u32 = 1u;
const SCALE: f32 = 0.70710678118;
@compute @workgroup_size(1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let token = gid.x;
    let query_head = gid.y;
    if (token >= TOKENS || query_head >= QUERY_HEADS) { return; }
    let kv_head = query_head / (QUERY_HEADS / KV_HEADS);
    let head_task = token * QUERY_HEADS + query_head;
    let query_base = head_task * HEAD_DIM;
    let kv_head_offset = kv_head * HEAD_DIM;
    var accumulator: array<f32, 256>;
    for (var d = 0u; d < HEAD_DIM; d = d + 1u) { accumulator[d] = 0.0; }
    var maximum = -3.402823466e+38;
    var denominator = 0.0;
    let allowed = PAST_LENGTH + token + 1u;
    for (var sequence = 0u; sequence < allowed; sequence = sequence + 1u) {
        var score = 0.0;
        for (var d = 0u; d < HEAD_DIM; d = d + 1u) {
            score = score + query[query_base + d] *
                    key[sequence * KV_HEADS * HEAD_DIM + kv_head_offset + d];
        }
        score = score * SCALE;
        let next_maximum = max(maximum, score);
        let old_weight = exp(maximum - next_maximum);
        let new_weight = exp(score - next_maximum);
        denominator = denominator * old_weight + new_weight;
        for (var d = 0u; d < HEAD_DIM; d = d + 1u) {
            accumulator[d] = accumulator[d] * old_weight +
                value[sequence * KV_HEADS * HEAD_DIM + kv_head_offset + d] * new_weight;
        }
        maximum = next_maximum;
    }
    let output_base = head_task * HEAD_DIM;
    for (var d = 0u; d < HEAD_DIM; d = d + 1u) {
        output[output_base + d] = accumulator[d] / denominator;
    }
}
)wgsl";

constexpr char kEmbeddingF32Wgsl[] = R"wgsl(
@group(0) @binding(0) var<storage, read> embedding_weights: array<f32>;
@group(0) @binding(1) var<storage, read> token_ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
const VOCABULARY_SIZE: u32 = 1u;
const EMBEDDING_DIMENSION: u32 = 1u;
const TOKEN_COUNT: u32 = 1u;
const TOTAL_ELEMENTS: u32 = 1u;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= TOTAL_ELEMENTS) { return; }
    let token_row = gid.x / EMBEDDING_DIMENSION;
    let column = gid.x % EMBEDDING_DIMENSION;
    if (token_row >= TOKEN_COUNT) { return; }
    let token = token_ids[token_row];
    if (token >= VOCABULARY_SIZE) {
        output[gid.x] = bitcast<f32>(0x7fc00000u);
        return;
    }
    output[gid.x] = embedding_weights[token * EMBEDDING_DIMENSION + column];
}
)wgsl";


constexpr char kEmbeddingF32Glsl[] = R"glsl(#version 310 es
precision highp float;
precision highp int;
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer EmbeddingWeights { float embedding_weights[]; };
layout(std430, binding = 1) readonly buffer TokenIds { uint token_ids[]; };
layout(std430, binding = 2) writeonly buffer Output { float output_values[]; };
const uint VOCABULARY_SIZE = 1u;
const uint EMBEDDING_DIMENSION = 1u;
const uint TOKEN_COUNT = 1u;
const uint TOTAL_ELEMENTS = 1u;
void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= TOTAL_ELEMENTS) return;
    uint token_row = index / EMBEDDING_DIMENSION;
    uint column = index % EMBEDDING_DIMENSION;
    if (token_row >= TOKEN_COUNT) return;
    uint token = token_ids[token_row];
    if (token >= VOCABULARY_SIZE) {
        output_values[index] = uintBitsToFloat(0x7fc00000u);
        return;
    }
    output_values[index] = embedding_weights[token * EMBEDDING_DIMENSION + column];
}
)glsl";

constexpr char kAttentionF32Glsl[] = R"glsl(#version 310 es
precision highp float;
precision highp int;
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(std430, binding = 0) readonly buffer Query { float query_values[]; };
layout(std430, binding = 1) readonly buffer Key { float key_values[]; };
layout(std430, binding = 2) readonly buffer Value { float value_values[]; };
layout(std430, binding = 3) writeonly buffer Output { float output_values[]; };
const uint TOKENS = 1u;
const uint QUERY_HEADS = 1u;
const uint KV_HEADS = 1u;
const uint HEAD_DIM = 2u;
const uint PAST_LENGTH = 0u;
const uint SEQUENCE_LENGTH = 1u;
const float SCALE = 0.70710678118;
void main() {
    uint token = gl_WorkGroupID.x;
    uint query_head = gl_WorkGroupID.y;
    if (token >= TOKENS || query_head >= QUERY_HEADS) return;
    uint kv_head = query_head / (QUERY_HEADS / KV_HEADS);
    uint head_task = token * QUERY_HEADS + query_head;
    uint query_base = head_task * HEAD_DIM;
    uint kv_head_offset = kv_head * HEAD_DIM;
    float accumulator[256];
    for (uint d = 0u; d < HEAD_DIM; ++d) accumulator[d] = 0.0;
    float maximum = -3.402823466e+38;
    float denominator = 0.0;
    uint allowed = PAST_LENGTH + token + 1u;
    for (uint sequence = 0u; sequence < allowed; ++sequence) {
        float score = 0.0;
        for (uint d = 0u; d < HEAD_DIM; ++d)
            score += query_values[query_base + d] * key_values[sequence * KV_HEADS * HEAD_DIM + kv_head_offset + d];
        score *= SCALE;
        float next_maximum = max(maximum, score);
        float old_weight = exp(maximum - next_maximum);
        float new_weight = exp(score - next_maximum);
        denominator = denominator * old_weight + new_weight;
        for (uint d = 0u; d < HEAD_DIM; ++d)
            accumulator[d] = accumulator[d] * old_weight +
                value_values[sequence * KV_HEADS * HEAD_DIM + kv_head_offset + d] * new_weight;
        maximum = next_maximum;
    }
    uint output_base = head_task * HEAD_DIM;
    for (uint d = 0u; d < HEAD_DIM; ++d)
        output_values[output_base + d] = accumulator[d] / denominator;
}
)glsl";

constexpr char kRopeF32Glsl[] = R"glsl(#version 310 es
precision highp float;
precision highp int;
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer Input { float input_values[]; };
layout(std430, binding = 1) writeonly buffer Output { float output_values[]; };
const uint TOTAL_ELEMENTS = 1u;
const uint HEADS = 1u;
const uint HEAD_DIM = 2u;
const uint ROTARY_DIM = 2u;
const uint START_POSITION = 0u;
const float THETA = 10000.0;
const uint NEOX_STYLE = 0u;
void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= TOTAL_ELEMENTS) return;
    uint dimension = index % HEAD_DIM;
    if (dimension >= ROTARY_DIM) {
        output_values[index] = input_values[index];
        return;
    }
    uint token = index / (HEADS * HEAD_DIM);
    uint half_dimension = ROTARY_DIM / 2u;
    uint pair = dimension / 2u;
    uint partner_dimension = dimension ^ 1u;
    if (NEOX_STYLE == 1u) {
        pair = dimension % half_dimension;
        partner_dimension = dimension < half_dimension ? dimension + half_dimension : dimension - half_dimension;
    }
    float exponent = 2.0 * float(pair) / float(ROTARY_DIM);
    float inverse_frequency = pow(THETA, -exponent);
    float angle = float(START_POSITION + token) * inverse_frequency;
    float cosine = cos(angle);
    float sine = sin(angle);
    uint base = index - dimension;
    float value = input_values[index];
    float partner = input_values[base + partner_dimension];
    if (NEOX_STYLE == 0u) {
        output_values[index] = (dimension & 1u) == 1u
            ? value * cosine + partner * sine
            : value * cosine - partner * sine;
    } else {
        output_values[index] = dimension >= half_dimension
            ? value * cosine + partner * sine
            : value * cosine - partner * sine;
    }
}
)glsl";

constexpr char kAddF32Wgsl[] = R"wgsl(
@group(0) @binding(0) var<storage, read> lhs: array<f32>;
@group(0) @binding(1) var<storage, read> rhs: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < arrayLength(&output)) {
        output[gid.x] = lhs[gid.x] + rhs[gid.x];
    }
}
)wgsl";

constexpr char kAddF32Glsl[] = R"glsl(#version 310 es
precision highp float;
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer Lhs { float lhs[]; };
layout(std430, binding = 1) readonly buffer Rhs { float rhs[]; };
layout(std430, binding = 2) writeonly buffer Output { float output_values[]; };
void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index < uint(output_values.length())) output_values[index] = lhs[index] + rhs[index];
}
)glsl";

constexpr char kSiluF32Glsl[] = R"glsl(#version 310 es
precision highp float;
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer Input { float input_values[]; };
layout(std430, binding = 1) writeonly buffer Output { float output_values[]; };
void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index < uint(output_values.length())) {
        float x = input_values[index];
        output_values[index] = x / (1.0 + exp(-x));
    }
}
)glsl";

constexpr char kSoftmaxF32Glsl[] = R"glsl(#version 310 es
precision highp float;
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer Input { float input_values[]; };
layout(std430, binding = 1) writeonly buffer Output { float output_values[]; };
const uint ROWS = 1u;
const uint COLUMNS = 1u;
shared float partial[64];
void main() {
    uint row = gl_WorkGroupID.x;
    uint lane = gl_LocalInvocationID.x;
    if (row >= ROWS) return;
    float local_max = -3.402823466e+38;
    for (uint column = lane; column < COLUMNS; column += 64u)
        local_max = max(local_max, input_values[row * COLUMNS + column]);
    partial[lane] = local_max;
    barrier();
    for (uint stride = 32u; stride > 0u; stride >>= 1u) {
        if (lane < stride) partial[lane] = max(partial[lane], partial[lane + stride]);
        barrier();
    }
    float maximum = partial[0];
    float local_sum = 0.0;
    for (uint column = lane; column < COLUMNS; column += 64u)
        local_sum += exp(input_values[row * COLUMNS + column] - maximum);
    partial[lane] = local_sum;
    barrier();
    for (uint stride = 32u; stride > 0u; stride >>= 1u) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        barrier();
    }
    float denominator = partial[0];
    for (uint column = lane; column < COLUMNS; column += 64u) {
        uint index = row * COLUMNS + column;
        output_values[index] = exp(input_values[index] - maximum) / denominator;
    }
}
)glsl";

constexpr char kRmsNormF32Glsl[] = R"glsl(#version 310 es
precision highp float;
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer Input { float input_values[]; };
layout(std430, binding = 1) readonly buffer Weight { float weight_values[]; };
layout(std430, binding = 2) writeonly buffer Output { float output_values[]; };
const uint ROWS = 1u;
const uint COLUMNS = 1u;
const float EPSILON = 0.00001;
shared float partial[64];
void main() {
    uint row = gl_WorkGroupID.x;
    uint lane = gl_LocalInvocationID.x;
    if (row >= ROWS) return;
    float sum = 0.0;
    for (uint column = lane; column < COLUMNS; column += 64u) {
        float x = input_values[row * COLUMNS + column];
        sum += x * x;
    }
    partial[lane] = sum;
    barrier();
    for (uint stride = 32u; stride > 0u; stride >>= 1u) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        barrier();
    }
    float inverse = inversesqrt(partial[0] / float(COLUMNS) + EPSILON);
    for (uint column = lane; column < COLUMNS; column += 64u) {
        uint index = row * COLUMNS + column;
        output_values[index] = input_values[index] * inverse * weight_values[column];
    }
}
)glsl";

constexpr char kMatMulF32Glsl[] = R"glsl(#version 310 es
precision highp float;
layout(local_size_x = 8, local_size_y = 8) in;
layout(std430, binding = 0) readonly buffer Lhs { float lhs[]; };
layout(std430, binding = 1) readonly buffer Rhs { float rhs[]; };
layout(std430, binding = 2) writeonly buffer Output { float output_values[]; };
const uint ROWS = 1u;
const uint COLUMNS = 1u;
const uint INNER = 1u;
void main() {
    uint row = gl_GlobalInvocationID.y;
    uint column = gl_GlobalInvocationID.x;
    if (row >= ROWS || column >= COLUMNS) return;
    float sum = 0.0;
    for (uint k = 0u; k < INNER; ++k) sum += lhs[row * INNER + k] * rhs[k * COLUMNS + column];
    output_values[row * COLUMNS + column] = sum;
}
)glsl";

constexpr char kMultiplyF32Glsl[] = R"glsl(#version 310 es
precision highp float;
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer Lhs { float lhs[]; };
layout(std430, binding = 1) readonly buffer Rhs { float rhs[]; };
layout(std430, binding = 2) writeonly buffer Output { float output_values[]; };
void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index < uint(output_values.length())) output_values[index] = lhs[index] * rhs[index];
}
)glsl";

constexpr char kMultiplyF32Wgsl[] = R"wgsl(
@group(0) @binding(0) var<storage, read> lhs: array<f32>;
@group(0) @binding(1) var<storage, read> rhs: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < arrayLength(&output)) {
        output[gid.x] = lhs[gid.x] * rhs[gid.x];
    }
}
)wgsl";

constexpr char kSiluF32Wgsl[] = R"wgsl(
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
const ELEMENT_COUNT: u32 = 1u;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < ELEMENT_COUNT) {
        let value = input[gid.x];
        output[gid.x] = value / (1.0 + exp(-value));
    }
}
)wgsl";

constexpr char kSoftmaxF32Wgsl[] = R"wgsl(
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
const COLUMNS: u32 = 1u;
var<workgroup> partial: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local: vec3<u32>) {
    let row = group.x;
    var local_max = -3.402823466e+38;
    for (var column = local.x; column < COLUMNS; column = column + 64u) {
        local_max = max(local_max, input[row * COLUMNS + column]);
    }
    partial[local.x] = local_max;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if (stride == 0u) { break; }
        if (local.x < stride) { partial[local.x] = max(partial[local.x], partial[local.x + stride]); }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let maximum = partial[0];
    var local_sum = 0.0;
    for (var column = local.x; column < COLUMNS; column = column + 64u) {
        local_sum = local_sum + exp(input[row * COLUMNS + column] - maximum);
    }
    partial[local.x] = local_sum;
    workgroupBarrier();
    stride = 32u;
    loop {
        if (stride == 0u) { break; }
        if (local.x < stride) { partial[local.x] = partial[local.x] + partial[local.x + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let denominator = partial[0];
    for (var column = local.x; column < COLUMNS; column = column + 64u) {
        output[row * COLUMNS + column] = exp(input[row * COLUMNS + column] - maximum) / denominator;
    }
}
)wgsl";

constexpr char kRopeF32Wgsl[] = R"wgsl(
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
const TOTAL_ELEMENTS: u32 = 1u;
const HEADS: u32 = 1u;
const HEAD_DIM: u32 = 2u;
const ROTARY_DIM: u32 = 2u;
const START_POSITION: u32 = 0u;
const THETA: f32 = 10000.0;
const NEOX_STYLE: u32 = 0u;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= TOTAL_ELEMENTS) { return; }
    let dimension = gid.x % HEAD_DIM;
    if (dimension >= ROTARY_DIM) {
        output[gid.x] = input[gid.x];
        return;
    }
    let token = gid.x / (HEADS * HEAD_DIM);
    let half_dimension = ROTARY_DIM / 2u;
    var pair = dimension / 2u;
    var partner_dimension = dimension ^ 1u;
    if (NEOX_STYLE == 1u) {
        pair = dimension % half_dimension;
        partner_dimension = select(dimension + half_dimension, dimension - half_dimension,
                                   dimension >= half_dimension);
    }
    let exponent = 2.0 * f32(pair) / f32(ROTARY_DIM);
    let inverse_frequency = pow(THETA, -exponent);
    let angle = f32(START_POSITION + token) * inverse_frequency;
    let cosine = cos(angle);
    let sine = sin(angle);
    let base = gid.x - dimension;
    let value = input[gid.x];
    let partner = input[base + partner_dimension];
    if (NEOX_STYLE == 0u) {
        output[gid.x] = select(value * cosine - partner * sine,
                               value * cosine + partner * sine,
                               (dimension & 1u) == 1u);
    } else {
        output[gid.x] = select(value * cosine - partner * sine,
                               value * cosine + partner * sine,
                               dimension >= half_dimension);
    }
}
)wgsl";

constexpr char kMatMulF32Wgsl[] = R"wgsl(
@group(0) @binding(0) var<storage, read> lhs: array<f32>;
@group(0) @binding(1) var<storage, read> rhs: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
const ROWS: u32 = 1u;
const COLUMNS: u32 = 1u;
const INNER: u32 = 1u;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let column = gid.x;
    if (row >= ROWS || column >= COLUMNS) { return; }
    var sum = 0.0;
    for (var k = 0u; k < INNER; k = k + 1u) {
        sum = sum + lhs[row * INNER + k] * rhs[k * COLUMNS + column];
    }
    output[row * COLUMNS + column] = sum;
}
)wgsl";

constexpr char kRmsNormF32Wgsl[] = R"wgsl(
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
const COLUMNS: u32 = 1u;
const EPSILON: f32 = 0.00001;
var<workgroup> partial: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local: vec3<u32>) {
    let row = group.x;
    var sum = 0.0;
    for (var column = local.x; column < COLUMNS; column = column + 64u) {
        let value = input[row * COLUMNS + column];
        sum = sum + value * value;
    }
    partial[local.x] = sum;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if (stride == 0u) { break; }
        if (local.x < stride) { partial[local.x] = partial[local.x] + partial[local.x + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let inverse = inverseSqrt(partial[0] / f32(COLUMNS) + EPSILON);
    for (var column = local.x; column < COLUMNS; column = column + 64u) {
        output[row * COLUMNS + column] =
            input[row * COLUMNS + column] * inverse * weight[column];
    }
}
)wgsl";

std::string replace_once(std::string source, const std::string& from, const std::string& to) {
    const size_t position = source.find(from);
    if (position != std::string::npos) source.replace(position, from.size(), to);
    return source;
}

bool is_contiguous_f32(const Tensor& tensor) {
    const TensorDesc& desc = tensor.descriptor();
    return desc.data_type == DataType::F32 &&
           desc.layout == TensorLayout::Contiguous &&
           desc.strides_bytes.empty() &&
           desc.memory != MemoryDomain::Host;
}

Status build_binary_f32_candidate(const Tensor& lhs, const Tensor& rhs,
                                    const Tensor& output, ComputeDispatch& dispatch,
                                    const char* label, const char* kernel_id, const char* shader) {
    dispatch = ComputeDispatch{};
    if (!is_contiguous_f32(lhs) || !is_contiguous_f32(rhs) || !is_contiguous_f32(output))
        return {StatusCode::Unsupported, "Binary F32 operator requires contiguous device/shared tensors"};
    if (lhs.descriptor().shape != rhs.descriptor().shape ||
        lhs.descriptor().shape != output.descriptor().shape ||
        lhs.element_count() == 0 || lhs.element_count() > UINT32_MAX ||
        lhs.element_count() != rhs.element_count() || lhs.element_count() != output.element_count())
        return {StatusCode::InvalidArgument, "Binary F32 operator tensor shapes or element counts do not match"};
    if (!lhs.buffer() || !rhs.buffer() || !output.buffer() ||
        output.buffer().get() == lhs.buffer().get() || output.buffer().get() == rhs.buffer().get())
        return {StatusCode::Unsupported, "Binary F32 output must use a distinct valid buffer"};
    if (lhs.byte_size() != rhs.byte_size() || lhs.byte_size() != output.byte_size() ||
        lhs.byte_size() != lhs.element_count() * sizeof(float))
        return {StatusCode::InvalidArgument, "Binary F32 tensor byte ranges are inconsistent"};
    dispatch.label = label;
    dispatch.kernel_id = kernel_id;
    dispatch.backend = BackendKind::DawnWebGpu;
    dispatch.wgsl_source = shader;
    dispatch.workgroup_size_x = 64;
    dispatch.workgroups_x = static_cast<uint32_t>((lhs.element_count() + 63) / 64);
    dispatch.bindings = {
        {0, lhs.buffer(), lhs.byte_offset(), lhs.byte_size()},
        {1, rhs.buffer(), rhs.byte_offset(), rhs.byte_size()},
        {2, output.buffer(), output.byte_offset(), output.byte_size()},
    };
    return Status::success();
}

} // namespace



const char* attention_f32_wgsl_source() noexcept { return kAttentionF32Wgsl; }

Status build_attention_f32_candidate(const Tensor& query, const Tensor& key,
                                     const Tensor& value, const Tensor& output,
                                     uint32_t past_length, ComputeDispatch& dispatch) {
    dispatch = ComputeDispatch{};
    if (!is_contiguous_f32(query) || !is_contiguous_f32(key) ||
        !is_contiguous_f32(value) || !is_contiguous_f32(output))
        return {StatusCode::Unsupported, "Attention candidate requires contiguous device/shared F32 tensors"};
    const auto& q = query.descriptor().shape;
    const auto& k = key.descriptor().shape;
    const auto& v = value.descriptor().shape;
    if (q.size() != 3 || k.size() != 3 || v != k || output.descriptor().shape != q ||
        q[0] == 0 || q[1] == 0 || q[2] == 0 || k[0] == 0 || k[1] == 0 ||
        q[2] != k[2] || q[1] % k[1] != 0 || q[2] > 256 ||
        q[0] > UINT32_MAX || q[1] > UINT32_MAX || k[1] > UINT32_MAX ||
        k[0] > UINT32_MAX || q[2] > UINT32_MAX ||
        q[0] > UINT32_MAX - past_length || k[0] != static_cast<uint64_t>(past_length) + q[0] ||
        query.element_count() > UINT32_MAX || key.element_count() > UINT32_MAX ||
        value.element_count() > UINT32_MAX || output.element_count() > UINT32_MAX)
        return {StatusCode::InvalidArgument, "Attention expects Q[T,Hq,D], K/V[S,Hkv,D] with causal S=past+T and valid GQA dimensions"};
    if (!query.buffer() || !key.buffer() || !value.buffer() || !output.buffer() ||
        output.buffer().get() == query.buffer().get() || output.buffer().get() == key.buffer().get() ||
        output.buffer().get() == value.buffer().get())
        return {StatusCode::InvalidArgument, "Attention output must use a distinct valid buffer"};
    dispatch.label = "attention-f32-causal-gqa-online-softmax-unverified";
    dispatch.kernel_id = "dawn-attention-f32-causal-gqa-v1";
    dispatch.backend = BackendKind::DawnWebGpu;
    dispatch.wgsl_source = kAttentionF32Wgsl;
    dispatch.workgroup_size_x = 1;
    dispatch.workgroup_size_y = 1;
    const double scale = 1.0 / std::sqrt(static_cast<double>(q[2]));
    std::ostringstream scale_literal;
    scale_literal << std::setprecision(12) << scale;
    std::string scale_text = scale_literal.str();
    if (scale_text.find_first_of(".eE") == std::string::npos) scale_text += ".0";
    const std::pair<std::string, std::string> constants[] = {
        {"const TOKENS: u32 = 1u;", "const TOKENS: u32 = " + std::to_string(q[0]) + "u;"},
        {"const QUERY_HEADS: u32 = 1u;", "const QUERY_HEADS: u32 = " + std::to_string(q[1]) + "u;"},
        {"const KV_HEADS: u32 = 1u;", "const KV_HEADS: u32 = " + std::to_string(k[1]) + "u;"},
        {"const HEAD_DIM: u32 = 2u;", "const HEAD_DIM: u32 = " + std::to_string(q[2]) + "u;"},
        {"const PAST_LENGTH: u32 = 0u;", "const PAST_LENGTH: u32 = " + std::to_string(past_length) + "u;"},
        {"const SEQUENCE_LENGTH: u32 = 1u;", "const SEQUENCE_LENGTH: u32 = " + std::to_string(k[0]) + "u;"},
        {"const SCALE: f32 = 0.70710678118;", "const SCALE: f32 = " + scale_text + ";"},
    };
    for (const auto& constant : constants) {
        const size_t position = dispatch.wgsl_source.find(constant.first);
        if (position == std::string::npos)
            return {StatusCode::BackendFailure, "Attention WGSL constant placeholder is missing"};
        dispatch.wgsl_source.replace(position, constant.first.size(), constant.second);
    }
    dispatch.workgroups_x = static_cast<uint32_t>(q[0]);
    dispatch.workgroups_y = static_cast<uint32_t>(q[1]);
    dispatch.bindings = {
        {0, query.buffer(), query.byte_offset(), query.byte_size()},
        {1, key.buffer(), key.byte_offset(), key.byte_size()},
        {2, value.buffer(), value.byte_offset(), value.byte_size()},
        {3, output.buffer(), output.byte_offset(), output.byte_size()},
    };
    return Status::success();
}

Status build_attention_f32_gles_candidate(const Tensor& query, const Tensor& key,
                                          const Tensor& value, const Tensor& output,
                                          uint32_t past_length, ComputeDispatch& dispatch) {
    Status status = build_attention_f32_candidate(query, key, value, output, past_length, dispatch);
    if (!status.ok()) return status;
    const auto& q = query.descriptor().shape;
    const auto& k = key.descriptor().shape;
    const double scale = 1.0 / std::sqrt(static_cast<double>(q[2]));
    std::ostringstream scale_literal;
    scale_literal << std::setprecision(12) << scale;
    std::string scale_text = scale_literal.str();
    if (scale_text.find_first_of(".eE") == std::string::npos) scale_text += ".0";
    dispatch.backend = BackendKind::OpenGles;
    dispatch.kernel_id = "gles-attention-f32-causal-gqa-v1";
    dispatch.label = "gles-attention-f32-causal-gqa-unverified";
    dispatch.wgsl_source.clear();
    dispatch.glsl_source = kAttentionF32Glsl;
    const std::pair<std::string, std::string> constants[] = {
        {"const uint TOKENS = 1u;", "const uint TOKENS = " + std::to_string(q[0]) + "u;"},
        {"const uint QUERY_HEADS = 1u;", "const uint QUERY_HEADS = " + std::to_string(q[1]) + "u;"},
        {"const uint KV_HEADS = 1u;", "const uint KV_HEADS = " + std::to_string(k[1]) + "u;"},
        {"const uint HEAD_DIM = 2u;", "const uint HEAD_DIM = " + std::to_string(q[2]) + "u;"},
        {"const uint PAST_LENGTH = 0u;", "const uint PAST_LENGTH = " + std::to_string(past_length) + "u;"},
        {"const uint SEQUENCE_LENGTH = 1u;", "const uint SEQUENCE_LENGTH = " + std::to_string(k[0]) + "u;"},
        {"const float SCALE = 0.70710678118;", "const float SCALE = " + scale_text + ";"}
    };
    for (const auto& constant : constants) {
        const size_t pos = dispatch.glsl_source.find(constant.first);
        if (pos == std::string::npos) return {StatusCode::BackendFailure, "Attention GLSL constant placeholder is missing"};
        dispatch.glsl_source.replace(pos, constant.first.size(), constant.second);
    }
    return Status::success();
}

Status reference_attention_f32(const float* query, const float* key, const float* value,
                               float* output, uint32_t tokens, uint32_t query_heads,
                               uint32_t kv_heads, uint32_t head_dimension,
                               uint32_t past_length) {
    if ((!query || !key || !value || !output) && tokens && query_heads && kv_heads && head_dimension)
        return {StatusCode::InvalidArgument, "Attention reference received a null pointer"};
    if (tokens == 0 || query_heads == 0 || kv_heads == 0 || head_dimension == 0 ||
        head_dimension > 256 || query_heads % kv_heads != 0 ||
        tokens > UINT32_MAX - past_length)
        return {StatusCode::InvalidArgument, "Attention reference dimensions are invalid"};
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dimension));
    std::vector<float> accumulator(head_dimension);
    for (uint32_t token = 0; token < tokens; ++token) {
        for (uint32_t head = 0; head < query_heads; ++head) {
            const uint32_t kv_head = head / (query_heads / kv_heads);
            const uint64_t qbase = (static_cast<uint64_t>(token) * query_heads + head) * head_dimension;
            const uint64_t kvhead_offset = static_cast<uint64_t>(kv_head) * head_dimension;
            std::fill(accumulator.begin(), accumulator.end(), 0.0f);
            float maximum = -std::numeric_limits<float>::infinity();
            float denominator = 0.0f;
            for (uint32_t position = 0; position <= past_length + token; ++position) {
                float score = 0.0f;
                for (uint32_t d = 0; d < head_dimension; ++d)
                    score += query[qbase + d] * key[static_cast<uint64_t>(position) * kv_heads * head_dimension + kvhead_offset + d];
                score *= scale;
                const float next_maximum = std::max(maximum, score);
                const float old_weight = std::exp(maximum - next_maximum);
                const float new_weight = std::exp(score - next_maximum);
                denominator = denominator * old_weight + new_weight;
                for (uint32_t d = 0; d < head_dimension; ++d)
                    accumulator[d] = accumulator[d] * old_weight +
                        value[static_cast<uint64_t>(position) * kv_heads * head_dimension + kvhead_offset + d] * new_weight;
                maximum = next_maximum;
            }
            const uint64_t out_base = (static_cast<uint64_t>(token) * query_heads + head) * head_dimension;
            for (uint32_t d = 0; d < head_dimension; ++d)
                output[out_base + d] = accumulator[d] / denominator;
        }
    }
    return Status::success();
}

const char* embedding_f32_wgsl_source() noexcept { return kEmbeddingF32Wgsl; }

Status build_embedding_f32_candidate(const Tensor& embedding_weights,
                                     const std::shared_ptr<Buffer>& token_ids_buffer,
                                     const std::vector<uint32_t>& validated_token_ids,
                                     const Tensor& output,
                                     ComputeDispatch& dispatch) {
    dispatch = ComputeDispatch{};
    if (!is_contiguous_f32(embedding_weights) || !is_contiguous_f32(output) ||
        !token_ids_buffer || token_ids_buffer->memory_domain() == MemoryDomain::Host)
        return {StatusCode::Unsupported, "Embedding requires F32 weights/output and a device/shared token-id buffer"};
    const auto& weights_shape = embedding_weights.descriptor().shape;
    const auto& output_shape = output.descriptor().shape;
    if (weights_shape.size() != 2 || output_shape.size() != 2 ||
        weights_shape[0] == 0 || weights_shape[1] == 0 ||
        output_shape[0] == 0 || output_shape[1] != weights_shape[1] ||
        validated_token_ids.size() != output_shape[0] ||
        weights_shape[0] > UINT32_MAX || weights_shape[1] > UINT32_MAX ||
        embedding_weights.element_count() > UINT32_MAX ||
        output_shape[0] > UINT32_MAX || output.element_count() > UINT32_MAX)
        return {StatusCode::InvalidArgument, "Embedding expects weights [vocabulary,dimension] and output [tokens,dimension]"};
    for (uint32_t token : validated_token_ids) {
        if (token >= weights_shape[0])
            return {StatusCode::InvalidArgument, "Embedding token id is outside the vocabulary"};
    }
    const uint64_t token_bytes = output_shape[0] * sizeof(uint32_t);
    if (token_ids_buffer->size_bytes() < token_bytes ||
        !embedding_weights.buffer() || !output.buffer() ||
        output.buffer().get() == embedding_weights.buffer().get() ||
        output.buffer().get() == token_ids_buffer.get())
        return {StatusCode::InvalidArgument, "Embedding input/output buffer ranges are inconsistent"};
    dispatch.label = "embedding-f32-generic-unverified";
    dispatch.kernel_id = "dawn-embedding-f32-generic-v1";
    dispatch.backend = BackendKind::DawnWebGpu;
    dispatch.wgsl_source = kEmbeddingF32Wgsl;
    dispatch.workgroup_size_x = 64;
    const std::pair<std::string, std::string> constants[] = {
        {"const VOCABULARY_SIZE: u32 = 1u;", "const VOCABULARY_SIZE: u32 = " + std::to_string(weights_shape[0]) + "u;"},
        {"const EMBEDDING_DIMENSION: u32 = 1u;", "const EMBEDDING_DIMENSION: u32 = " + std::to_string(weights_shape[1]) + "u;"},
        {"const TOKEN_COUNT: u32 = 1u;", "const TOKEN_COUNT: u32 = " + std::to_string(output_shape[0]) + "u;"},
        {"const TOTAL_ELEMENTS: u32 = 1u;", "const TOTAL_ELEMENTS: u32 = " + std::to_string(output.element_count()) + "u;"},
    };
    for (const auto& constant : constants) {
        const size_t position = dispatch.wgsl_source.find(constant.first);
        if (position == std::string::npos)
            return {StatusCode::BackendFailure, "Embedding WGSL constant placeholder is missing"};
        dispatch.wgsl_source.replace(position, constant.first.size(), constant.second);
    }
    dispatch.workgroups_x = static_cast<uint32_t>((output.element_count() + 63) / 64);
    dispatch.bindings = {
        {0, embedding_weights.buffer(), embedding_weights.byte_offset(), embedding_weights.byte_size()},
        {1, token_ids_buffer, 0, token_bytes},
        {2, output.buffer(), output.byte_offset(), output.byte_size()},
    };
    return Status::success();
}

Status reference_embedding_f32(const float* embedding_weights, uint32_t vocabulary_size,
                               uint32_t embedding_dimension,
                               const std::vector<uint32_t>& token_ids, float* output) {
    if ((!embedding_weights || !output) && !token_ids.empty())
        return {StatusCode::InvalidArgument, "Embedding reference received a null pointer"};
    if (vocabulary_size == 0 || embedding_dimension == 0 || token_ids.empty())
        return {StatusCode::InvalidArgument, "Embedding dimensions and token list must be non-empty"};
    for (size_t row = 0; row < token_ids.size(); ++row) {
        const uint32_t token = token_ids[row];
        if (token >= vocabulary_size)
            return {StatusCode::InvalidArgument, "Embedding token id is outside the vocabulary"};
        for (uint32_t column = 0; column < embedding_dimension; ++column) {
            output[row * embedding_dimension + column] =
                embedding_weights[static_cast<uint64_t>(token) * embedding_dimension + column];
        }
    }
    return Status::success();
}

Status build_embedding_f32_gles_candidate(const Tensor& embedding_weights,
                                         const std::shared_ptr<Buffer>& token_ids_buffer,
                                         const std::vector<uint32_t>& validated_token_ids,
                                         const Tensor& output, ComputeDispatch& dispatch) {
    Status status = build_embedding_f32_candidate(embedding_weights, token_ids_buffer,
                                                   validated_token_ids, output, dispatch);
    if (!status.ok()) return status;
    dispatch.label = "gles-embedding-f32-generic-unverified";
    dispatch.kernel_id = "gles-embedding-f32-generic-v1";
    dispatch.backend = BackendKind::OpenGles;
    dispatch.wgsl_source.clear();
    dispatch.glsl_source = kEmbeddingF32Glsl;
    const auto& weights = embedding_weights.descriptor().shape;
    const auto& output_shape = output.descriptor().shape;
    const std::pair<std::string, std::string> constants[] = {
        {"const uint VOCABULARY_SIZE = 1u;", "const uint VOCABULARY_SIZE = " + std::to_string(weights[0]) + "u;"},
        {"const uint EMBEDDING_DIMENSION = 1u;", "const uint EMBEDDING_DIMENSION = " + std::to_string(weights[1]) + "u;"},
        {"const uint TOKEN_COUNT = 1u;", "const uint TOKEN_COUNT = " + std::to_string(output_shape[0]) + "u;"},
        {"const uint TOTAL_ELEMENTS = 1u;", "const uint TOTAL_ELEMENTS = " + std::to_string(output.element_count()) + "u;"},
    };
    for (const auto& constant : constants) {
        const size_t position = dispatch.glsl_source.find(constant.first);
        if (position == std::string::npos)
            return {StatusCode::BackendFailure, "GLES Embedding GLSL placeholder is missing"};
        dispatch.glsl_source.replace(position, constant.first.size(), constant.second);
    }
    dispatch.workgroup_size_x = 64;
    return Status::success();
}
const char* add_f32_wgsl_source() noexcept { return kAddF32Wgsl; }


Status build_add_f32_candidate(const Tensor& lhs, const Tensor& rhs,
                               const Tensor& output, ComputeDispatch& dispatch) {
    return build_binary_f32_candidate(lhs, rhs, output, dispatch,
                                      "add-f32-generic-unverified", "dawn-add-f32-generic-v1", kAddF32Wgsl);
}

const char* multiply_f32_wgsl_source() noexcept { return kMultiplyF32Wgsl; }

Status build_multiply_f32_candidate(const Tensor& lhs, const Tensor& rhs,
                                    const Tensor& output, ComputeDispatch& dispatch) {
    return build_binary_f32_candidate(lhs, rhs, output, dispatch,
                                      "multiply-f32-generic-unverified", "dawn-multiply-f32-generic-v1", kMultiplyF32Wgsl);
}

Status build_add_f32_gles_candidate(const Tensor& lhs, const Tensor& rhs,
                                    const Tensor& output, ComputeDispatch& dispatch) {
    Status status = build_binary_f32_candidate(lhs, rhs, output, dispatch,
        "gles-add-f32-generic-unverified", "gles-add-f32-generic-v1", kAddF32Wgsl);
    if (!status.ok()) return status;
    dispatch.backend = BackendKind::OpenGles;
    dispatch.kernel_id = "gles-add-f32-generic-v1";
    dispatch.wgsl_source.clear();
    dispatch.glsl_source = kAddF32Glsl;
    return Status::success();
}


Status validate_gles_add_f32_kernel(Backend& backend, OperatorRegistry& registry) {
    if (backend.kind() != BackendKind::OpenGles || !backend.healthy())
        return {StatusCode::NotReady, "GLES Add validation requires an initialized healthy OpenGLES backend"};
    auto kernel = std::find_if(registry.kernels_.begin(), registry.kernels_.end(),
        [](const KernelDefinition& definition) {
            return definition.kernel_id == "gles-add-f32-generic-v1" &&
                   definition.backend == BackendKind::OpenGles;
        });
    if (kernel == registry.kernels_.end())
        return {StatusCode::NotReady, "GLES Add candidate is not registered"};
    if (kernel->validation == KernelValidation::Validated) return Status::success();
    if (kernel->validation != KernelValidation::Unverified || kernel->glsl_source.empty())
        return {StatusCode::Unsupported, "GLES Add candidate is rejected or has no GLSL ES source"};

    constexpr float lhs_values[] = {1.0f, -2.0f, 3.5f, 4.0f};
    constexpr float rhs_values[] = {0.5f, 2.0f, -1.0f, 8.0f};
    float expected[4] = {};
    float actual[4] = {};
    Status status = reference_add_f32(lhs_values, rhs_values, expected, 4);
    if (!status.ok()) return status;

    BufferDesc desc{sizeof(lhs_values), alignof(float), MemoryDomain::Device, false, "gles-add-validation"};
    std::shared_ptr<Buffer> lhs_buffer, rhs_buffer, output_buffer;
    status = backend.allocate(desc, lhs_buffer); if (!status.ok()) return status;
    status = backend.allocate(desc, rhs_buffer); if (!status.ok()) return status;
    status = backend.allocate(desc, output_buffer); if (!status.ok()) return status;
    status = backend.upload(lhs_buffer, 0, lhs_values, sizeof(lhs_values)); if (!status.ok()) return status;
    status = backend.upload(rhs_buffer, 0, rhs_values, sizeof(rhs_values)); if (!status.ok()) return status;

    TensorDesc tensor_desc;
    tensor_desc.shape = {4};
    tensor_desc.data_type = DataType::F32;
    tensor_desc.memory = MemoryDomain::Device;
    Tensor lhs, rhs, output;
    status = Tensor::create(tensor_desc, lhs_buffer, 0, lhs); if (!status.ok()) return status;
    status = Tensor::create(tensor_desc, rhs_buffer, 0, rhs); if (!status.ok()) return status;
    status = Tensor::create(tensor_desc, output_buffer, 0, output); if (!status.ok()) return status;

    ComputeDispatch dispatch;
    status = build_add_f32_gles_candidate(lhs, rhs, output, dispatch);
    if (!status.ok()) return status;
    if (dispatch.kernel_id != kernel->kernel_id || dispatch.glsl_source != kernel->glsl_source)
        return {StatusCode::InvalidArgument, "GLES Add validation source does not match the registered candidate"};
    dispatch.validation = KernelValidation::Validated;
    std::unique_ptr<CommandBatch> commands;
    status = backend.begin_commands(commands); if (!status.ok()) return status;
    status = commands->dispatch_compute(dispatch); if (!status.ok()) return status;
    status = backend.submit(std::move(commands)); if (!status.ok()) return status;
    status = backend.synchronize(); if (!status.ok()) return status;
    status = backend.readback(output_buffer, 0, actual, sizeof(actual)); if (!status.ok()) return status;

    for (size_t i = 0; i < 4; ++i) {
        const float tolerance = 1.0e-5f * std::max(1.0f, std::fabs(expected[i]));
        if (!std::isfinite(actual[i]) || std::fabs(actual[i] - expected[i]) > tolerance)
            return {StatusCode::NumericalFailure, "GLES Add GPU output did not match the independent F32 reference"};
    }
    kernel->validation = KernelValidation::Validated;
    return Status::success();
}

Status build_multiply_f32_gles_candidate(const Tensor& lhs, const Tensor& rhs,
                                         const Tensor& output, ComputeDispatch& dispatch) {
    Status status = build_binary_f32_candidate(lhs, rhs, output, dispatch,
        "gles-multiply-f32-generic-unverified", "gles-multiply-f32-generic-v1", kMultiplyF32Wgsl);
    if (!status.ok()) return status;
    dispatch.backend = BackendKind::OpenGles;
    dispatch.kernel_id = "gles-multiply-f32-generic-v1";
    dispatch.wgsl_source.clear();
    dispatch.glsl_source = kMultiplyF32Glsl;
    return Status::success();
}

Status reference_multiply_f32(const float* lhs, const float* rhs, float* output,
                              uint64_t element_count) {
    if ((!lhs || !rhs || !output) && element_count != 0)
        return {StatusCode::InvalidArgument, "Multiply reference received a null pointer"};
    for (uint64_t i = 0; i < element_count; ++i) output[i] = lhs[i] * rhs[i];
    return Status::success();
}

const char* matmul_f32_wgsl_source() noexcept { return kMatMulF32Wgsl; }

Status build_matmul_f32_candidate(const Tensor& lhs, const Tensor& rhs,
                                  const Tensor& output, ComputeDispatch& dispatch) {
    dispatch = ComputeDispatch{};
    if (!is_contiguous_f32(lhs) || !is_contiguous_f32(rhs) || !is_contiguous_f32(output))
        return {StatusCode::Unsupported, "MatMul candidate requires contiguous device/shared F32 tensors"};
    const auto& a = lhs.descriptor().shape;
    const auto& b = rhs.descriptor().shape;
    const auto& c = output.descriptor().shape;
    if (a.size() != 2 || b.size() != 2 || c.size() != 2 || a[1] != b[0] ||
        c[0] != a[0] || c[1] != b[1])
        return {StatusCode::InvalidArgument, "MatMul expects [M,K] x [K,N] -> [M,N]"};
    if (!lhs.buffer() || !rhs.buffer() || !output.buffer() ||
        output.buffer().get() == lhs.buffer().get() || output.buffer().get() == rhs.buffer().get())
        return {StatusCode::Unsupported, "MatMul output must use a distinct buffer"};
    if (a[0] > UINT32_MAX || a[1] > UINT32_MAX || b[1] > UINT32_MAX ||
        lhs.element_count() > UINT32_MAX || rhs.element_count() > UINT32_MAX ||
        output.element_count() > UINT32_MAX ||
        a[0] == 0 || a[1] == 0 || b[1] == 0)
        return {StatusCode::Unsupported, "MatMul dimensions exceed the generic WGSL kernel range"};
    const uint64_t groups_x = (b[1] + 7) / 8;
    const uint64_t groups_y = (a[0] + 7) / 8;
    if (groups_x > UINT32_MAX || groups_y > UINT32_MAX)
        return {StatusCode::Unsupported, "MatMul dispatch dimensions exceed uint32"};
    dispatch.label = "matmul-f32-generic-unverified";
    dispatch.kernel_id = "dawn-matmul-f32-generic-v1";
    dispatch.backend = BackendKind::DawnWebGpu;
    dispatch.wgsl_source = kMatMulF32Wgsl;
    dispatch.workgroup_size_x = 8;
    dispatch.workgroup_size_y = 8;
    dispatch.wgsl_source = replace_once(std::move(dispatch.wgsl_source), "const ROWS: u32 = 1u;",
        "const ROWS: u32 = " + std::to_string(a[0]) + "u;");
    dispatch.wgsl_source = replace_once(std::move(dispatch.wgsl_source), "const COLUMNS: u32 = 1u;",
        "const COLUMNS: u32 = " + std::to_string(b[1]) + "u;");
    dispatch.wgsl_source = replace_once(std::move(dispatch.wgsl_source), "const INNER: u32 = 1u;",
        "const INNER: u32 = " + std::to_string(a[1]) + "u;");
    dispatch.workgroups_x = static_cast<uint32_t>(groups_x);
    dispatch.workgroups_y = static_cast<uint32_t>(groups_y);
    dispatch.bindings = {
        {0, lhs.buffer(), lhs.byte_offset(), lhs.byte_size()},
        {1, rhs.buffer(), rhs.byte_offset(), rhs.byte_size()},
        {2, output.buffer(), output.byte_offset(), output.byte_size()},
    };
    return Status::success();
}

Status build_matmul_f32_gles_candidate(const Tensor& lhs, const Tensor& rhs,
                                       const Tensor& output, ComputeDispatch& dispatch) {
    Status status = build_matmul_f32_candidate(lhs, rhs, output, dispatch);
    if (!status.ok()) return status;
    const auto& a = lhs.descriptor().shape;
    const auto& b = rhs.descriptor().shape;
    dispatch.backend = BackendKind::OpenGles;
    dispatch.kernel_id = "gles-matmul-f32-generic-v1";
    dispatch.label = "gles-matmul-f32-generic-unverified";
    dispatch.wgsl_source.clear();
    dispatch.glsl_source = kMatMulF32Glsl;
    dispatch.glsl_source = replace_once(std::move(dispatch.glsl_source), "const uint ROWS = 1u;",
        "const uint ROWS = " + std::to_string(a[0]) + "u;");
    dispatch.glsl_source = replace_once(std::move(dispatch.glsl_source), "const uint COLUMNS = 1u;",
        "const uint COLUMNS = " + std::to_string(b[1]) + "u;");
    dispatch.glsl_source = replace_once(std::move(dispatch.glsl_source), "const uint INNER = 1u;",
        "const uint INNER = " + std::to_string(a[1]) + "u;");
    return Status::success();
}

Status reference_matmul_f32(const float* lhs, const float* rhs, float* output,
                            uint64_t rows, uint64_t columns, uint64_t inner) {
    if ((!lhs || !rhs || !output) && rows && columns && inner)
        return {StatusCode::InvalidArgument, "MatMul reference received a null pointer"};
    if (rows == 0 || columns == 0 || inner == 0)
        return {StatusCode::InvalidArgument, "MatMul dimensions must be non-zero"};
    for (uint64_t row = 0; row < rows; ++row) {
        for (uint64_t column = 0; column < columns; ++column) {
            float sum = 0.0f;
            for (uint64_t k = 0; k < inner; ++k)
                sum += lhs[row * inner + k] * rhs[k * columns + column];
            output[row * columns + column] = sum;
        }
    }
    return Status::success();
}

const char* rms_norm_f32_wgsl_source() noexcept { return kRmsNormF32Wgsl; }

Status build_rms_norm_f32_candidate(const Tensor& input, const Tensor& weight,
                                    const Tensor& output, float epsilon,
                                    ComputeDispatch& dispatch) {
    dispatch = ComputeDispatch{};
    if (!is_contiguous_f32(input) || !is_contiguous_f32(weight) || !is_contiguous_f32(output))
        return {StatusCode::Unsupported, "RMSNorm candidate requires contiguous device/shared F32 tensors"};
    const auto& x = input.descriptor().shape;
    const auto& w = weight.descriptor().shape;
    if (x.size() != 2 || w.size() != 1 || output.descriptor().shape != x ||
        w[0] != x[1] || x[0] == 0 || x[1] == 0 || x[0] > UINT32_MAX || x[1] > UINT32_MAX ||
        input.element_count() > UINT32_MAX || output.element_count() > UINT32_MAX)
        return {StatusCode::InvalidArgument, "RMSNorm expects input/output [rows,columns] and weight [columns]"};
    if (!std::isfinite(epsilon) || epsilon <= 0.0f)
        return {StatusCode::InvalidArgument, "RMSNorm epsilon must be finite and positive"};
    if (!input.buffer() || !weight.buffer() || !output.buffer() ||
        output.buffer().get() == input.buffer().get() || output.buffer().get() == weight.buffer().get())
        return {StatusCode::Unsupported, "RMSNorm output must use a distinct buffer"};
    dispatch.label = "rms-norm-f32-generic-unverified";
    dispatch.kernel_id = "dawn-rms-norm-f32-generic-v1";
    dispatch.backend = BackendKind::DawnWebGpu;
    dispatch.wgsl_source = kRmsNormF32Wgsl;
    dispatch.workgroup_size_x = 64;
    dispatch.wgsl_source = replace_once(std::move(dispatch.wgsl_source), "const COLUMNS: u32 = 1u;",
        "const COLUMNS: u32 = " + std::to_string(x[1]) + "u;");
    std::ostringstream epsilon_literal;
    epsilon_literal << std::setprecision(9) << epsilon;
    std::string epsilon_text = epsilon_literal.str();
    if (epsilon_text.find_first_of(".eE") == std::string::npos) epsilon_text += ".0";
    dispatch.wgsl_source = replace_once(std::move(dispatch.wgsl_source), "const EPSILON: f32 = 0.00001;",
        "const EPSILON: f32 = " + epsilon_text + ";");
    dispatch.workgroups_x = static_cast<uint32_t>(x[0]);
    dispatch.bindings = {
        {0, input.buffer(), input.byte_offset(), input.byte_size()},
        {1, weight.buffer(), weight.byte_offset(), weight.byte_size()},
        {2, output.buffer(), output.byte_offset(), output.byte_size()},
    };
    return Status::success();
}

Status build_rms_norm_f32_gles_candidate(const Tensor& input, const Tensor& weight,
                                         const Tensor& output, float epsilon,
                                         ComputeDispatch& dispatch) {
    Status status = build_rms_norm_f32_candidate(input, weight, output, epsilon, dispatch);
    if (!status.ok()) return status;
    const auto& shape = input.descriptor().shape;
    std::ostringstream epsilon_literal;
    epsilon_literal << std::setprecision(9) << epsilon;
    std::string epsilon_text = epsilon_literal.str();
    if (epsilon_text.find_first_of(".eE") == std::string::npos) epsilon_text += ".0";
    dispatch.backend = BackendKind::OpenGles;
    dispatch.kernel_id = "gles-rms-norm-f32-generic-v1";
    dispatch.label = "gles-rms-norm-f32-generic-unverified";
    dispatch.wgsl_source.clear();
    dispatch.glsl_source = kRmsNormF32Glsl;
    dispatch.glsl_source = replace_once(std::move(dispatch.glsl_source), "const uint ROWS = 1u;",
        "const uint ROWS = " + std::to_string(shape[0]) + "u;");
    dispatch.glsl_source = replace_once(std::move(dispatch.glsl_source), "const uint COLUMNS = 1u;",
        "const uint COLUMNS = " + std::to_string(shape[1]) + "u;");
    dispatch.glsl_source = replace_once(std::move(dispatch.glsl_source), "const float EPSILON = 0.00001;",
        "const float EPSILON = " + epsilon_text + ";");
    return Status::success();
}

Status reference_rms_norm_f32(const float* input, const float* weight, float* output,
                              uint64_t rows, uint64_t columns, float epsilon) {
    if ((!input || !weight || !output) && rows && columns)
        return {StatusCode::InvalidArgument, "RMSNorm reference received a null pointer"};
    if (rows == 0 || columns == 0 || !std::isfinite(epsilon) || epsilon <= 0.0f)
        return {StatusCode::InvalidArgument, "RMSNorm dimensions or epsilon are invalid"};
    for (uint64_t row = 0; row < rows; ++row) {
        double sum = 0.0;
        for (uint64_t column = 0; column < columns; ++column) {
            const double value = input[row * columns + column];
            sum += value * value;
        }
        const float inverse = 1.0f / std::sqrt(static_cast<float>(sum / columns) + epsilon);
        for (uint64_t column = 0; column < columns; ++column)
            output[row * columns + column] = input[row * columns + column] * inverse * weight[column];
    }
    return Status::success();
}



const char* rope_f32_wgsl_source() noexcept { return kRopeF32Wgsl; }

Status build_rope_f32_candidate(const Tensor& input, const Tensor& output,
                                const RopeConfig& config, ComputeDispatch& dispatch) {
    dispatch = ComputeDispatch{};
    if (!is_contiguous_f32(input) || !is_contiguous_f32(output))
        return {StatusCode::Unsupported, "RoPE requires contiguous device/shared F32 tensors"};
    const auto& shape = input.descriptor().shape;
    if (shape.size() != 3 || output.descriptor().shape != shape ||
        shape[0] == 0 || shape[1] == 0 || shape[2] == 0 ||
        shape[0] > UINT32_MAX || shape[1] > UINT32_MAX || shape[2] > UINT32_MAX ||
        input.element_count() > UINT32_MAX || output.element_count() > UINT32_MAX ||
        !input.buffer() || !output.buffer() || input.buffer().get() == output.buffer().get())
        return {StatusCode::InvalidArgument, "RoPE expects matching [tokens,heads,head_dimension] tensors in WGSL range"};
    if (config.rotary_dimensions < 2 || config.rotary_dimensions > shape[2] ||
        (config.rotary_dimensions & 1u) != 0 || !std::isfinite(config.theta) ||
        config.theta <= 1.0f || shape[0] > UINT32_MAX - config.start_position)
        return {StatusCode::InvalidArgument, "RoPE dimensions, position or theta are invalid"};
    dispatch.label = "rope-f32-generic-unverified";
    dispatch.kernel_id = "dawn-rope-f32-generic-v1";
    dispatch.backend = BackendKind::DawnWebGpu;
    dispatch.wgsl_source = kRopeF32Wgsl;
    dispatch.workgroup_size_x = 64;
    const std::pair<std::string, std::string> constants[] = {
        {"const TOTAL_ELEMENTS: u32 = 1u;", "const TOTAL_ELEMENTS: u32 = " + std::to_string(input.element_count()) + "u;"},
        {"const HEADS: u32 = 1u;", "const HEADS: u32 = " + std::to_string(shape[1]) + "u;"},
        {"const HEAD_DIM: u32 = 2u;", "const HEAD_DIM: u32 = " + std::to_string(shape[2]) + "u;"},
        {"const ROTARY_DIM: u32 = 2u;", "const ROTARY_DIM: u32 = " + std::to_string(config.rotary_dimensions) + "u;"},
        {"const START_POSITION: u32 = 0u;", "const START_POSITION: u32 = " + std::to_string(config.start_position) + "u;"},
        {"const NEOX_STYLE: u32 = 0u;", std::string("const NEOX_STYLE: u32 = ") + (config.neox_style ? "1u;" : "0u;")},
    };
    for (const auto& constant : constants) {
        const size_t position = dispatch.wgsl_source.find(constant.first);
        if (position == std::string::npos)
            return {StatusCode::BackendFailure, "RoPE WGSL constant placeholder is missing"};
        dispatch.wgsl_source.replace(position, constant.first.size(), constant.second);
    }
    std::ostringstream theta_literal;
    theta_literal << std::setprecision(9) << config.theta;
    std::string theta_text = theta_literal.str();
    if (theta_text.find_first_of(".eE") == std::string::npos) theta_text += ".0";
    dispatch.wgsl_source = replace_once(std::move(dispatch.wgsl_source),
        "const THETA: f32 = 10000.0;", "const THETA: f32 = " + theta_text + ";");
    dispatch.workgroups_x = static_cast<uint32_t>((input.element_count() + 63) / 64);
    dispatch.bindings = {
        {0, input.buffer(), input.byte_offset(), input.byte_size()},
        {1, output.buffer(), output.byte_offset(), output.byte_size()},
    };
    return Status::success();
}

Status build_rope_f32_gles_candidate(const Tensor& input, const Tensor& output,
                                     const RopeConfig& config, ComputeDispatch& dispatch) {
    Status status = build_rope_f32_candidate(input, output, config, dispatch);
    if (!status.ok()) return status;
    const auto& shape = input.descriptor().shape;
    std::ostringstream theta_literal;
    theta_literal << std::setprecision(9) << config.theta;
    std::string theta_text = theta_literal.str();
    if (theta_text.find_first_of(".eE") == std::string::npos) theta_text += ".0";
    dispatch.backend = BackendKind::OpenGles;
    dispatch.kernel_id = "gles-rope-f32-generic-v1";
    dispatch.label = "gles-rope-f32-generic-unverified";
    dispatch.wgsl_source.clear();
    dispatch.glsl_source = kRopeF32Glsl;
    const std::pair<std::string, std::string> constants[] = {
        {"const uint TOTAL_ELEMENTS = 1u;", "const uint TOTAL_ELEMENTS = " + std::to_string(input.element_count()) + "u;"},
        {"const uint HEADS = 1u;", "const uint HEADS = " + std::to_string(shape[1]) + "u;"},
        {"const uint HEAD_DIM = 2u;", "const uint HEAD_DIM = " + std::to_string(shape[2]) + "u;"},
        {"const uint ROTARY_DIM = 2u;", "const uint ROTARY_DIM = " + std::to_string(config.rotary_dimensions) + "u;"},
        {"const uint START_POSITION = 0u;", "const uint START_POSITION = " + std::to_string(config.start_position) + "u;"},
        {"const float THETA = 10000.0;", "const float THETA = " + theta_text + ";"},
        {"const uint NEOX_STYLE = 0u;", std::string("const uint NEOX_STYLE = ") + (config.neox_style ? "1u;" : "0u;")}
    };
    for (const auto& constant : constants) {
        const size_t pos = dispatch.glsl_source.find(constant.first);
        if (pos == std::string::npos) return {StatusCode::BackendFailure, "RoPE GLSL constant placeholder is missing"};
        dispatch.glsl_source.replace(pos, constant.first.size(), constant.second);
    }
    return Status::success();
}

Status reference_rope_f32(const float* input, float* output,
                          uint32_t tokens, uint32_t heads, uint32_t head_dimension,
                          const RopeConfig& config) {
    if ((!input || !output) && tokens && heads && head_dimension)
        return {StatusCode::InvalidArgument, "RoPE reference received a null pointer"};
    if (tokens == 0 || heads == 0 || head_dimension == 0 ||
        config.rotary_dimensions < 2 || config.rotary_dimensions > head_dimension ||
        (config.rotary_dimensions & 1u) != 0 || !std::isfinite(config.theta) ||
        config.theta <= 1.0f || tokens > UINT32_MAX - config.start_position)
        return {StatusCode::InvalidArgument, "RoPE reference configuration is invalid"};
    uint64_t elements = tokens;
    if (heads != 0 && elements > std::numeric_limits<uint64_t>::max() / heads)
        return {StatusCode::InvalidArgument, "RoPE element count overflows"};
    elements *= heads;
    if (head_dimension != 0 && elements > std::numeric_limits<uint64_t>::max() / head_dimension)
        return {StatusCode::InvalidArgument, "RoPE element count overflows"};
    elements *= head_dimension;
    for (uint64_t i = 0; i < elements; ++i) output[i] = input[i];
    const uint32_t half = config.rotary_dimensions / 2;
    for (uint32_t token = 0; token < tokens; ++token) {
        const double position = static_cast<double>(config.start_position + token);
        for (uint32_t head = 0; head < heads; ++head) {
            const uint64_t base = (static_cast<uint64_t>(token) * heads + head) * head_dimension;
            for (uint32_t pair = 0; pair < half; ++pair) {
                const uint32_t first = config.neox_style ? pair : pair * 2;
                const uint32_t second = config.neox_style ? pair + half : pair * 2 + 1;
                const double exponent = 2.0 * static_cast<double>(pair) / config.rotary_dimensions;
                const double angle = position / std::pow(static_cast<double>(config.theta), exponent);
                const float cosine = static_cast<float>(std::cos(angle));
                const float sine = static_cast<float>(std::sin(angle));
                const float x0 = input[base + first];
                const float x1 = input[base + second];
                output[base + first] = x0 * cosine - x1 * sine;
                output[base + second] = x1 * cosine + x0 * sine;
            }
        }
    }
    return Status::success();
}

const char* silu_f32_wgsl_source() noexcept { return kSiluF32Wgsl; }

Status build_silu_f32_candidate(const Tensor& input, const Tensor& output,
                                ComputeDispatch& dispatch) {
    dispatch = ComputeDispatch{};
    if (!is_contiguous_f32(input) || !is_contiguous_f32(output))
        return {StatusCode::Unsupported, "SiLU requires contiguous device/shared F32 tensors"};
    if (input.descriptor().shape != output.descriptor().shape ||
        input.element_count() == 0 || input.element_count() > UINT32_MAX ||
        !input.buffer() || !output.buffer() || input.buffer().get() == output.buffer().get())
        return {StatusCode::InvalidArgument, "SiLU input/output shape, size or buffer is invalid"};
    dispatch.label = "silu-f32-generic-unverified";
    dispatch.kernel_id = "dawn-silu-f32-generic-v1";
    dispatch.backend = BackendKind::DawnWebGpu;
    dispatch.wgsl_source = kSiluF32Wgsl;
    dispatch.workgroup_size_x = 64;
    const std::string placeholder = "const ELEMENT_COUNT: u32 = 1u;";
    const std::string replacement = "const ELEMENT_COUNT: u32 = " +
                                    std::to_string(input.element_count()) + "u;";
    const size_t position = dispatch.wgsl_source.find(placeholder);
    if (position == std::string::npos)
        return {StatusCode::BackendFailure, "SiLU WGSL element-count placeholder is missing"};
    dispatch.wgsl_source.replace(position, placeholder.size(), replacement);
    dispatch.workgroups_x = static_cast<uint32_t>((input.element_count() + 63) / 64);
    dispatch.bindings = {
        {0, input.buffer(), input.byte_offset(), input.byte_size()},
        {1, output.buffer(), output.byte_offset(), output.byte_size()},
    };
    return Status::success();
}

Status build_silu_f32_gles_candidate(const Tensor& input, const Tensor& output,
                                     ComputeDispatch& dispatch) {
    Status status = build_silu_f32_candidate(input, output, dispatch);
    if (!status.ok()) return status;
    dispatch.backend = BackendKind::OpenGles;
    dispatch.kernel_id = "gles-silu-f32-generic-v1";
    dispatch.label = "gles-silu-f32-generic-unverified";
    dispatch.wgsl_source.clear();
    dispatch.glsl_source = kSiluF32Glsl;
    return Status::success();
}

Status reference_silu_f32(const float* input, float* output, uint64_t elements) {
    if ((!input || !output) && elements != 0)
        return {StatusCode::InvalidArgument, "SiLU reference received a null pointer"};
    for (uint64_t i = 0; i < elements; ++i)
        output[i] = input[i] / (1.0f + std::exp(-input[i]));
    return Status::success();
}

const char* softmax_f32_wgsl_source() noexcept { return kSoftmaxF32Wgsl; }

Status build_softmax_f32_candidate(const Tensor& input, const Tensor& output,
                                   ComputeDispatch& dispatch) {
    dispatch = ComputeDispatch{};
    if (!is_contiguous_f32(input) || !is_contiguous_f32(output))
        return {StatusCode::Unsupported, "Softmax requires contiguous device/shared F32 tensors"};
    const auto& shape = input.descriptor().shape;
    if (shape.size() != 2 || output.descriptor().shape != shape ||
        shape[0] == 0 || shape[1] == 0 || shape[0] > UINT32_MAX ||
        shape[1] > UINT32_MAX || input.element_count() > UINT32_MAX ||
        !input.buffer() || !output.buffer() || input.buffer().get() == output.buffer().get())
        return {StatusCode::InvalidArgument, "Softmax expects valid [rows,columns] input/output tensors"};
    dispatch.label = "softmax-f32-generic-unverified";
    dispatch.kernel_id = "dawn-softmax-f32-generic-v1";
    dispatch.backend = BackendKind::DawnWebGpu;
    dispatch.wgsl_source = kSoftmaxF32Wgsl;
    dispatch.workgroup_size_x = 64;
    const std::string placeholder = "const COLUMNS: u32 = 1u;";
    const std::string replacement = "const COLUMNS: u32 = " + std::to_string(shape[1]) + "u;";
    const size_t position = dispatch.wgsl_source.find(placeholder);
    if (position == std::string::npos)
        return {StatusCode::BackendFailure, "Softmax WGSL columns placeholder is missing"};
    dispatch.wgsl_source.replace(position, placeholder.size(), replacement);
    dispatch.workgroups_x = static_cast<uint32_t>(shape[0]);
    dispatch.bindings = {
        {0, input.buffer(), input.byte_offset(), input.byte_size()},
        {1, output.buffer(), output.byte_offset(), output.byte_size()},
    };
    return Status::success();
}

Status build_softmax_f32_gles_candidate(const Tensor& input, const Tensor& output,
                                        ComputeDispatch& dispatch) {
    Status status = build_softmax_f32_candidate(input, output, dispatch);
    if (!status.ok()) return status;
    const auto& shape = input.descriptor().shape;
    dispatch.backend = BackendKind::OpenGles;
    dispatch.kernel_id = "gles-softmax-f32-generic-v1";
    dispatch.label = "gles-softmax-f32-generic-unverified";
    dispatch.wgsl_source.clear();
    dispatch.glsl_source = kSoftmaxF32Glsl;
    dispatch.glsl_source = replace_once(std::move(dispatch.glsl_source), "const uint ROWS = 1u;",
        "const uint ROWS = " + std::to_string(shape[0]) + "u;");
    dispatch.glsl_source = replace_once(std::move(dispatch.glsl_source), "const uint COLUMNS = 1u;",
        "const uint COLUMNS = " + std::to_string(shape[1]) + "u;");
    return Status::success();
}

Status reference_softmax_f32(const float* input, float* output,
                             uint64_t rows, uint64_t columns) {
    if ((!input || !output) && rows && columns)
        return {StatusCode::InvalidArgument, "Softmax reference received a null pointer"};
    if (rows == 0 || columns == 0)
        return {StatusCode::InvalidArgument, "Softmax dimensions must be non-zero"};
    for (uint64_t row = 0; row < rows; ++row) {
        float maximum = -std::numeric_limits<float>::infinity();
        for (uint64_t column = 0; column < columns; ++column) {
            const float value = input[row * columns + column];
            if (!std::isfinite(value))
                return {StatusCode::NumericalFailure, "Softmax input contains NaN or infinity"};
            maximum = std::max(maximum, value);
        }
        double sum = 0.0;
        for (uint64_t column = 0; column < columns; ++column)
            sum += std::exp(static_cast<double>(input[row * columns + column] - maximum));
        if (!std::isfinite(sum) || sum <= 0.0)
            return {StatusCode::NumericalFailure, "Softmax normalization failed"};
        for (uint64_t column = 0; column < columns; ++column)
            output[row * columns + column] = static_cast<float>(
                std::exp(static_cast<double>(input[row * columns + column] - maximum)) / sum);
    }
    return Status::success();
}

Status reference_add_f32(const float* lhs, const float* rhs, float* output,
                         uint64_t element_count) {
    if ((!lhs || !rhs || !output) && element_count != 0) {
        return {StatusCode::InvalidArgument, "Add reference received a null pointer"};
    }
    for (uint64_t i = 0; i < element_count; ++i) output[i] = lhs[i] + rhs[i];
    return Status::success();
}

Status register_builtin_operator_candidates(OperatorRegistry& registry) {
    KernelDefinition add;
    add.operator_name = "Add";
    add.backend = BackendKind::DawnWebGpu;
    add.input_type = DataType::F32;
    add.output_type = DataType::F32;
    add.min_input_count = 2;
    add.max_input_count = 2;
    add.min_rank = 1;
    add.max_rank = 16;
    add.validation = KernelValidation::Unverified;
    add.kernel_id = "dawn-add-f32-generic-v1";
    add.wgsl_source = kAddF32Wgsl;
    add.entry_point = "main";
    Status status = registry.register_kernel(std::move(add));
    if (!status.ok()) return status;

    KernelDefinition multiply;
    multiply.operator_name = "Multiply";
    multiply.backend = BackendKind::DawnWebGpu;
    multiply.input_type = DataType::F32;
    multiply.output_type = DataType::F32;
    multiply.min_input_count = 2;
    multiply.max_input_count = 2;
    multiply.min_rank = 1;
    multiply.max_rank = 16;
    multiply.validation = KernelValidation::Unverified;
    multiply.kernel_id = "dawn-multiply-f32-generic-v1";
    multiply.wgsl_source = kMultiplyF32Wgsl;
    status = registry.register_kernel(std::move(multiply));
    if (!status.ok()) return status;


    KernelDefinition embedding;
    embedding.operator_name = "Embedding";
    embedding.backend = BackendKind::OpenGles;
    embedding.input_type = DataType::F32;
    embedding.input_types = {DataType::F32, DataType::U32};
    embedding.output_type = DataType::F32;
    embedding.min_input_count = 2;
    embedding.max_input_count = 2;
    embedding.min_rank = 2;
    embedding.max_rank = 2;
    embedding.validation = KernelValidation::Unverified;
    embedding.kernel_id = "gles-embedding-f32-generic-v1";
    embedding.glsl_source = kEmbeddingF32Glsl;
    Status status = registry.register_kernel(std::move(embedding));
    if (!status.ok()) return status;

    KernelDefinition rope;
    rope.operator_name = "RoPE";
    rope.backend = BackendKind::OpenGles;
    rope.input_type = DataType::F32;
    rope.output_type = DataType::F32;
    rope.min_input_count = 1;
    rope.max_input_count = 1;
    rope.min_rank = 3;
    rope.max_rank = 3;
    rope.validation = KernelValidation::Unverified;
    rope.kernel_id = "gles-rope-f32-generic-v1";
    rope.glsl_source = kRopeF32Glsl;
    status = registry.register_kernel(std::move(rope));
    if (!status.ok()) return status;

    KernelDefinition attention;
    attention.operator_name = "Attention";
    attention.backend = BackendKind::OpenGles;
    attention.input_type = DataType::F32;
    attention.output_type = DataType::F32;
    attention.min_input_count = 3;
    attention.max_input_count = 3;
    attention.min_rank = 3;
    attention.max_rank = 3;
    attention.validation = KernelValidation::Unverified;
    attention.kernel_id = "gles-attention-f32-causal-gqa-v1";
    attention.glsl_source = kAttentionF32Glsl;
    status = registry.register_kernel(std::move(attention));
    if (!status.ok()) return status;

    KernelDefinition matmul;
    matmul.operator_name = "MatMul";
    matmul.backend = BackendKind::DawnWebGpu;
    matmul.input_type = DataType::F32;
    matmul.output_type = DataType::F32;
    matmul.min_input_count = 2;
    matmul.max_input_count = 2;
    matmul.min_rank = 2;
    matmul.max_rank = 2;
    matmul.validation = KernelValidation::Unverified;
    matmul.kernel_id = "dawn-matmul-f32-generic-v1";
    matmul.wgsl_source = kMatMulF32Wgsl;
    status = registry.register_kernel(std::move(matmul));
    if (!status.ok()) return status;

    KernelDefinition rms;
    rms.operator_name = "RMSNorm";
    rms.backend = BackendKind::DawnWebGpu;
    rms.input_type = DataType::F32;
    rms.output_type = DataType::F32;
    rms.min_input_count = 2;
    rms.max_input_count = 2;
    rms.min_rank = 1;
    rms.max_rank = 2;
    rms.validation = KernelValidation::Unverified;
    rms.kernel_id = "dawn-rms-norm-f32-generic-v1";
    rms.wgsl_source = kRmsNormF32Wgsl;
    status = registry.register_kernel(std::move(rms));
    if (!status.ok()) return status;

    KernelDefinition silu;
    silu.operator_name = "SiLU";
    silu.backend = BackendKind::DawnWebGpu;
    silu.input_type = DataType::F32;
    silu.output_type = DataType::F32;
    silu.min_input_count = 1;
    silu.max_input_count = 1;
    silu.min_rank = 1;
    silu.max_rank = 16;
    silu.validation = KernelValidation::Unverified;
    silu.kernel_id = "dawn-silu-f32-generic-v1";
    silu.wgsl_source = kSiluF32Wgsl;
    status = registry.register_kernel(std::move(silu));
    if (!status.ok()) return status;

    KernelDefinition softmax;
    softmax.operator_name = "Softmax";
    softmax.backend = BackendKind::DawnWebGpu;
    softmax.input_type = DataType::F32;
    softmax.output_type = DataType::F32;
    softmax.min_input_count = 1;
    softmax.max_input_count = 1;
    softmax.min_rank = 2;
    softmax.max_rank = 2;
    softmax.validation = KernelValidation::Unverified;
    softmax.kernel_id = "dawn-softmax-f32-generic-v1";
    softmax.wgsl_source = kSoftmaxF32Wgsl;
    status = registry.register_kernel(std::move(softmax));
    if (!status.ok()) return status;


    KernelDefinition embedding;
    embedding.operator_name = "Embedding";
    embedding.kernel_id = "dawn-embedding-f32-generic-v1";
    embedding.backend = BackendKind::DawnWebGpu;
    embedding.input_type = DataType::F32;
    embedding.input_types = {DataType::F32, DataType::U32};
    embedding.output_type = DataType::F32;
    embedding.min_input_count = 2;
    embedding.max_input_count = 2;
    embedding.min_rank = 2;
    embedding.max_rank = 2;
    embedding.validation = KernelValidation::Unverified;
    embedding.wgsl_source = kEmbeddingF32Wgsl;
    status = registry.register_kernel(std::move(embedding));
    if (!status.ok()) return status;

    KernelDefinition q8_matmul;
    q8_matmul.operator_name = "MatMul_Q8_0";
    q8_matmul.kernel_id = "dawn-q8-0-matmul-f32-activation-v1";
    q8_matmul.backend = BackendKind::DawnWebGpu;
    q8_matmul.input_type = DataType::F32;
    q8_matmul.input_types = {DataType::F32, DataType::U32};
    q8_matmul.output_type = DataType::F32;
    q8_matmul.min_input_count = 2;
    q8_matmul.max_input_count = 2;
    q8_matmul.min_rank = 2;
    q8_matmul.max_rank = 2;
    q8_matmul.validation = KernelValidation::Unverified;
    q8_matmul.wgsl_source = q8_0_matmul_wgsl_source();
    status = registry.register_kernel(std::move(q8_matmul));
    if (!status.ok()) return status;

    KernelDefinition attention;
    attention.operator_name = "Attention";
    attention.backend = BackendKind::DawnWebGpu;
    attention.input_type = DataType::F32;
    attention.output_type = DataType::F32;
    attention.min_input_count = 3;
    attention.max_input_count = 3;
    attention.min_rank = 3;
    attention.max_rank = 3;
    attention.validation = KernelValidation::Unverified;
    attention.kernel_id = "dawn-attention-f32-causal-gqa-v1";
    attention.wgsl_source = kAttentionF32Wgsl;
    status = registry.register_kernel(std::move(attention));
    if (!status.ok()) return status;

    KernelDefinition rope;
    rope.operator_name = "RoPE";
    rope.backend = BackendKind::DawnWebGpu;
    rope.input_type = DataType::F32;
    rope.output_type = DataType::F32;
    rope.min_input_count = 1;
    rope.max_input_count = 1;
    rope.min_rank = 3;
    rope.max_rank = 3;
    rope.validation = KernelValidation::Unverified;
    rope.kernel_id = "dawn-rope-f32-generic-v1";
    rope.wgsl_source = kRopeF32Wgsl;
    status = registry.register_kernel(std::move(rope));
    if (!status.ok()) return status;

    KernelDefinition q5_0;
    q5_0.operator_name = "Dequantize_Q5_0";
    q5_0.backend = BackendKind::DawnWebGpu;
    q5_0.input_type = DataType::U32;
    q5_0.output_type = DataType::F32;
    q5_0.min_input_count = 1;
    q5_0.max_input_count = 1;
    q5_0.min_rank = 1;
    q5_0.max_rank = 16;
    q5_0.validation = KernelValidation::Unverified;
    q5_0.kernel_id = "dawn-q5-0-dequantize-generic-v1";
    q5_0.wgsl_source = q5_0_dequantize_wgsl_source();
    status = registry.register_kernel(std::move(q5_0));
    if (!status.ok()) return status;

    KernelDefinition q5_1;
    q5_1.operator_name = "Dequantize_Q5_1";
    q5_1.backend = BackendKind::DawnWebGpu;
    q5_1.input_type = DataType::U32;
    q5_1.output_type = DataType::F32;
    q5_1.min_input_count = 1;
    q5_1.max_input_count = 1;
    q5_1.min_rank = 1;
    q5_1.max_rank = 16;
    q5_1.validation = KernelValidation::Unverified;
    q5_1.kernel_id = "dawn-q5-1-dequantize-generic-v1";
    q5_1.wgsl_source = q5_1_dequantize_wgsl_source();
    status = registry.register_kernel(std::move(q5_1));
    if (!status.ok()) return status;

    KernelDefinition q4_1;
    q4_1.operator_name = "Dequantize_Q4_1";
    q4_1.backend = BackendKind::DawnWebGpu;
    q4_1.input_type = DataType::U32;
    q4_1.output_type = DataType::F32;
    q4_1.min_input_count = 1;
    q4_1.max_input_count = 1;
    q4_1.min_rank = 1;
    q4_1.max_rank = 16;
    q4_1.validation = KernelValidation::Unverified;
    q4_1.kernel_id = "dawn-q4-1-dequantize-generic-v1";
    q4_1.wgsl_source = q4_1_dequantize_wgsl_source();
    status = registry.register_kernel(std::move(q4_1));
    if (!status.ok()) return status;

    KernelDefinition q4;
    q4.operator_name = "Dequantize_Q4_0";
    q4.backend = BackendKind::DawnWebGpu;
    q4.input_type = DataType::U32;
    q4.output_type = DataType::F32;
    q4.min_input_count = 1;
    q4.max_input_count = 1;
    q4.min_rank = 1;
    q4.max_rank = 16;
    q4.validation = KernelValidation::Unverified;
    q4.kernel_id = "dawn-q4-0-dequantize-generic-v1";
    q4.wgsl_source = q4_0_dequantize_wgsl_source();
    status = registry.register_kernel(std::move(q4));
    if (!status.ok()) return status;

    KernelDefinition q8;
    q8.operator_name = "Dequantize_Q8_0";
    q8.backend = BackendKind::DawnWebGpu;
    q8.input_type = DataType::U32;
    q8.output_type = DataType::F32;
    q8.min_input_count = 1;
    q8.max_input_count = 1;
    q8.min_rank = 1;
    q8.max_rank = 16;
    q8.validation = KernelValidation::Unverified;
    q8.kernel_id = "dawn-q8-0-dequantize-generic-v1";
    q8.wgsl_source = q8_0_dequantize_wgsl_source();
    return registry.register_kernel(std::move(q8));
}


Status register_builtin_gles_operator_candidates(OperatorRegistry& registry) {
    KernelDefinition add;
    add.operator_name = "Add";
    add.backend = BackendKind::OpenGles;
    add.input_type = DataType::F32;
    add.output_type = DataType::F32;
    add.min_input_count = 2;
    add.max_input_count = 2;
    add.min_rank = 1;
    add.max_rank = 16;
    add.validation = KernelValidation::Unverified;
    add.kernel_id = "gles-add-f32-generic-v1";
    add.glsl_source = kAddF32Glsl;
    Status status = registry.register_kernel(std::move(add));
    if (!status.ok()) return status;

    KernelDefinition multiply;
    multiply.operator_name = "Multiply";
    multiply.backend = BackendKind::OpenGles;
    multiply.input_type = DataType::F32;
    multiply.output_type = DataType::F32;
    multiply.min_input_count = 2;
    multiply.max_input_count = 2;
    multiply.min_rank = 1;
    multiply.max_rank = 16;
    multiply.validation = KernelValidation::Unverified;
    multiply.kernel_id = "gles-multiply-f32-generic-v1";
    multiply.glsl_source = kMultiplyF32Glsl;
    status = registry.register_kernel(std::move(multiply));
    if (!status.ok()) return status;

    KernelDefinition matmul;
    matmul.operator_name = "MatMul";
    matmul.backend = BackendKind::OpenGles;
    matmul.input_type = DataType::F32;
    matmul.output_type = DataType::F32;
    matmul.min_input_count = 2;
    matmul.max_input_count = 2;
    matmul.min_rank = 2;
    matmul.max_rank = 2;
    matmul.validation = KernelValidation::Unverified;
    matmul.kernel_id = "gles-matmul-f32-generic-v1";
    matmul.glsl_source = kMatMulF32Glsl;
    status = registry.register_kernel(std::move(matmul));
    if (!status.ok()) return status;

    KernelDefinition rms;
    rms.operator_name = "RMSNorm";
    rms.backend = BackendKind::OpenGles;
    rms.input_type = DataType::F32;
    rms.output_type = DataType::F32;
    rms.min_input_count = 2;
    rms.max_input_count = 2;
    rms.min_rank = 2;
    rms.max_rank = 2;
    rms.validation = KernelValidation::Unverified;
    rms.kernel_id = "gles-rms-norm-f32-generic-v1";
    rms.glsl_source = kRmsNormF32Glsl;
    status = registry.register_kernel(std::move(rms));
    if (!status.ok()) return status;

    KernelDefinition silu;
    silu.operator_name = "SiLU";
    silu.backend = BackendKind::OpenGles;
    silu.input_type = DataType::F32;
    silu.output_type = DataType::F32;
    silu.min_input_count = 1;
    silu.max_input_count = 1;
    silu.min_rank = 1;
    silu.max_rank = 16;
    silu.validation = KernelValidation::Unverified;
    silu.kernel_id = "gles-silu-f32-generic-v1";
    silu.glsl_source = kSiluF32Glsl;
    status = registry.register_kernel(std::move(silu));
    if (!status.ok()) return status;

    KernelDefinition softmax;
    softmax.operator_name = "Softmax";
    softmax.backend = BackendKind::OpenGles;
    softmax.input_type = DataType::F32;
    softmax.output_type = DataType::F32;
    softmax.min_input_count = 1;
    softmax.max_input_count = 1;
    softmax.min_rank = 2;
    softmax.max_rank = 2;
    softmax.validation = KernelValidation::Unverified;
    softmax.kernel_id = "gles-softmax-f32-generic-v1";
    softmax.glsl_source = kSoftmaxF32Glsl;
    status = registry.register_kernel(std::move(softmax));
    if (!status.ok()) return status;

    KernelDefinition rope;
    rope.operator_name = "RoPE";
    rope.backend = BackendKind::OpenGles;
    rope.input_type = DataType::F32;
    rope.output_type = DataType::F32;
    rope.min_input_count = 1;
    rope.max_input_count = 1;
    rope.min_rank = 3;
    rope.max_rank = 3;
    rope.validation = KernelValidation::Unverified;
    rope.kernel_id = "gles-rope-f32-generic-v1";
    rope.glsl_source = kRopeF32Glsl;
    status = registry.register_kernel(std::move(rope));
    if (!status.ok()) return status;

    KernelDefinition embedding;
    embedding.operator_name = "Embedding";
    embedding.backend = BackendKind::OpenGles;
    embedding.input_type = DataType::F32;
    embedding.input_types = {DataType::F32, DataType::U32};
    embedding.output_type = DataType::F32;
    embedding.min_input_count = 2;
    embedding.max_input_count = 2;
    embedding.min_rank = 2;
    embedding.max_rank = 2;
    embedding.validation = KernelValidation::Unverified;
    embedding.kernel_id = "gles-embedding-f32-generic-v1";
    embedding.glsl_source = kEmbeddingF32Glsl;
    status = registry.register_kernel(std::move(embedding));
    if (!status.ok()) return status;

    KernelDefinition attention;
    attention.operator_name = "Attention";
    attention.backend = BackendKind::OpenGles;
    attention.input_type = DataType::F32;
    attention.output_type = DataType::F32;
    attention.min_input_count = 3;
    attention.max_input_count = 3;
    attention.min_rank = 3;
    attention.max_rank = 3;
    attention.validation = KernelValidation::Unverified;
    attention.kernel_id = "gles-attention-f32-causal-gqa-v1";
    attention.glsl_source = kAttentionF32Glsl;
    status = registry.register_kernel(std::move(attention));
    if (!status.ok()) return status;

    KernelDefinition q8_matmul;
    q8_matmul.operator_name = "MatMul_Q8_0";
    q8_matmul.backend = BackendKind::OpenGles;
    q8_matmul.input_type = DataType::F32;
    q8_matmul.input_types = {DataType::F32, DataType::U32};
    q8_matmul.output_type = DataType::F32;
    q8_matmul.min_input_count = 2;
    q8_matmul.max_input_count = 2;
    q8_matmul.min_rank = 2;
    q8_matmul.max_rank = 2;
    q8_matmul.validation = KernelValidation::Unverified;
    q8_matmul.kernel_id = "gles-q8-0-matmul-f32-activation-v1";
    q8_matmul.glsl_source = q8_0_matmul_glsl_source();
    return registry.register_kernel(std::move(q8_matmul));
}

} // namespace ai_chat::gpu_engine
