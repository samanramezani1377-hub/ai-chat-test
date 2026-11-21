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
                                    const char* label, const char* shader) {
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
    dispatch.wgsl_source = shader;
    dispatch.workgroups_x = static_cast<uint32_t>((lhs.element_count() + 63) / 64);
    dispatch.bindings = {
        {0, lhs.buffer(), lhs.byte_offset(), lhs.byte_size()},
        {1, rhs.buffer(), rhs.byte_offset(), rhs.byte_size()},
        {2, output.buffer(), output.byte_offset(), output.byte_size()},
    };
    return Status::success();
}

} // namespace

const char* add_f32_wgsl_source() noexcept { return kAddF32Wgsl; }

Status build_add_f32_candidate(const Tensor& lhs, const Tensor& rhs,
                               const Tensor& output, ComputeDispatch& dispatch) {
    return build_binary_f32_candidate(lhs, rhs, output, dispatch,
                                      "add-f32-generic-unverified", kAddF32Wgsl);
}

const char* multiply_f32_wgsl_source() noexcept { return kMultiplyF32Wgsl; }

Status build_multiply_f32_candidate(const Tensor& lhs, const Tensor& rhs,
                                    const Tensor& output, ComputeDispatch& dispatch) {
    return build_binary_f32_candidate(lhs, rhs, output, dispatch,
                                      "multiply-f32-generic-unverified", kMultiplyF32Wgsl);
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
    dispatch.wgsl_source = kMatMulF32Wgsl;
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
    dispatch.wgsl_source = kRmsNormF32Wgsl;
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
    dispatch.wgsl_source = kRopeF32Wgsl;
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
    uint64_t elements = static_cast<uint64_t>(tokens) * heads * head_dimension;
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
    dispatch.wgsl_source = kSiluF32Wgsl;
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
    dispatch.wgsl_source = kSoftmaxF32Wgsl;
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

} // namespace ai_chat::gpu_engine
