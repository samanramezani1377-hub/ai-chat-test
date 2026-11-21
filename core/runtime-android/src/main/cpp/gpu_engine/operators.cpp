#include "operators.h"

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

} // namespace

const char* add_f32_wgsl_source() noexcept { return kAddF32Wgsl; }

Status build_add_f32_candidate(const Tensor& lhs, const Tensor& rhs,
                               const Tensor& output, ComputeDispatch& dispatch) {
    dispatch = ComputeDispatch{};
    if (!is_contiguous_f32(lhs) || !is_contiguous_f32(rhs) || !is_contiguous_f32(output)) {
        return {StatusCode::Unsupported, "Generic Add kernel currently requires contiguous device/shared F32 tensors"};
    }
    if (lhs.descriptor().shape != rhs.descriptor().shape ||
        lhs.descriptor().shape != output.descriptor().shape ||
        lhs.element_count() != rhs.element_count() ||
        lhs.element_count() != output.element_count()) {
        return {StatusCode::InvalidArgument, "Add tensor shapes must match exactly"};
    }
    if (!lhs.buffer() || !rhs.buffer() || !output.buffer() ||
        output.buffer().get() == lhs.buffer().get() ||
        output.buffer().get() == rhs.buffer().get()) {
        return {StatusCode::Unsupported, "Add output must use a distinct valid buffer until aliasing is validated"};
    }
    if (lhs.byte_size() != rhs.byte_size() || lhs.byte_size() != output.byte_size() ||
        lhs.byte_size() != lhs.element_count() * sizeof(float)) {
        return {StatusCode::InvalidArgument, "Add tensor byte ranges are inconsistent"};
    }
    constexpr uint64_t kWorkgroupSize = 64;
    const uint64_t groups = 1 + (lhs.element_count() - 1) / kWorkgroupSize;
    if (groups == 0 || groups > std::numeric_limits<uint32_t>::max()) {
        return {StatusCode::Unsupported, "Add tensor exceeds the generic dispatch workgroup range"};
    }

    dispatch.label = "add-f32-generic-unverified";
    dispatch.wgsl_source = kAddF32Wgsl;
    dispatch.entry_point = "main";
    dispatch.workgroups_x = static_cast<uint32_t>(groups);
    dispatch.bindings = {
        {0, lhs.buffer(), lhs.byte_offset(), lhs.byte_size()},
        {1, rhs.buffer(), rhs.byte_offset(), rhs.byte_size()},
        {2, output.buffer(), output.byte_offset(), output.byte_size()},
    };
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
        w[0] != x[1] || x[0] == 0 || x[1] == 0 || x[0] > UINT32_MAX || x[1] > UINT32_MAX)
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
    return registry.register_kernel(std::move(rms));
}

} // namespace ai_chat::gpu_engine
