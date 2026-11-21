#include "operators.h"

#include <algorithm>
#include <limits>
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
    return registry.register_kernel(std::move(add));
}

} // namespace ai_chat::gpu_engine
