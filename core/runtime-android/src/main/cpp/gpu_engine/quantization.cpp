#include "quantization.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <utility>
#include <string>
#include <vector>

namespace ai_chat::gpu_engine {
namespace {




constexpr char kQ5_0DequantizeWgsl[] = R"wgsl(
@group(0) @binding(0) var<storage, read> encoded: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
const ELEMENT_COUNT: u32 = 32u;
fn read_u8(byte_offset: u32) -> u32 {
    let word = encoded[byte_offset >> 2u];
    let shift = (byte_offset & 3u) * 8u;
    return (word >> shift) & 255u;
}
fn read_u16(byte_offset: u32) -> u32 {
    let word = encoded[byte_offset >> 2u];
    let shift = (byte_offset & 3u) * 8u;
    return (word >> shift) & 65535u;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= ELEMENT_COUNT) { return; }
    let block_index = gid.x / 32u;
    let within = gid.x % 32u;
    let offset = block_index * 22u;
    let scale = unpack2x16float(vec2<u32>(read_u16(offset), 0u)).x;
    let high_bits = read_u8(offset + 2u) | (read_u8(offset + 3u) << 8u) |
                    (read_u8(offset + 4u) << 16u) | (read_u8(offset + 5u) << 24u);
    let packed = read_u8(offset + 6u + (within % 16u));
    let nibble = select(packed & 15u, packed >> 4u, within >= 16u);
    let high = (high_bits >> within) & 1u;
    output[gid.x] = scale * f32(i32(nibble | (high << 4u)) - 16);
}
)wgsl";

constexpr char kQ5_1DequantizeWgsl[] = R"wgsl(
@group(0) @binding(0) var<storage, read> encoded: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
const ELEMENT_COUNT: u32 = 32u;
fn read_u8(byte_offset: u32) -> u32 {
    let word = encoded[byte_offset >> 2u];
    let shift = (byte_offset & 3u) * 8u;
    return (word >> shift) & 255u;
}
fn read_u16(byte_offset: u32) -> u32 {
    let word = encoded[byte_offset >> 2u];
    let shift = (byte_offset & 3u) * 8u;
    return (word >> shift) & 65535u;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= ELEMENT_COUNT) { return; }
    let block_index = gid.x / 32u;
    let within = gid.x % 32u;
    let offset = block_index * 24u;
    let scale = unpack2x16float(vec2<u32>(read_u16(offset), 0u)).x;
    let minimum = unpack2x16float(vec2<u32>(read_u16(offset + 2u), 0u)).x;
    let high_bits = read_u8(offset + 4u) | (read_u8(offset + 5u) << 8u) |
                    (read_u8(offset + 6u) << 16u) | (read_u8(offset + 7u) << 24u);
    let packed = read_u8(offset + 8u + (within % 16u));
    let nibble = select(packed & 15u, packed >> 4u, within >= 16u);
    let high = (high_bits >> within) & 1u;
    output[gid.x] = scale * f32(nibble | (high << 4u)) + minimum;
}
)wgsl";

constexpr char kQ4_1DequantizeWgsl[] = R"wgsl(
@group(0) @binding(0) var<storage, read> encoded: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
const ELEMENT_COUNT: u32 = 32u;
fn read_u8(byte_offset: u32) -> u32 {
    let word = encoded[byte_offset >> 2u];
    let shift = (byte_offset & 3u) * 8u;
    return (word >> shift) & 255u;
}
fn read_u16(byte_offset: u32) -> u32 {
    let word = encoded[byte_offset >> 2u];
    let shift = (byte_offset & 3u) * 8u;
    return (word >> shift) & 65535u;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= ELEMENT_COUNT) { return; }
    let block_index = gid.x / 32u;
    let within_block = gid.x % 32u;
    let block_offset = block_index * 20u;
    let scale = unpack2x16float(vec2<u32>(read_u16(block_offset), 0u)).x;
    let minimum = unpack2x16float(vec2<u32>(read_u16(block_offset + 2u), 0u)).x;
    let packed = read_u8(block_offset + 4u + (within_block % 16u));
    let nibble = select(packed & 15u, packed >> 4u, within_block >= 16u);
    output[gid.x] = scale * f32(nibble) + minimum;
}
)wgsl";

constexpr char kQ4_0DequantizeWgsl[] = R"wgsl(
@group(0) @binding(0) var<storage, read> encoded: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
const ELEMENT_COUNT: u32 = 32u;
fn read_u8(byte_offset: u32) -> u32 {
    let word = encoded[byte_offset >> 2u];
    let shift = (byte_offset & 3u) * 8u;
    return (word >> shift) & 255u;
}
fn read_u16(byte_offset: u32) -> u32 {
    let word = encoded[byte_offset >> 2u];
    let shift = (byte_offset & 3u) * 8u;
    return (word >> shift) & 65535u;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= ELEMENT_COUNT) { return; }
    let block_index = gid.x / 32u;
    let within_block = gid.x % 32u;
    let block_offset = block_index * 18u;
    let scale_bits = read_u16(block_offset);
    let scale = unpack2x16float(vec2<u32>(scale_bits, 0u)).x;
    let packed = read_u8(block_offset + 2u + (within_block % 16u));
    let nibble = select(packed & 15u, packed >> 4u, within_block >= 16u);
    output[gid.x] = scale * f32(i32(nibble) - 8);
}
)wgsl";


constexpr char kQ8_0MatMulWgsl[] = R"wgsl(
@group(0) @binding(0) var<storage, read> activations: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
const ROWS: u32 = 1u;
const OUTPUT_CHANNELS: u32 = 1u;
const INNER: u32 = 32u;
const WEIGHT_ROW_BYTES: u32 = 34u;
fn read_u8(byte_offset: u32) -> u32 {
    let word = weights[byte_offset >> 2u];
    let shift = (byte_offset & 3u) * 8u;
    return (word >> shift) & 255u;
}
fn read_u16(byte_offset: u32) -> u32 {
    let word = weights[byte_offset >> 2u];
    let shift = (byte_offset & 3u) * 8u;
    return (word >> shift) & 65535u;
}
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let channel = gid.x;
    if (row >= ROWS || channel >= OUTPUT_CHANNELS) { return; }
    var sum = 0.0;
    for (var k = 0u; k < INNER; k = k + 1u) {
        let block = k / 32u;
        let within = k % 32u;
        let block_offset = channel * WEIGHT_ROW_BYTES + block * 34u;
        let scale = unpack2x16float(vec2<u32>(read_u16(block_offset), 0u)).x;
        let raw = read_u8(block_offset + 2u + within);
        let quantized = select(i32(raw), i32(raw) - 256, raw >= 128u);
        sum = sum + activations[row * INNER + k] * scale * f32(quantized);
    }
    output[row * OUTPUT_CHANNELS + channel] = sum;
}
)wgsl";

constexpr char kQ8_0DequantizeWgsl[] = R"wgsl(
@group(0) @binding(0) var<storage, read> encoded: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
const ELEMENT_COUNT: u32 = 32u;
fn read_u8(byte_offset: u32) -> u32 {
    let word = encoded[byte_offset >> 2u];
    let shift = (byte_offset & 3u) * 8u;
    return (word >> shift) & 255u;
}
fn read_u16(byte_offset: u32) -> u32 {
    let word = encoded[byte_offset >> 2u];
    let shift = (byte_offset & 3u) * 8u;
    return (word >> shift) & 65535u;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= ELEMENT_COUNT) { return; }
    let block_index = gid.x / 32u;
    let within_block = gid.x % 32u;
    let block_offset = block_index * 34u;
    let scale_bits = read_u16(block_offset);
    let scale = unpack2x16float(vec2<u32>(scale_bits, 0u)).x;
    let raw = read_u8(block_offset + 2u + within_block);
    let signed_value = select(i32(raw), i32(raw) - 256, raw >= 128u);
    output[gid.x] = scale * f32(signed_value);
}
)wgsl";



bool is_device_contiguous_f32(const Tensor& tensor) {
    const TensorDesc& desc = tensor.descriptor();
    return desc.data_type == DataType::F32 &&
           desc.layout == TensorLayout::Contiguous &&
           desc.strides_bytes.empty() &&
           desc.memory != MemoryDomain::Host;
}

float half_to_float(uint16_t half) {
    const uint32_t sign = static_cast<uint32_t>(half & 0x8000u) << 16;
    uint32_t exponent = (half >> 10) & 0x1fu;
    uint32_t mantissa = half & 0x03ffu;
    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int shift = 0;
            while ((mantissa & 0x0400u) == 0) { mantissa <<= 1; ++shift; }
            mantissa &= 0x03ffu;
            const uint32_t float_exponent = static_cast<uint32_t>(127 - 15 - shift + 1);
            bits = sign | (float_exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1fu) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        exponent = exponent + (127 - 15);
        bits = sign | (exponent << 23) | (mantissa << 13);
    }
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

uint16_t read_u16_le(const uint8_t* source) {
    return static_cast<uint16_t>(source[0]) |
           static_cast<uint16_t>(static_cast<uint16_t>(source[1]) << 8);
}

} // namespace



Status decode_q5_0_blocks(const uint8_t* source, size_t source_bytes,
                          float* destination, size_t destination_elements) {
    constexpr size_t kElementsPerBlock = 32;
    constexpr size_t kBytesPerBlock = 22;
    if ((!source || !destination) && (source_bytes != 0 || destination_elements != 0))
        return {StatusCode::InvalidArgument, "Q5_0 decoder received a null buffer"};
    if (destination_elements == 0 || destination_elements % kElementsPerBlock != 0)
        return {StatusCode::InvalidArgument, "Q5_0 destination must contain complete blocks"};
    const size_t blocks = destination_elements / kElementsPerBlock;
    if (blocks > std::numeric_limits<size_t>::max() / kBytesPerBlock ||
        source_bytes != blocks * kBytesPerBlock)
        return {StatusCode::InvalidArgument, "Q5_0 source and destination sizes do not match complete blocks"};
    for (size_t block = 0; block < blocks; ++block) {
        const uint8_t* encoded = source + block * kBytesPerBlock;
        const float scale = half_to_float(read_u16_le(encoded));
        if (!std::isfinite(scale))
            return {StatusCode::NumericalFailure, "Q5_0 block has a non-finite FP16 scale"};
        const uint32_t high_bits = static_cast<uint32_t>(encoded[2]) |
            (static_cast<uint32_t>(encoded[3]) << 8) |
            (static_cast<uint32_t>(encoded[4]) << 16) |
            (static_cast<uint32_t>(encoded[5]) << 24);
        for (size_t i = 0; i < 16; ++i) {
            const uint8_t packed = encoded[6 + i];
            const int32_t low = static_cast<int32_t>(packed & 0x0fu) |
                                static_cast<int32_t>(((high_bits >> i) & 1u) << 4);
            const int32_t high = static_cast<int32_t>(packed >> 4) |
                                 static_cast<int32_t>(((high_bits >> (i + 16)) & 1u) << 4);
            destination[block * 32 + i] = scale * static_cast<float>(low - 16);
            destination[block * 32 + i + 16] = scale * static_cast<float>(high - 16);
        }
    }
    return Status::success();
}

Status decode_q5_1_blocks(const uint8_t* source, size_t source_bytes,
                          float* destination, size_t destination_elements) {
    constexpr size_t kElementsPerBlock = 32;
    constexpr size_t kBytesPerBlock = 24;
    if ((!source || !destination) && (source_bytes != 0 || destination_elements != 0))
        return {StatusCode::InvalidArgument, "Q5_1 decoder received a null buffer"};
    if (destination_elements == 0 || destination_elements % kElementsPerBlock != 0)
        return {StatusCode::InvalidArgument, "Q5_1 destination must contain complete blocks"};
    const size_t blocks = destination_elements / kElementsPerBlock;
    if (blocks > std::numeric_limits<size_t>::max() / kBytesPerBlock ||
        source_bytes != blocks * kBytesPerBlock)
        return {StatusCode::InvalidArgument, "Q5_1 source and destination sizes do not match complete blocks"};
    for (size_t block = 0; block < blocks; ++block) {
        const uint8_t* encoded = source + block * kBytesPerBlock;
        const float scale = half_to_float(read_u16_le(encoded));
        const float minimum = half_to_float(read_u16_le(encoded + 2));
        if (!std::isfinite(scale) || !std::isfinite(minimum))
            return {StatusCode::NumericalFailure, "Q5_1 block has a non-finite FP16 scale or minimum"};
        const uint32_t high_bits = static_cast<uint32_t>(encoded[4]) |
            (static_cast<uint32_t>(encoded[5]) << 8) |
            (static_cast<uint32_t>(encoded[6]) << 16) |
            (static_cast<uint32_t>(encoded[7]) << 24);
        for (size_t i = 0; i < 16; ++i) {
            const uint8_t packed = encoded[8 + i];
            const uint32_t low = static_cast<uint32_t>(packed & 0x0fu) |
                                 (((high_bits >> i) & 1u) << 4);
            const uint32_t high = static_cast<uint32_t>(packed >> 4) |
                                  (((high_bits >> (i + 16)) & 1u) << 4);
            destination[block * 32 + i] = scale * static_cast<float>(low) + minimum;
            destination[block * 32 + i + 16] = scale * static_cast<float>(high) + minimum;
        }
    }
    return Status::success();
}

Status decode_q4_0_blocks(const uint8_t* source, size_t source_bytes,
                          float* destination, size_t destination_elements) {
    constexpr size_t kElementsPerBlock = 32;
    constexpr size_t kBytesPerBlock = 18;
    if ((!source || !destination) && (source_bytes != 0 || destination_elements != 0))
        return {StatusCode::InvalidArgument, "Q4_0 decoder received a null buffer"};
    if (destination_elements == 0 || destination_elements % kElementsPerBlock != 0)
        return {StatusCode::InvalidArgument, "Q4_0 destination must contain complete blocks"};
    const size_t blocks = destination_elements / kElementsPerBlock;
    if (blocks > std::numeric_limits<size_t>::max() / kBytesPerBlock ||
        source_bytes != blocks * kBytesPerBlock)
        return {StatusCode::InvalidArgument, "Q4_0 source and destination sizes do not match complete blocks"};
    for (size_t block = 0; block < blocks; ++block) {
        const uint8_t* encoded = source + block * kBytesPerBlock;
        const float scale = half_to_float(read_u16_le(encoded));
        if (!std::isfinite(scale))
            return {StatusCode::NumericalFailure, "Q4_0 block has a non-finite FP16 scale"};
        for (size_t i = 0; i < 16; ++i) {
            const uint8_t packed = encoded[2 + i];
            destination[block * 32 + i] =
                scale * static_cast<float>(static_cast<int32_t>(packed & 0x0fu) - 8);
            destination[block * 32 + i + 16] =
                scale * static_cast<float>(static_cast<int32_t>(packed >> 4) - 8);
        }
    }
    return Status::success();
}

Status decode_q4_1_blocks(const uint8_t* source, size_t source_bytes,
                          float* destination, size_t destination_elements) {
    constexpr size_t kElementsPerBlock = 32;
    constexpr size_t kBytesPerBlock = 20;
    if ((!source || !destination) && (source_bytes != 0 || destination_elements != 0))
        return {StatusCode::InvalidArgument, "Q4_1 decoder received a null buffer"};
    if (destination_elements == 0 || destination_elements % kElementsPerBlock != 0)
        return {StatusCode::InvalidArgument, "Q4_1 destination must contain complete blocks"};
    const size_t blocks = destination_elements / kElementsPerBlock;
    if (blocks > std::numeric_limits<size_t>::max() / kBytesPerBlock ||
        source_bytes != blocks * kBytesPerBlock)
        return {StatusCode::InvalidArgument, "Q4_1 source and destination sizes do not match complete blocks"};
    for (size_t block = 0; block < blocks; ++block) {
        const uint8_t* encoded = source + block * kBytesPerBlock;
        const float scale = half_to_float(read_u16_le(encoded));
        const float minimum = half_to_float(read_u16_le(encoded + 2));
        if (!std::isfinite(scale) || !std::isfinite(minimum))
            return {StatusCode::NumericalFailure, "Q4_1 block has a non-finite FP16 scale or minimum"};
        for (size_t i = 0; i < 16; ++i) {
            const uint8_t packed = encoded[4 + i];
            destination[block * 32 + i] =
                scale * static_cast<float>(packed & 0x0fu) + minimum;
            destination[block * 32 + i + 16] =
                scale * static_cast<float>(packed >> 4) + minimum;
        }
    }
    return Status::success();
}

Status decode_q8_0_blocks(const uint8_t* source, size_t source_bytes,
                          float* destination, size_t destination_elements) {
    constexpr size_t kElementsPerBlock = 32;
    constexpr size_t kBytesPerBlock = 34;
    if ((!source || !destination) && (source_bytes != 0 || destination_elements != 0))
        return {StatusCode::InvalidArgument, "Q8_0 decoder received a null buffer"};
    if (destination_elements == 0 || destination_elements % kElementsPerBlock != 0)
        return {StatusCode::InvalidArgument, "Q8_0 destination size must contain complete blocks"};
    const size_t block_count = destination_elements / kElementsPerBlock;
    if (block_count > std::numeric_limits<size_t>::max() / kBytesPerBlock ||
        source_bytes != block_count * kBytesPerBlock)
        return {StatusCode::InvalidArgument, "Q8_0 source and destination sizes do not match complete blocks"};

    for (size_t block = 0; block < block_count; ++block) {
        const uint8_t* encoded = source + block * kBytesPerBlock;
        const float scale = half_to_float(read_u16_le(encoded));
        if (!std::isfinite(scale))
            return {StatusCode::NumericalFailure, "Q8_0 block has a non-finite FP16 scale"};
        for (size_t i = 0; i < kElementsPerBlock; ++i) {
            const uint8_t raw = encoded[2 + i];
            const int32_t quantized = raw < 128u ? static_cast<int32_t>(raw)
                                                  : static_cast<int32_t>(raw) - 256;
            destination[block * kElementsPerBlock + i] = scale * static_cast<float>(quantized);
        }
    }
    return Status::success();
}






Status build_q8_0_matmul_candidate(const Tensor& activations,
                                  const std::shared_ptr<Buffer>& quantized_weights,
                                  uint64_t weight_bytes,
                                  const Tensor& output,
                                  ComputeDispatch& dispatch) {
    dispatch = ComputeDispatch{};
    if (!quantized_weights || !is_device_contiguous_f32(activations) ||
        !is_device_contiguous_f32(output) || !activations.buffer() || !output.buffer() || activations.descriptor().shape.size() != 2 ||
        output.descriptor().shape.size() != 2)
        return {StatusCode::Unsupported, "Q8_0 MatMul requires F32 [M,K] activations and F32 [M,N] output"};
    const auto& a = activations.descriptor().shape;
    const auto& o = output.descriptor().shape;
    if (a[0] == 0 || a[1] == 0 || a[1] % 32 != 0 ||
        o[0] != a[0] || o[1] == 0 || a[0] > UINT32_MAX ||
        a[1] > UINT32_MAX || o[1] > UINT32_MAX ||
        activations.element_count() > UINT32_MAX || output.element_count() > UINT32_MAX)
        return {StatusCode::InvalidArgument, "Q8_0 MatMul dimensions are invalid or inner dimension is not block-aligned"};
    const uint64_t blocks_per_row = a[1] / 32;
    if (blocks_per_row > UINT64_MAX / 34 ||
        o[1] > UINT64_MAX / (blocks_per_row * 34))
        return {StatusCode::InvalidArgument, "Q8_0 weight byte size overflows"};
    const uint64_t expected_bytes = o[1] * blocks_per_row * 34;
    if (weight_bytes != expected_bytes || weight_bytes > static_cast<uint64_t>(UINT32_MAX) - 3)
        return {StatusCode::InvalidArgument, "Q8_0 weight byte count does not match [N,K] shape"};
    const uint64_t padded_bytes = (weight_bytes + 3) & ~uint64_t{3};
    if (quantized_weights->size_bytes() < padded_bytes ||
        output.byte_size() != output.element_count() * sizeof(float) ||
        output.buffer().get() == activations.buffer().get() ||
        output.buffer().get() == quantized_weights.get())
        return {StatusCode::InvalidArgument, "Q8_0 MatMul buffer ranges are inconsistent"};
    dispatch.label = "q8-0-matmul-f32-activation-generic-unverified";
    dispatch.kernel_id = "dawn-q8-0-matmul-f32-activation-v1";
    dispatch.backend = BackendKind::DawnWebGpu;
    dispatch.wgsl_source = kQ8_0MatMulWgsl;
    const std::pair<std::string, std::string> constants[] = {
        {"const ROWS: u32 = 1u;", "const ROWS: u32 = " + std::to_string(a[0]) + "u;"},
        {"const OUTPUT_CHANNELS: u32 = 1u;", "const OUTPUT_CHANNELS: u32 = " + std::to_string(o[1]) + "u;"},
        {"const INNER: u32 = 32u;", "const INNER: u32 = " + std::to_string(a[1]) + "u;"},
        {"const WEIGHT_ROW_BYTES: u32 = 34u;", "const WEIGHT_ROW_BYTES: u32 = " +
            std::to_string(blocks_per_row * 34) + "u;"},
    };
    for (const auto& constant : constants) {
        const size_t position = dispatch.wgsl_source.find(constant.first);
        if (position == std::string::npos)
            return {StatusCode::BackendFailure, "Q8_0 MatMul WGSL placeholder is missing"};
        dispatch.wgsl_source.replace(position, constant.first.size(), constant.second);
    }
    dispatch.workgroups_x = static_cast<uint32_t>((o[1] + 7) / 8);
    dispatch.workgroups_y = static_cast<uint32_t>((a[0] + 7) / 8);
    dispatch.bindings = {
        {0, activations.buffer(), activations.byte_offset(), activations.byte_size()},
        {1, quantized_weights, 0, padded_bytes},
        {2, output.buffer(), output.byte_offset(), output.byte_size()},
    };
    return Status::success();
}

Status reference_q8_0_matmul(const float* activations, const uint8_t* quantized_weights,
                             uint64_t weight_bytes, float* output,
                             uint32_t rows, uint32_t output_channels, uint32_t inner) {
    if ((!activations || !quantized_weights || !output) && rows && output_channels && inner)
        return {StatusCode::InvalidArgument, "Q8_0 MatMul reference received a null pointer"};
    if (rows == 0 || output_channels == 0 || inner == 0 || inner % 32 != 0)
        return {StatusCode::InvalidArgument, "Q8_0 MatMul dimensions must be non-zero and inner block-aligned"};
    const uint64_t blocks_per_row = inner / 32;
    if (blocks_per_row > UINT64_MAX / 34 ||
        output_channels > UINT64_MAX / (blocks_per_row * 34) ||
        weight_bytes != static_cast<uint64_t>(output_channels) * blocks_per_row * 34)
        return {StatusCode::InvalidArgument, "Q8_0 MatMul weight byte size is inconsistent"};
    std::vector<float> decoded(inner);
    for (uint32_t channel = 0; channel < output_channels; ++channel) {
        const uint8_t* row_weights = quantized_weights + static_cast<uint64_t>(channel) * blocks_per_row * 34;
        Status status = decode_q8_0_blocks(row_weights, blocks_per_row * 34, decoded.data(), inner);
        if (!status.ok()) return status;
        for (uint32_t row = 0; row < rows; ++row) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < inner; ++k)
                sum += activations[static_cast<uint64_t>(row) * inner + k] * decoded[k];
            output[static_cast<uint64_t>(row) * output_channels + channel] = sum;
        }
    }
    return Status::success();
}

const char* q5_0_dequantize_wgsl_source() noexcept { return kQ5_0DequantizeWgsl; }

Status build_q5_0_dequantize_candidate(const std::shared_ptr<Buffer>& encoded,
                                       uint64_t encoded_bytes,
                                       const Tensor& output,
                                       ComputeDispatch& dispatch) {
    dispatch = ComputeDispatch{};
    if (!encoded || !output.buffer() || output.descriptor().data_type != DataType::F32 ||
        output.descriptor().layout != TensorLayout::Contiguous ||
        output.descriptor().memory == MemoryDomain::Host || output.descriptor().shape.empty())
        return {StatusCode::Unsupported, "Q5_0 GPU decoder requires a device/shared encoded buffer and contiguous F32 output"};
    constexpr uint64_t block_elements = 32, block_bytes = 22;
    if (output.element_count() == 0 || output.element_count() % block_elements != 0 ||
        output.element_count() > UINT32_MAX)
        return {StatusCode::InvalidArgument, "Q5_0 output must contain complete blocks and fit WGSL indexing"};
    const uint64_t blocks = output.element_count() / block_elements;
    if (blocks > UINT64_MAX / block_bytes || encoded_bytes != blocks * block_bytes ||
        encoded_bytes > static_cast<uint64_t>(UINT32_MAX) - 3)
        return {StatusCode::InvalidArgument, "Q5_0 encoded size does not match output elements"};
    const uint64_t padded = (encoded_bytes + 3) & ~uint64_t{3};
    if (encoded->size_bytes() < padded || output.byte_size() != output.element_count() * sizeof(float) ||
        encoded.get() == output.buffer().get())
        return {StatusCode::InvalidArgument, "Q5_0 input/output buffer ranges are inconsistent"};
    dispatch.label = "q5-0-dequantize-generic-unverified";
    dispatch.kernel_id = "dawn-q5-0-dequantize-generic-v1";
    dispatch.backend = BackendKind::DawnWebGpu;
    dispatch.wgsl_source = kQ5_0DequantizeWgsl;
    const std::string placeholder = "const ELEMENT_COUNT: u32 = 32u;";
    const std::string replacement = "const ELEMENT_COUNT: u32 = " + std::to_string(output.element_count()) + "u;";
    const size_t position = dispatch.wgsl_source.find(placeholder);
    if (position == std::string::npos) return {StatusCode::BackendFailure, "Q5_0 WGSL placeholder missing"};
    dispatch.wgsl_source.replace(position, placeholder.size(), replacement);
    dispatch.workgroups_x = static_cast<uint32_t>((output.element_count() + 63) / 64);
    dispatch.bindings = {{0, encoded, 0, padded}, {1, output.buffer(), output.byte_offset(), output.byte_size()}};
    return Status::success();
}

const char* q5_1_dequantize_wgsl_source() noexcept { return kQ5_1DequantizeWgsl; }

Status build_q5_1_dequantize_candidate(const std::shared_ptr<Buffer>& encoded,
                                       uint64_t encoded_bytes,
                                       const Tensor& output,
                                       ComputeDispatch& dispatch) {
    dispatch = ComputeDispatch{};
    if (!encoded || !output.buffer() || output.descriptor().data_type != DataType::F32 ||
        output.descriptor().layout != TensorLayout::Contiguous ||
        output.descriptor().memory == MemoryDomain::Host || output.descriptor().shape.empty())
        return {StatusCode::Unsupported, "Q5_1 GPU decoder requires a device/shared encoded buffer and contiguous F32 output"};
    constexpr uint64_t block_elements = 32, block_bytes = 24;
    if (output.element_count() == 0 || output.element_count() % block_elements != 0 ||
        output.element_count() > UINT32_MAX)
        return {StatusCode::InvalidArgument, "Q5_1 output must contain complete blocks and fit WGSL indexing"};
    const uint64_t blocks = output.element_count() / block_elements;
    if (blocks > UINT64_MAX / block_bytes || encoded_bytes != blocks * block_bytes ||
        encoded_bytes > static_cast<uint64_t>(UINT32_MAX) - 3)
        return {StatusCode::InvalidArgument, "Q5_1 encoded size does not match output elements"};
    const uint64_t padded = (encoded_bytes + 3) & ~uint64_t{3};
    if (encoded->size_bytes() < padded || output.byte_size() != output.element_count() * sizeof(float) ||
        encoded.get() == output.buffer().get())
        return {StatusCode::InvalidArgument, "Q5_1 input/output buffer ranges are inconsistent"};
    dispatch.label = "q5-1-dequantize-generic-unverified";
    dispatch.kernel_id = "dawn-q5-1-dequantize-generic-v1";
    dispatch.backend = BackendKind::DawnWebGpu;
    dispatch.wgsl_source = kQ5_1DequantizeWgsl;
    const std::string placeholder = "const ELEMENT_COUNT: u32 = 32u;";
    const std::string replacement = "const ELEMENT_COUNT: u32 = " + std::to_string(output.element_count()) + "u;";
    const size_t position = dispatch.wgsl_source.find(placeholder);
    if (position == std::string::npos) return {StatusCode::BackendFailure, "Q5_1 WGSL placeholder missing"};
    dispatch.wgsl_source.replace(position, placeholder.size(), replacement);
    dispatch.workgroups_x = static_cast<uint32_t>((output.element_count() + 63) / 64);
    dispatch.bindings = {{0, encoded, 0, padded}, {1, output.buffer(), output.byte_offset(), output.byte_size()}};
    return Status::success();
}

const char* q4_1_dequantize_wgsl_source() noexcept { return kQ4_1DequantizeWgsl; }

Status build_q4_1_dequantize_candidate(const std::shared_ptr<Buffer>& encoded,
                                       uint64_t encoded_bytes,
                                       const Tensor& output,
                                       ComputeDispatch& dispatch) {
    dispatch = ComputeDispatch{};
    if (!encoded || !output.buffer() || output.descriptor().data_type != DataType::F32 ||
        output.descriptor().layout != TensorLayout::Contiguous ||
        output.descriptor().memory == MemoryDomain::Host || output.descriptor().shape.empty())
        return {StatusCode::Unsupported, "Q4_1 GPU decoder requires a device/shared encoded buffer and contiguous F32 output"};
    constexpr uint64_t kBlockElements = 32;
    constexpr uint64_t kBlockBytes = 20;
    if (output.element_count() == 0 || output.element_count() % kBlockElements != 0 ||
        output.element_count() > UINT32_MAX)
        return {StatusCode::InvalidArgument, "Q4_1 output must contain complete blocks and fit the WGSL index range"};
    const uint64_t blocks = output.element_count() / kBlockElements;
    if (blocks > UINT64_MAX / kBlockBytes || encoded_bytes != blocks * kBlockBytes ||
        encoded_bytes > static_cast<uint64_t>(UINT32_MAX) - 3)
        return {StatusCode::InvalidArgument, "Q4_1 encoded byte count does not match the output tensor"};
    const uint64_t padded_bytes = (encoded_bytes + 3) & ~uint64_t{3};
    if (encoded->size_bytes() < padded_bytes ||
        output.byte_size() != output.element_count() * sizeof(float) ||
        encoded.get() == output.buffer().get())
        return {StatusCode::InvalidArgument, "Q4_1 input/output buffer ranges are inconsistent"};
    dispatch.label = "q4-1-dequantize-generic-unverified";
    dispatch.kernel_id = "dawn-q4-1-dequantize-generic-v1";
    dispatch.backend = BackendKind::DawnWebGpu;
    dispatch.wgsl_source = kQ4_1DequantizeWgsl;
    const std::string placeholder = "const ELEMENT_COUNT: u32 = 32u;";
    const std::string replacement = "const ELEMENT_COUNT: u32 = " +
                                    std::to_string(output.element_count()) + "u;";
    const size_t position = dispatch.wgsl_source.find(placeholder);
    if (position == std::string::npos)
        return {StatusCode::BackendFailure, "Q4_1 WGSL element-count placeholder is missing"};
    dispatch.wgsl_source.replace(position, placeholder.size(), replacement);
    dispatch.workgroups_x = static_cast<uint32_t>((output.element_count() + 63) / 64);
    dispatch.bindings = {
        {0, encoded, 0, padded_bytes},
        {1, output.buffer(), output.byte_offset(), output.byte_size()},
    };
    return Status::success();
}

const char* q4_0_dequantize_wgsl_source() noexcept { return kQ4_0DequantizeWgsl; }

Status build_q4_0_dequantize_candidate(const std::shared_ptr<Buffer>& encoded,
                                       uint64_t encoded_bytes,
                                       const Tensor& output,
                                       ComputeDispatch& dispatch) {
    dispatch = ComputeDispatch{};
    if (!encoded || !output.buffer() || output.descriptor().data_type != DataType::F32 ||
        output.descriptor().layout != TensorLayout::Contiguous ||
        output.descriptor().memory == MemoryDomain::Host || output.descriptor().shape.empty())
        return {StatusCode::Unsupported, "Q4_0 GPU decoder requires a device/shared encoded buffer and contiguous F32 output"};
    constexpr uint64_t kBlockElements = 32;
    constexpr uint64_t kBlockBytes = 18;
    if (output.element_count() == 0 || output.element_count() % kBlockElements != 0 ||
        output.element_count() > UINT32_MAX)
        return {StatusCode::InvalidArgument, "Q4_0 output must contain complete blocks and fit the WGSL index range"};
    const uint64_t blocks = output.element_count() / kBlockElements;
    if (blocks > UINT64_MAX / kBlockBytes || encoded_bytes != blocks * kBlockBytes ||
        encoded_bytes > static_cast<uint64_t>(UINT32_MAX) - 3)
        return {StatusCode::InvalidArgument, "Q4_0 encoded byte count does not match the output tensor"};
    const uint64_t padded_bytes = (encoded_bytes + 3) & ~uint64_t{3};
    if (encoded->size_bytes() < padded_bytes ||
        output.byte_size() != output.element_count() * sizeof(float) ||
        encoded.get() == output.buffer().get())
        return {StatusCode::InvalidArgument, "Q4_0 input/output buffer ranges are inconsistent"};
    dispatch.label = "q4-0-dequantize-generic-unverified";
    dispatch.kernel_id = "dawn-q4-0-dequantize-generic-v1";
    dispatch.backend = BackendKind::DawnWebGpu;
    dispatch.wgsl_source = kQ4_0DequantizeWgsl;
    const std::string placeholder = "const ELEMENT_COUNT: u32 = 32u;";
    const std::string replacement = "const ELEMENT_COUNT: u32 = " +
                                    std::to_string(output.element_count()) + "u;";
    const size_t position = dispatch.wgsl_source.find(placeholder);
    if (position == std::string::npos)
        return {StatusCode::BackendFailure, "Q4_0 WGSL element-count placeholder is missing"};
    dispatch.wgsl_source.replace(position, placeholder.size(), replacement);
    dispatch.workgroups_x = static_cast<uint32_t>((output.element_count() + 63) / 64);
    dispatch.bindings = {
        {0, encoded, 0, padded_bytes},
        {1, output.buffer(), output.byte_offset(), output.byte_size()},
    };
    return Status::success();
}

const char* q8_0_dequantize_wgsl_source() noexcept { return kQ8_0DequantizeWgsl; }

Status build_q8_0_dequantize_candidate(const std::shared_ptr<Buffer>& encoded,
                                       uint64_t encoded_bytes,
                                       const Tensor& output,
                                       ComputeDispatch& dispatch) {
    dispatch = ComputeDispatch{};
    if (!encoded || !output.buffer() || output.descriptor().data_type != DataType::F32 ||
        output.descriptor().layout != TensorLayout::Contiguous ||
        output.descriptor().memory == MemoryDomain::Host || output.descriptor().shape.empty()) {
        return {StatusCode::Unsupported, "Q8_0 GPU decoder requires a device/shared encoded buffer and contiguous F32 output"};
    }
    constexpr uint64_t kBlockElements = 32;
    constexpr uint64_t kBlockBytes = 34;
    if (output.element_count() == 0 || output.element_count() % kBlockElements != 0 ||
        output.element_count() > UINT32_MAX) {
        return {StatusCode::InvalidArgument, "Q8_0 output must contain complete blocks and fit the WGSL index range"};
    }
    const uint64_t blocks = output.element_count() / kBlockElements;
    if (blocks > UINT64_MAX / kBlockBytes || encoded_bytes != blocks * kBlockBytes ||
        encoded_bytes > static_cast<uint64_t>(UINT32_MAX) - 3) {
        return {StatusCode::InvalidArgument, "Q8_0 encoded byte count does not match the output tensor"};
    }
    const uint64_t padded_bytes = (encoded_bytes + 3) & ~uint64_t{3};
    if (encoded->size_bytes() < padded_bytes ||
        output.byte_size() != output.element_count() * sizeof(float) ||
        encoded.get() == output.buffer().get()) {
        return {StatusCode::InvalidArgument, "Q8_0 input/output buffer ranges are inconsistent"};
    }
    dispatch.label = "q8-0-dequantize-generic-unverified";
    dispatch.kernel_id = "dawn-q8-0-dequantize-generic-v1";
    dispatch.backend = BackendKind::DawnWebGpu;
    dispatch.wgsl_source = kQ8_0DequantizeWgsl;
    const std::string placeholder = "const ELEMENT_COUNT: u32 = 32u;";
    const std::string replacement = "const ELEMENT_COUNT: u32 = " +
                                    std::to_string(output.element_count()) + "u;";
    const size_t position = dispatch.wgsl_source.find(placeholder);
    if (position == std::string::npos)
        return {StatusCode::BackendFailure, "Q8_0 WGSL element-count placeholder is missing"};
    dispatch.wgsl_source.replace(position, placeholder.size(), replacement);
    dispatch.workgroups_x = static_cast<uint32_t>((output.element_count() + 63) / 64);
    dispatch.bindings = {
        {0, encoded, 0, padded_bytes},
        {1, output.buffer(), output.byte_offset(), output.byte_size()},
    };
    return Status::success();
}

Status register_builtin_quantization_formats(QuantizationRegistry& registry) {
    QuantizationFormat q4_0;
    q4_0.file_type_id = 2;
    q4_0.name = "Q4_0";
    q4_0.block_elements = 32;
    q4_0.bytes_per_block = 18;
    q4_0.quantized = true;
    q4_0.decoder_state = SupportState::Validated;
    q4_0.gpu_kernel_state = SupportState::Unverified;
    Status status = registry.register_format(std::move(q4_0));
    if (!status.ok()) return status;

    QuantizationFormat q4_1;
    q4_1.file_type_id = 3;
    q4_1.name = "Q4_1";
    q4_1.block_elements = 32;
    q4_1.bytes_per_block = 20;
    q4_1.quantized = true;
    q4_1.decoder_state = SupportState::Validated;
    q4_1.gpu_kernel_state = SupportState::Unverified;
    status = registry.register_format(std::move(q4_1));
    if (!status.ok()) return status;

    QuantizationFormat q5_0;
    q5_0.file_type_id = 6;
    q5_0.name = "Q5_0";
    q5_0.block_elements = 32;
    q5_0.bytes_per_block = 22;
    q5_0.quantized = true;
    q5_0.decoder_state = SupportState::Validated;
    q5_0.gpu_kernel_state = SupportState::Unverified;
    status = registry.register_format(std::move(q5_0));
    if (!status.ok()) return status;

    QuantizationFormat q5_1;
    q5_1.file_type_id = 7;
    q5_1.name = "Q5_1";
    q5_1.block_elements = 32;
    q5_1.bytes_per_block = 24;
    q5_1.quantized = true;
    q5_1.decoder_state = SupportState::Validated;
    q5_1.gpu_kernel_state = SupportState::Unverified;
    status = registry.register_format(std::move(q5_1));
    if (!status.ok()) return status;

    QuantizationFormat q8;
    q8.file_type_id = 8;
    q8.name = "Q8_0";
    q8.block_elements = 32;
    q8.bytes_per_block = 34;
    q8.quantized = true;
    q8.decoder_state = SupportState::Validated;
    q8.gpu_kernel_state = SupportState::Unverified;
    return registry.register_format(std::move(q8));
}

} // namespace ai_chat::gpu_engine
