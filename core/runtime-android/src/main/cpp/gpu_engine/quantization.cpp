#include "quantization.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <utility>
#include <string>

namespace ai_chat::gpu_engine {
namespace {


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
