#include "quantization.h"

#include <cmath>
#include <cstring>
#include <limits>

namespace ai_chat::gpu_engine {
namespace {

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

Status decode_q8_0_blocks(const uint8_t* source, size_t source_bytes,
                          float* destination, size_t destination_elements) {
    constexpr size_t kElementsPerBlock = 32;
    constexpr size_t kBytesPerBlock = 34;
    if ((!source || !destination) && (source_bytes != 0 || destination_elements != 0))
        return {StatusCode::InvalidArgument, "Q8_0 decoder received a null buffer"};
    if (destination_elements == 0 || destination_elements % kElementsPerBlock != 0 ||
        source_bytes != (destination_elements / kElementsPerBlock) * kBytesPerBlock)
        return {StatusCode::InvalidArgument, "Q8_0 source and destination sizes do not match complete blocks"};

    for (size_t block = 0; block < destination_elements / kElementsPerBlock; ++block) {
        const uint8_t* encoded = source + block * kBytesPerBlock;
        const float scale = half_to_float(read_u16_le(encoded));
        if (!std::isfinite(scale))
            return {StatusCode::NumericalFailure, "Q8_0 block has a non-finite FP16 scale"};
        for (size_t i = 0; i < kElementsPerBlock; ++i) {
            const int8_t quantized = static_cast<int8_t>(encoded[2 + i]);
            destination[block * kElementsPerBlock + i] = scale * static_cast<float>(quantized);
        }
    }
    return Status::success();
}

Status register_builtin_quantization_formats(QuantizationRegistry& registry) {
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
