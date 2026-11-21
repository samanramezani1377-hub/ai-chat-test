#pragma once

#include "model_registry.h"
#include "execution_engine.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace ai_chat::gpu_engine {

// Decodes complete GGUF Q4_0, Q4_1 and Q8_0 blocks.
Status decode_q4_0_blocks(const uint8_t* source, size_t source_bytes,
                          float* destination, size_t destination_elements);
Status decode_q4_1_blocks(const uint8_t* source, size_t source_bytes,
                          float* destination, size_t destination_elements);

// Decodes complete GGUF Q8_0 blocks (32 signed values + one FP16 scale).
// This validates the host decoder only; no GPU Q8_0 MatMul kernel is implied.
Status decode_q8_0_blocks(const uint8_t* source, size_t source_bytes,
                          float* destination, size_t destination_elements);

Status register_builtin_quantization_formats(QuantizationRegistry& registry);
const char* q4_1_dequantize_wgsl_source() noexcept;
Status build_q4_1_dequantize_candidate(const std::shared_ptr<Buffer>& encoded,
                                       uint64_t encoded_bytes,
                                       const Tensor& output,
                                       ComputeDispatch& dispatch);
const char* q4_0_dequantize_wgsl_source() noexcept;
Status build_q4_0_dequantize_candidate(const std::shared_ptr<Buffer>& encoded,
                                       uint64_t encoded_bytes,
                                       const Tensor& output,
                                       ComputeDispatch& dispatch);
const char* q8_0_dequantize_wgsl_source() noexcept;
Status build_q8_0_dequantize_candidate(const std::shared_ptr<Buffer>& encoded,
                                       uint64_t encoded_bytes,
                                       const Tensor& output,
                                       ComputeDispatch& dispatch);

} // namespace ai_chat::gpu_engine
