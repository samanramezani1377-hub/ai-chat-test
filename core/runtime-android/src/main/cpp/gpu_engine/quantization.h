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
Status decode_q5_0_blocks(const uint8_t* source, size_t source_bytes,
                          float* destination, size_t destination_elements);
Status decode_q5_1_blocks(const uint8_t* source, size_t source_bytes,
                          float* destination, size_t destination_elements);
Status decode_q4_1_blocks(const uint8_t* source, size_t source_bytes,
                          float* destination, size_t destination_elements);

// Decodes complete GGUF Q8_0 blocks (32 signed values + one FP16 scale).
// Host decoder validation is independent; WGSL/GLES Q8_0 MatMul candidates remain UNVERIFIED.
Status decode_q8_0_blocks(const uint8_t* source, size_t source_bytes,
                          float* destination, size_t destination_elements);
const char* q8_0_matmul_wgsl_source() noexcept;
const char* q4_0_matmul_wgsl_source() noexcept;
const char* q4_0_matmul_glsl_source() noexcept;
Status build_q4_0_matmul_candidate(const Tensor& activations, const std::shared_ptr<Buffer>& quantized_weights, uint64_t weight_bytes, const Tensor& output, ComputeDispatch& dispatch);
Status build_q4_0_matmul_gles_candidate(const Tensor& activations, const std::shared_ptr<Buffer>& quantized_weights, uint64_t weight_bytes, const Tensor& output, ComputeDispatch& dispatch);
Status reference_q4_0_matmul(const float* activations, const uint8_t* quantized_weights, uint64_t weight_bytes, float* output, uint32_t rows, uint32_t output_channels, uint32_t inner);
Status validate_q4_0_matmul_kernel(Backend& backend, OperatorRegistry& operators, QuantizationRegistry& quantizations);
// Candidate only: remains UNVERIFIED until GLES numerical parity tests pass.
const char* q8_0_matmul_glsl_source() noexcept;
Status validate_q8_0_matmul_kernel(Backend& backend, OperatorRegistry& operators, QuantizationRegistry& quantizations);
Status build_q8_0_matmul_gles_candidate(const Tensor& activations,
                                       const std::shared_ptr<Buffer>& quantized_weights,
                                       uint64_t weight_bytes, const Tensor& output,
                                       ComputeDispatch& dispatch);
Status build_q8_0_matmul_candidate(const Tensor& activations,
                                  const std::shared_ptr<Buffer>& quantized_weights,
                                  uint64_t weight_bytes,
                                  const Tensor& output,
                                  ComputeDispatch& dispatch);
Status reference_q8_0_matmul(const float* activations, const uint8_t* quantized_weights,
                             uint64_t weight_bytes, float* output,
                             uint32_t rows, uint32_t output_channels, uint32_t inner);


struct ModelQuantizationBackendReport {
    bool compatible = false;
    BackendKind backend = BackendKind::Unknown;
    std::vector<uint32_t> unsupported_file_type_ids;
    std::vector<std::string> blockers;
};

ModelQuantizationBackendReport evaluate_model_quantization_backend(
    const GgufFileInfo& model, const QuantizationRegistry& registry, BackendKind backend);

Status register_builtin_quantization_formats(QuantizationRegistry& registry);
Status register_builtin_quantized_operator_candidates(OperatorRegistry& registry);
const char* q5_0_dequantize_wgsl_source() noexcept;
Status build_q5_0_dequantize_candidate(const std::shared_ptr<Buffer>& encoded,
                                       uint64_t encoded_bytes,
                                       const Tensor& output,
                                       ComputeDispatch& dispatch);
const char* q5_1_dequantize_wgsl_source() noexcept;
Status build_q5_1_dequantize_candidate(const std::shared_ptr<Buffer>& encoded,
                                       uint64_t encoded_bytes,
                                       const Tensor& output,
                                       ComputeDispatch& dispatch);
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
