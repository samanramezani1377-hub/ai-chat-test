#pragma once

#include "model_registry.h"
#include "execution_engine.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace ai_chat::gpu_engine {

// Decodes complete GGUF Q4_0, Q4_1, Q5_0, Q5_1 and Q8_0 blocks.
Status decode_q4_0_blocks(const uint8_t* source, size_t source_bytes,
                          float* destination, size_t destination_elements);

Status validate_q6_k_matmul_kernel(Backend&,OperatorRegistry&,QuantizationRegistry&);

Status build_q6_k_matmul_candidate(const Tensor&,const std::shared_ptr<Buffer>&,uint64_t,const Tensor&,ComputeDispatch&);
Status build_q6_k_matmul_gles_candidate(const Tensor&,const std::shared_ptr<Buffer>&,uint64_t,const Tensor&,ComputeDispatch&);
Status reference_q6_k_matmul(const float*,const uint8_t*,uint64_t,float*,uint32_t,uint32_t,uint32_t);



Status validate_q4_k_matmul_kernel(Backend&,OperatorRegistry&,QuantizationRegistry&);
Status validate_q5_k_matmul_kernel(Backend&,OperatorRegistry&,QuantizationRegistry&);

Status build_q4_k_matmul_candidate(const Tensor&,const std::shared_ptr<Buffer>&,uint64_t,const Tensor&,ComputeDispatch&);
Status build_q4_k_matmul_gles_candidate(const Tensor&,const std::shared_ptr<Buffer>&,uint64_t,const Tensor&,ComputeDispatch&);
Status reference_q4_k_matmul(const float*,const uint8_t*,uint64_t,float*,uint32_t,uint32_t,uint32_t);
Status build_q5_k_matmul_candidate(const Tensor&,const std::shared_ptr<Buffer>&,uint64_t,const Tensor&,ComputeDispatch&);
Status build_q5_k_matmul_gles_candidate(const Tensor&,const std::shared_ptr<Buffer>&,uint64_t,const Tensor&,ComputeDispatch&);
Status reference_q5_k_matmul(const float*,const uint8_t*,uint64_t,float*,uint32_t,uint32_t,uint32_t);


Status validate_q2_k_matmul_kernel(Backend&,OperatorRegistry&,QuantizationRegistry&);
Status validate_q3_k_matmul_kernel(Backend&,OperatorRegistry&,QuantizationRegistry&);

Status build_q2_k_matmul_candidate(const Tensor&,const std::shared_ptr<Buffer>&,uint64_t,const Tensor&,ComputeDispatch&);
Status build_q2_k_matmul_gles_candidate(const Tensor&,const std::shared_ptr<Buffer>&,uint64_t,const Tensor&,ComputeDispatch&);
Status reference_q2_k_matmul(const float*,const uint8_t*,uint64_t,float*,uint32_t,uint32_t,uint32_t);
Status build_q3_k_matmul_candidate(const Tensor&,const std::shared_ptr<Buffer>&,uint64_t,const Tensor&,ComputeDispatch&);
Status build_q3_k_matmul_gles_candidate(const Tensor&,const std::shared_ptr<Buffer>&,uint64_t,const Tensor&,ComputeDispatch&);
Status reference_q3_k_matmul(const float*,const uint8_t*,uint64_t,float*,uint32_t,uint32_t,uint32_t);

Status decode_q2_k_blocks(const uint8_t*,size_t,float*,size_t);
Status decode_q3_k_blocks(const uint8_t*,size_t,float*,size_t);

Status decode_q4_k_blocks(const uint8_t*,size_t,float*,size_t);
Status decode_q5_k_blocks(const uint8_t*,size_t,float*,size_t);

Status decode_q6_k_blocks(const uint8_t* source,size_t source_bytes,float* destination,size_t destination_elements);

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
// Q4_0 packed-block kernels are candidates until backend-specific numerical parity passes.
const char* q4_0_matmul_wgsl_source() noexcept;
const char* q4_0_matmul_glsl_source() noexcept;
const char* q4_0_embedding_wgsl_source() noexcept;
const char* q4_0_embedding_glsl_source() noexcept;
const char* q8_0_embedding_wgsl_source() noexcept;
const char* q8_0_embedding_glsl_source() noexcept;
const char* q4_1_embedding_wgsl_source() noexcept;
const char* q4_1_embedding_glsl_source() noexcept;
Status build_q4_1_embedding_candidate(const std::shared_ptr<Buffer>& quantized_weights, uint64_t weight_bytes, uint32_t vocabulary_size, uint32_t embedding_dimension, const std::shared_ptr<Buffer>& token_ids_buffer, const std::vector<uint32_t>& token_ids, const Tensor& output, ComputeDispatch& dispatch);
Status build_q4_1_embedding_gles_candidate(const std::shared_ptr<Buffer>& quantized_weights, uint64_t weight_bytes, uint32_t vocabulary_size, uint32_t embedding_dimension, const std::shared_ptr<Buffer>& token_ids_buffer, const std::vector<uint32_t>& token_ids, const Tensor& output, ComputeDispatch& dispatch);
Status reference_q4_1_embedding(const uint8_t* quantized_weights, uint64_t weight_bytes, uint32_t vocabulary_size, uint32_t embedding_dimension, const std::vector<uint32_t>& token_ids, float* output);
Status validate_q4_1_embedding_kernel(Backend& backend, OperatorRegistry& operators);
Status build_q4_0_embedding_candidate(const std::shared_ptr<Buffer>& quantized_weights, uint64_t weight_bytes, uint32_t vocabulary_size, uint32_t embedding_dimension, const std::shared_ptr<Buffer>& token_ids_buffer, const std::vector<uint32_t>& token_ids, const Tensor& output, ComputeDispatch& dispatch);
Status build_q4_0_embedding_gles_candidate(const std::shared_ptr<Buffer>& quantized_weights, uint64_t weight_bytes, uint32_t vocabulary_size, uint32_t embedding_dimension, const std::shared_ptr<Buffer>& token_ids_buffer, const std::vector<uint32_t>& token_ids, const Tensor& output, ComputeDispatch& dispatch);
Status reference_q4_0_embedding(const uint8_t* quantized_weights, uint64_t weight_bytes, uint32_t vocabulary_size, uint32_t embedding_dimension, const std::vector<uint32_t>& token_ids, float* output);
Status validate_q4_0_embedding_kernel(Backend& backend, OperatorRegistry& operators);
Status build_q8_0_embedding_candidate(const std::shared_ptr<Buffer>& quantized_weights, uint64_t weight_bytes, uint32_t vocabulary_size, uint32_t embedding_dimension, const std::shared_ptr<Buffer>& token_ids_buffer, const std::vector<uint32_t>& token_ids, const Tensor& output, ComputeDispatch& dispatch);
Status build_q8_0_embedding_gles_candidate(const std::shared_ptr<Buffer>& quantized_weights, uint64_t weight_bytes, uint32_t vocabulary_size, uint32_t embedding_dimension, const std::shared_ptr<Buffer>& token_ids_buffer, const std::vector<uint32_t>& token_ids, const Tensor& output, ComputeDispatch& dispatch);
Status reference_q8_0_embedding(const uint8_t* quantized_weights, uint64_t weight_bytes, uint32_t vocabulary_size, uint32_t embedding_dimension, const std::vector<uint32_t>& token_ids, float* output);
Status validate_q8_0_embedding_kernel(Backend& backend, OperatorRegistry& operators);
const char* q4_1_matmul_wgsl_source() noexcept;
const char* q4_1_matmul_glsl_source() noexcept;

const char* q5_0_matmul_wgsl_source() noexcept;
const char* q5_0_matmul_glsl_source() noexcept;
const char* q5_1_matmul_wgsl_source() noexcept;
const char* q5_1_matmul_glsl_source() noexcept;
Status validate_q5_matmul_kernel_impl(Backend& backend, OperatorRegistry& operators, QuantizationRegistry& quantizations, bool q5_1);



Status validate_f16_matmul_kernel(Backend&,OperatorRegistry&,QuantizationRegistry&);
Status validate_bf16_matmul_kernel(Backend&,OperatorRegistry&,QuantizationRegistry&);
Status validate_f16_embedding_kernel(Backend&,OperatorRegistry&,QuantizationRegistry&);
Status validate_bf16_embedding_kernel(Backend&,OperatorRegistry&,QuantizationRegistry&);

Status build_f16_embedding_candidate(const std::shared_ptr<Buffer>&,uint64_t,uint32_t,uint32_t,const std::shared_ptr<Buffer>&,const std::vector<uint32_t>&,const Tensor&,ComputeDispatch&);
Status build_f16_embedding_gles_candidate(const std::shared_ptr<Buffer>&,uint64_t,uint32_t,uint32_t,const std::shared_ptr<Buffer>&,const std::vector<uint32_t>&,const Tensor&,ComputeDispatch&);
Status build_bf16_embedding_candidate(const std::shared_ptr<Buffer>&,uint64_t,uint32_t,uint32_t,const std::shared_ptr<Buffer>&,const std::vector<uint32_t>&,const Tensor&,ComputeDispatch&);
Status build_bf16_embedding_gles_candidate(const std::shared_ptr<Buffer>&,uint64_t,uint32_t,uint32_t,const std::shared_ptr<Buffer>&,const std::vector<uint32_t>&,const Tensor&,ComputeDispatch&);
Status reference_f16_embedding(const uint8_t*,uint64_t,uint32_t,uint32_t,const std::vector<uint32_t>&,float*,bool bf16=false);

Status build_f16_matmul_candidate(const Tensor&,const std::shared_ptr<Buffer>&,uint64_t,const Tensor&,ComputeDispatch&);
Status build_f16_matmul_gles_candidate(const Tensor&,const std::shared_ptr<Buffer>&,uint64_t,const Tensor&,ComputeDispatch&);
Status build_bf16_matmul_candidate(const Tensor&,const std::shared_ptr<Buffer>&,uint64_t,const Tensor&,ComputeDispatch&);
Status build_bf16_matmul_gles_candidate(const Tensor&,const std::shared_ptr<Buffer>&,uint64_t,const Tensor&,ComputeDispatch&);
Status reference_f16_matmul(const float*,const uint8_t*,uint64_t,float*,uint32_t,uint32_t,uint32_t,bool bf16=false);

Status validate_q5_0_matmul_kernel(Backend& backend, OperatorRegistry& operators, QuantizationRegistry& quantizations);
Status validate_q5_1_matmul_kernel(Backend& backend, OperatorRegistry& operators, QuantizationRegistry& quantizations);



const char* q5_0_embedding_wgsl_source() noexcept;
const char* q5_0_embedding_glsl_source() noexcept;
const char* q5_1_embedding_wgsl_source() noexcept;
const char* q5_1_embedding_glsl_source() noexcept;


Status validate_q5_0_embedding_kernel(Backend& backend, OperatorRegistry& operators, QuantizationRegistry& quantizations);
Status validate_q5_1_embedding_kernel(Backend& backend, OperatorRegistry& operators, QuantizationRegistry& quantizations);

Status build_q5_0_embedding_candidate(const std::shared_ptr<Buffer>&,uint64_t,uint32_t,uint32_t,const std::shared_ptr<Buffer>&,const std::vector<uint32_t>&,const Tensor&,ComputeDispatch&);
Status build_q5_0_embedding_gles_candidate(const std::shared_ptr<Buffer>&,uint64_t,uint32_t,uint32_t,const std::shared_ptr<Buffer>&,const std::vector<uint32_t>&,const Tensor&,ComputeDispatch&);
Status reference_q5_0_embedding(const uint8_t*,uint64_t,uint32_t,uint32_t,const std::vector<uint32_t>&,float*);
Status build_q5_1_embedding_candidate(const std::shared_ptr<Buffer>&,uint64_t,uint32_t,uint32_t,const std::shared_ptr<Buffer>&,const std::vector<uint32_t>&,const Tensor&,ComputeDispatch&);
Status build_q5_1_embedding_gles_candidate(const std::shared_ptr<Buffer>&,uint64_t,uint32_t,uint32_t,const std::shared_ptr<Buffer>&,const std::vector<uint32_t>&,const Tensor&,ComputeDispatch&);
Status reference_q5_1_embedding(const uint8_t*,uint64_t,uint32_t,uint32_t,const std::vector<uint32_t>&,float*);

Status build_q5_0_matmul_candidate(const Tensor&, const std::shared_ptr<Buffer>&, uint64_t, const Tensor&, ComputeDispatch&);
Status build_q5_0_matmul_gles_candidate(const Tensor&, const std::shared_ptr<Buffer>&, uint64_t, const Tensor&, ComputeDispatch&);
Status reference_q5_0_matmul(const float*, const uint8_t*, uint64_t, float*, uint32_t, uint32_t, uint32_t);
Status build_q5_1_matmul_candidate(const Tensor&, const std::shared_ptr<Buffer>&, uint64_t, const Tensor&, ComputeDispatch&);
Status build_q5_1_matmul_gles_candidate(const Tensor&, const std::shared_ptr<Buffer>&, uint64_t, const Tensor&, ComputeDispatch&);
Status reference_q5_1_matmul(const float*, const uint8_t*, uint64_t, float*, uint32_t, uint32_t, uint32_t);

Status build_q4_1_matmul_candidate(const Tensor& activations, const std::shared_ptr<Buffer>& quantized_weights, uint64_t weight_bytes, const Tensor& output, ComputeDispatch& dispatch);
Status build_q4_1_matmul_gles_candidate(const Tensor& activations, const std::shared_ptr<Buffer>& quantized_weights, uint64_t weight_bytes, const Tensor& output, ComputeDispatch& dispatch);
Status reference_q4_1_matmul(const float* activations, const uint8_t* quantized_weights, uint64_t weight_bytes, float* output, uint32_t rows, uint32_t output_channels, uint32_t inner);
Status validate_q4_1_matmul_kernel(Backend& backend, OperatorRegistry& operators, QuantizationRegistry& quantizations);
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
const char* q4_0_dequantize_glsl_source() noexcept;
const char* q4_1_dequantize_glsl_source() noexcept;
const char* q8_0_dequantize_glsl_source() noexcept;
Status build_q4_0_dequantize_gles_candidate(const std::shared_ptr<Buffer>& encoded, uint64_t encoded_bytes, const Tensor& output, ComputeDispatch& dispatch);
Status build_q4_1_dequantize_gles_candidate(const std::shared_ptr<Buffer>& encoded, uint64_t encoded_bytes, const Tensor& output, ComputeDispatch& dispatch);
Status build_q8_0_dequantize_gles_candidate(const std::shared_ptr<Buffer>& encoded, uint64_t encoded_bytes, const Tensor& output, ComputeDispatch& dispatch);

Status validate_q4_0_dequantize_kernel(Backend& backend, ComputeDispatch& validated_dispatch);
Status validate_q4_1_dequantize_kernel(Backend& backend, ComputeDispatch& validated_dispatch);
Status validate_q5_0_dequantize_kernel(Backend& backend, ComputeDispatch& validated_dispatch);
Status validate_q5_1_dequantize_kernel(Backend& backend, ComputeDispatch& validated_dispatch);
Status validate_q8_0_dequantize_kernel(Backend& backend, ComputeDispatch& validated_dispatch);

const char* q5_0_dequantize_wgsl_source() noexcept;
const char* q5_0_dequantize_glsl_source() noexcept;
const char* q5_1_dequantize_glsl_source() noexcept;
Status build_q5_0_dequantize_gles_candidate(const std::shared_ptr<Buffer>& encoded,
                                            uint64_t encoded_bytes, const Tensor& output,
                                            ComputeDispatch& dispatch);
Status build_q5_1_dequantize_gles_candidate(const std::shared_ptr<Buffer>& encoded,
                                            uint64_t encoded_bytes, const Tensor& output,
                                            ComputeDispatch& dispatch);
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
