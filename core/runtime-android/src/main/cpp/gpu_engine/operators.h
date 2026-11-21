#pragma once

#include "execution_engine.h"

namespace ai_chat::gpu_engine {

// Generic WGSL candidate. It remains UNVERIFIED until compared with the
// reference implementation on real Dawn/WebGPU devices.
const char* add_f32_wgsl_source() noexcept;

Status build_add_f32_candidate(const Tensor& lhs, const Tensor& rhs,
                              const Tensor& output, ComputeDispatch& dispatch);

Status reference_add_f32(const float* lhs, const float* rhs, float* output,
                         uint64_t element_count);
const char* multiply_f32_wgsl_source() noexcept;
Status build_multiply_f32_candidate(const Tensor& lhs, const Tensor& rhs,
                                    const Tensor& output, ComputeDispatch& dispatch);
Status reference_multiply_f32(const float* lhs, const float* rhs, float* output,
                              uint64_t element_count);

const char* embedding_f32_wgsl_source() noexcept;
Status build_embedding_f32_candidate(const Tensor& embedding_weights,
                                     const std::shared_ptr<Buffer>& token_ids_buffer,
                                     const std::vector<uint32_t>& validated_token_ids,
                                     const Tensor& output,
                                     ComputeDispatch& dispatch);
Status reference_embedding_f32(const float* embedding_weights, uint32_t vocabulary_size,
                               uint32_t embedding_dimension,
                               const std::vector<uint32_t>& token_ids, float* output);

const char* matmul_f32_wgsl_source() noexcept;
Status build_matmul_f32_candidate(const Tensor& lhs, const Tensor& rhs,
                                  const Tensor& output, ComputeDispatch& dispatch);
Status reference_matmul_f32(const float* lhs, const float* rhs, float* output,
                            uint64_t rows, uint64_t columns, uint64_t inner);

const char* rms_norm_f32_wgsl_source() noexcept;
Status build_rms_norm_f32_candidate(const Tensor& input, const Tensor& weight,
                                    const Tensor& output, float epsilon,
                                    ComputeDispatch& dispatch);
Status reference_rms_norm_f32(const float* input, const float* weight, float* output,
                              uint64_t rows, uint64_t columns, float epsilon);

const char* silu_f32_wgsl_source() noexcept;
Status build_silu_f32_candidate(const Tensor& input, const Tensor& output,
                                ComputeDispatch& dispatch);
Status reference_silu_f32(const float* input, float* output, uint64_t elements);

const char* softmax_f32_wgsl_source() noexcept;
Status build_softmax_f32_candidate(const Tensor& input, const Tensor& output,
                                   ComputeDispatch& dispatch);
Status reference_softmax_f32(const float* input, float* output,
                             uint64_t rows, uint64_t columns);

struct RopeConfig {
    uint32_t rotary_dimensions = 0;
    uint32_t start_position = 0;
    float theta = 10000.0f;
    bool neox_style = false;
};
const char* rope_f32_wgsl_source() noexcept;
Status build_rope_f32_candidate(const Tensor& input, const Tensor& output,
                                const RopeConfig& config, ComputeDispatch& dispatch);
Status reference_rope_f32(const float* input, float* output,
                          uint32_t tokens, uint32_t heads, uint32_t head_dimension,
                          const RopeConfig& config);

Status register_builtin_operator_candidates(OperatorRegistry& registry);

} // namespace ai_chat::gpu_engine
