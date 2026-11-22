#pragma once

#include "execution_engine.h"

namespace ai_chat::gpu_engine {

struct RopeConfig;

// Generic backend-specific candidates remain UNVERIFIED until GPU dispatch/readback
// has passed numerical parity tests on the selected backend.
const char* add_f32_wgsl_source() noexcept;

Status build_add_f32_candidate(const Tensor& lhs, const Tensor& rhs,
                              const Tensor& output, ComputeDispatch& dispatch);

Status reference_add_f32(const float* lhs, const float* rhs, float* output,
                         uint64_t element_count);
Status reference_add_f32_trailing_broadcast(const float* lhs, const float* rhs,
                                            float* output, uint64_t element_count,
                                            uint64_t broadcast_width);
Status build_add_f32_gles_candidate(const Tensor& lhs, const Tensor& rhs,
                                    const Tensor& output, ComputeDispatch& dispatch);
const char* multiply_f32_wgsl_source() noexcept;
Status build_multiply_f32_candidate(const Tensor& lhs, const Tensor& rhs,
                                    const Tensor& output, ComputeDispatch& dispatch);
Status build_multiply_f32_gles_candidate(const Tensor& lhs, const Tensor& rhs,
                                         const Tensor& output, ComputeDispatch& dispatch);
Status build_matmul_f32_gles_candidate(const Tensor& lhs, const Tensor& rhs,
                                       const Tensor& output, ComputeDispatch& dispatch);
Status build_rms_norm_f32_gles_candidate(const Tensor& input, const Tensor& weight,
                                         const Tensor& output, float epsilon,
                                         ComputeDispatch& dispatch);
Status build_silu_f32_gles_candidate(const Tensor& input, const Tensor& output,
                                     ComputeDispatch& dispatch);
Status build_softmax_f32_gles_candidate(const Tensor& input, const Tensor& output,
                                        ComputeDispatch& dispatch);
Status build_rope_f32_gles_candidate(const Tensor& input, const Tensor& output,
                                     const RopeConfig& config, ComputeDispatch& dispatch);
Status build_embedding_f32_gles_candidate(const Tensor& embedding_weights,
                                          const std::shared_ptr<Buffer>& token_ids_buffer,
                                          const std::vector<uint32_t>& validated_token_ids,
                                          const Tensor& output, ComputeDispatch& dispatch);

Status build_attention_f32_gles_candidate(const Tensor& query, const Tensor& key,
                                          const Tensor& value, const Tensor& output,
                                          uint32_t past_length, ComputeDispatch& dispatch);
Status register_builtin_gles_operator_candidates(OperatorRegistry& registry);
Status validate_gles_add_f32_kernel(Backend& backend, OperatorRegistry& registry);
Status validate_dawn_add_f32_kernel(Backend& backend, OperatorRegistry& registry);
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
Status build_sigmoid_f32_candidate(const Tensor& input, const Tensor& output,
                                   ComputeDispatch& dispatch);
Status build_sigmoid_f32_gles_candidate(const Tensor& input, const Tensor& output,
                                        ComputeDispatch& dispatch);
Status reference_sigmoid_f32(const float* input, float* output, uint64_t elements);
Status build_softplus_f32_candidate(const Tensor& input, const Tensor& output,
                                    ComputeDispatch& dispatch);
Status build_softplus_f32_gles_candidate(const Tensor& input, const Tensor& output,
                                         ComputeDispatch& dispatch);
Status reference_softplus_f32(const float* input, float* output, uint64_t elements);
Status build_split_f32_candidate(const Tensor& input, const Tensor& output,
                                 uint32_t axis, uint32_t offset,
                                 ComputeDispatch& dispatch);
Status build_split_f32_gles_candidate(const Tensor& input, const Tensor& output,
                                      uint32_t axis, uint32_t offset,
                                      ComputeDispatch& dispatch);
Status reference_split_f32(const float* input, float* output,
                           const std::vector<uint64_t>& input_shape,
                           const std::vector<uint64_t>& output_shape,
                           uint32_t axis, uint32_t offset);
Status register_split_operator_candidate(OperatorRegistry& registry, BackendKind backend);
Status build_gdn_l2_norm_f32_candidate(const Tensor& input, const Tensor& output,
                                       uint32_t state_size, uint32_t group_count,
                                       float epsilon, ComputeDispatch& dispatch);
Status build_gdn_l2_norm_f32_gles_candidate(const Tensor& input, const Tensor& output,
                                            uint32_t state_size, uint32_t group_count,
                                            float epsilon, ComputeDispatch& dispatch);
Status reference_gdn_l2_norm_f32(const float* input, float* output,
                                 uint32_t tokens, uint32_t state_size,
                                 uint32_t group_count, float epsilon);
Status register_gdn_l2_norm_operator_candidate(OperatorRegistry& registry, BackendKind backend);

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

const char* attention_f32_wgsl_source() noexcept;
Status build_attention_f32_candidate(const Tensor& query, const Tensor& key,
                                     const Tensor& value, const Tensor& output,
                                     uint32_t past_length, ComputeDispatch& dispatch);
Status reference_attention_f32(const float* query, const float* key, const float* value,
                               float* output, uint32_t tokens, uint32_t query_heads,
                               uint32_t kv_heads, uint32_t head_dimension,
                               uint32_t past_length);

struct KernelValidationReport {
    std::vector<std::string> validated;
    std::vector<std::string> rejected;
    std::vector<std::string> unavailable;
    bool complete = false;
};

// Runs deterministic GPU-vs-reference tests and promotes only kernels that
// pass readback parity on the selected physical backend.
Status validate_builtin_operator_kernels(Backend& backend, OperatorRegistry& registry,
                                         KernelValidationReport& report);
Status register_builtin_operator_candidates(OperatorRegistry& registry);

} // namespace ai_chat::gpu_engine
