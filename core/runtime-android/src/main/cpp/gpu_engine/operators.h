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

Status register_builtin_operator_candidates(OperatorRegistry& registry);

} // namespace ai_chat::gpu_engine
