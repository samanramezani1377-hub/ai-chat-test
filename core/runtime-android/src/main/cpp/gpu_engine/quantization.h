#pragma once

#include "model_registry.h"

#include <cstddef>
#include <cstdint>

namespace ai_chat::gpu_engine {

// Decodes complete GGUF Q8_0 blocks (32 signed values + one FP16 scale).
// This validates the host decoder only; no GPU Q8_0 MatMul kernel is implied.
Status decode_q8_0_blocks(const uint8_t* source, size_t source_bytes,
                          float* destination, size_t destination_elements);

Status register_builtin_quantization_formats(QuantizationRegistry& registry);

} // namespace ai_chat::gpu_engine
