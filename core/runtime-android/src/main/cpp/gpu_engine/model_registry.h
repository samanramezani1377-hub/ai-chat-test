#pragma once

#include "gguf_reader.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace ai_chat::gpu_engine {

enum class SupportState : uint8_t {
    Unverified = 0,
    Validated,
    Unsupported,
    FailedValidation,
};

struct QuantizationFormat {
    uint32_t file_type_id = 0;
    std::string name;
    uint32_t block_elements = 0;
    uint32_t bytes_per_block = 0;
    bool quantized = false;
    SupportState decoder_state = SupportState::Unverified;
    SupportState gpu_kernel_state = SupportState::Unverified;
};

class QuantizationRegistry {
public:
    Status register_format(QuantizationFormat format);
    const QuantizationFormat* find(uint32_t file_type_id) const noexcept;
    bool can_decode(uint32_t file_type_id) const noexcept;
    bool can_execute_on_gpu(uint32_t file_type_id) const noexcept;
    std::vector<QuantizationFormat> formats() const;

private:
    std::unordered_map<uint32_t, QuantizationFormat> formats_;
};

enum class ArchitectureFamily : uint8_t {
    Unknown = 0,
    DenseTransformer,
    MixtureOfExperts,
    Hybrid,
    RecurrentStateSpace,
    Multimodal,
    Convolutional,
};

struct ArchitectureDescriptor {
    std::string name;
    ArchitectureFamily family = ArchitectureFamily::Unknown;
    std::string version;
    std::vector<std::string> required_operators;
    bool uses_attention = false;
    bool uses_kv_cache = false;
    bool uses_recurrent_state = false;
    bool supports_dynamic_shapes = false;
    SupportState validation = SupportState::Unverified;
};

class ArchitectureAdapter {
public:
    virtual ~ArchitectureAdapter() = default;
    virtual const ArchitectureDescriptor& descriptor() const noexcept = 0;
    virtual Status validate_metadata(const GgufFileInfo& model) const = 0;
};

class ArchitectureRegistry {
public:
    Status register_adapter(std::shared_ptr<ArchitectureAdapter> adapter);
    std::shared_ptr<const ArchitectureAdapter> find(const std::string& architecture) const;
    std::vector<std::string> registered_architectures() const;

private:
    std::unordered_map<std::string, std::shared_ptr<ArchitectureAdapter>> adapters_;
};

struct ModelPreflightReport {
    std::string architecture = "unknown";
    std::vector<std::string> quantization_formats;
    uint64_t tensor_count = 0;
    uint64_t known_weight_bytes = 0;
    uint64_t unknown_storage_tensors = 0;
    uint64_t unverified_quantized_tensors = 0;
    bool architecture_known = false;
    bool all_tensor_storage_known = false;
    bool all_quantization_decoders_validated = false;
    bool gpu_execution_ready = false;
    std::vector<std::string> blockers;
};

Status register_builtin_architecture_adapters(ArchitectureRegistry& registry);

ModelPreflightReport evaluate_model_preflight(const GgufFileInfo& model,
                                              const ArchitectureRegistry& architectures,
                                              const QuantizationRegistry& quantizations);

} // namespace ai_chat::gpu_engine
