#include "model_registry.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace ai_chat::gpu_engine {

Status QuantizationRegistry::register_format(QuantizationFormat format) {
    if (format.name.empty() || format.block_elements == 0 || format.bytes_per_block == 0) {
        return {StatusCode::InvalidArgument, "Quantization format requires a name and valid block layout"};
    }
    if (formats_.find(format.file_type_id) != formats_.end()) {
        return {StatusCode::InvalidArgument, "Quantization file type id is already registered"};
    }
    formats_.emplace(format.file_type_id, std::move(format));
    return Status::success();
}

const QuantizationFormat* QuantizationRegistry::find(uint32_t file_type_id) const noexcept {
    const auto found = formats_.find(file_type_id);
    return found == formats_.end() ? nullptr : &found->second;
}

bool QuantizationRegistry::can_decode(uint32_t file_type_id) const noexcept {
    const auto* format = find(file_type_id);
    return format && format->decoder_state == SupportState::Validated;
}

bool QuantizationRegistry::can_execute_on_gpu(uint32_t file_type_id) const noexcept {
    const auto* format = find(file_type_id);
    return format && format->decoder_state == SupportState::Validated &&
           format->gpu_kernel_state == SupportState::Validated;
}

std::vector<QuantizationFormat> QuantizationRegistry::formats() const {
    std::vector<QuantizationFormat> result;
    result.reserve(formats_.size());
    for (const auto& entry : formats_) result.push_back(entry.second);
    std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
        return a.file_type_id < b.file_type_id;
    });
    return result;
}

Status ArchitectureRegistry::register_adapter(std::shared_ptr<ArchitectureAdapter> adapter) {
    if (!adapter) return {StatusCode::InvalidArgument, "Cannot register a null architecture adapter"};
    const ArchitectureDescriptor& descriptor = adapter->descriptor();
    if (descriptor.name.empty() || descriptor.family == ArchitectureFamily::Unknown) {
        return {StatusCode::InvalidArgument, "Architecture adapter requires a name and known family"};
    }
    if (adapters_.find(descriptor.name) != adapters_.end()) {
        return {StatusCode::InvalidArgument, "Architecture adapter is already registered"};
    }
    adapters_.emplace(descriptor.name, std::move(adapter));
    return Status::success();
}

std::shared_ptr<const ArchitectureAdapter> ArchitectureRegistry::find(
        const std::string& architecture) const {
    const auto found = adapters_.find(architecture);
    return found == adapters_.end() ? nullptr : found->second;
}

std::vector<std::string> ArchitectureRegistry::registered_architectures() const {
    std::vector<std::string> result;
    result.reserve(adapters_.size());
    for (const auto& entry : adapters_) result.push_back(entry.first);
    std::sort(result.begin(), result.end());
    return result;
}

ModelPreflightReport evaluate_model_preflight(const GgufFileInfo& model,
                                              const ArchitectureRegistry& architectures,
                                              const QuantizationRegistry& quantizations) {
    ModelPreflightReport report;
    report.architecture = model.architecture;
    report.tensor_count = model.tensor_count;
    report.all_tensor_storage_known = true;
    report.all_quantization_decoders_validated = true;

    const auto adapter = architectures.find(model.architecture);
    report.architecture_known = static_cast<bool>(adapter);
    if (!adapter) {
        report.blockers.emplace_back("UNSUPPORTED_ARCHITECTURE:" + model.architecture);
    } else if (adapter->descriptor().validation != SupportState::Validated) {
        report.blockers.emplace_back("UNVERIFIED_ARCHITECTURE_ADAPTER:" + model.architecture);
        report.all_quantization_decoders_validated = false;
    } else {
        const Status metadata_status = adapter->validate_metadata(model);
        if (!metadata_status.ok()) {
            report.blockers.emplace_back("ARCHITECTURE_METADATA_INVALID:" + metadata_status.message);
        }
    }

    for (const GgufTensorInfo& tensor : model.tensors) {
        if (!tensor.byte_size_known) {
            report.all_tensor_storage_known = false;
            ++report.unknown_storage_tensors;
            report.blockers.emplace_back("UNKNOWN_TENSOR_STORAGE:" + tensor.name);
            continue;
        }
        if (tensor.byte_size > std::numeric_limits<uint64_t>::max() - report.known_weight_bytes) {
            report.all_tensor_storage_known = false;
            report.blockers.emplace_back("WEIGHT_SIZE_OVERFLOW:" + tensor.name);
        } else {
            report.known_weight_bytes += tensor.byte_size;
        }

        if (tensor.type.quantized) {
            if (std::find(report.quantization_formats.begin(), report.quantization_formats.end(),
                          tensor.type.name) == report.quantization_formats.end()) {
                report.quantization_formats.push_back(tensor.type.name);
            }
            const QuantizationFormat* format = quantizations.find(tensor.type_id);
            if (!format || format->decoder_state != SupportState::Validated) {
                report.all_quantization_decoders_validated = false;
                ++report.unverified_quantized_tensors;
                report.blockers.emplace_back("UNVERIFIED_QUANT_DECODER:" + tensor.type.name);
            } else if (format->gpu_kernel_state != SupportState::Validated) {
                report.blockers.emplace_back("UNVERIFIED_GPU_QUANT_KERNEL:" + tensor.type.name);
            }
        }
    }

    report.gpu_execution_ready = report.architecture_known &&
        adapter->descriptor().validation == SupportState::Validated &&
        report.all_tensor_storage_known && report.all_quantization_decoders_validated &&
        report.blockers.empty();
    return report;
}

} // namespace ai_chat::gpu_engine
