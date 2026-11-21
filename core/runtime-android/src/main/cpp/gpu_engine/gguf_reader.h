#pragma once

#include "gpu_engine.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace ai_chat::gpu_engine {

struct GgufFileStream;

struct GgufArrayInfo {
    uint32_t element_type = 0;
    uint64_t count = 0;
};

using GgufMetadataValue = std::variant<uint64_t, int64_t, double, bool, std::string, GgufArrayInfo,
                                      std::vector<std::string>, std::vector<double>, std::vector<int64_t>>;

struct GgufTensorTypeInfo {
    uint32_t type_id = 0;
    std::string name;
    uint32_t block_elements = 0;
    uint32_t bytes_per_block = 0;
    bool quantized = false;
    bool storage_size_known = false;
    bool decoder_validated = false;
};

struct GgufTensorInfo {
    std::string name;
    std::vector<uint64_t> dimensions;
    uint32_t type_id = 0;
    uint64_t relative_offset = 0;
    uint64_t absolute_offset = 0;
    uint64_t byte_size = 0;
    bool byte_size_known = false;
    GgufTensorTypeInfo type;
};

struct GgufTokenizerInfo {
    std::string model = "unknown";
    std::string pre_tokenizer = "unknown";
    std::vector<std::string> tokens;
    std::vector<std::string> merges;
    std::vector<double> scores;
    std::vector<int64_t> token_types;
    int64_t bos_token_id = -1;
    int64_t eos_token_id = -1;
    int64_t unknown_token_id = -1;
    int64_t padding_token_id = -1;
    bool add_bos_token = false;
    bool add_eos_token = false;
    bool add_space_prefix = false;
    bool remove_extra_whitespaces = false;
    bool escape_whitespaces = true;
    bool byte_fallback = false;
};

struct GgufFileInfo {
    uint32_t version = 0;
    uint64_t file_size_bytes = 0;
    uint64_t data_section_offset = 0;
    uint64_t tensor_count = 0;
    uint64_t metadata_count = 0;
    uint32_t alignment = 32;
    std::string architecture = "unknown";
    std::string model_name = "unknown";
    GgufTokenizerInfo tokenizer;
    std::unordered_map<std::string, GgufMetadataValue> metadata;
    std::vector<GgufTensorInfo> tensors;
};

class GgufReader {
public:
    static Status open(const std::string& path, GgufReader& out);

    const GgufFileInfo& info() const noexcept { return info_; }
    const GgufTensorInfo* find_tensor(const std::string& name) const noexcept;

    // Reads only the requested byte range; callers can stream large weight tensors.
    Status read_tensor_chunk(const GgufTensorInfo& tensor, uint64_t tensor_offset,
                             void* destination, uint64_t size_bytes) const;

    static GgufTensorTypeInfo tensor_type_info(uint32_t type_id);

private:
    std::string path_;
    GgufFileInfo info_;
    std::shared_ptr<GgufFileStream> file_stream_;
    std::unordered_map<std::string, size_t> tensor_indices_;
};

} // namespace ai_chat::gpu_engine
