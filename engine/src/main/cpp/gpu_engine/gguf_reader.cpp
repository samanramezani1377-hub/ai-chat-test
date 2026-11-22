#include "gguf_reader.h"

#include <algorithm>
#include <cstring>
#include <array>
#include <fstream>
#include <cmath>
#include <limits>
#include <set>
#include <utility>
#include <mutex>
#include <type_traits>

namespace ai_chat::gpu_engine {

struct GgufFileStream {
    std::ifstream input;
    std::mutex mutex;
};

namespace {

class Reader {
public:
    explicit Reader(std::ifstream& input, uint64_t file_size)
        : input_(input), file_size_(file_size) {}

    uint64_t position() {
        const auto pos = input_.tellg();
        if (pos < 0) return file_size_;
        return static_cast<uint64_t>(pos);
    }

    uint64_t remaining() {
        const uint64_t pos = position();
        return pos <= file_size_ ? file_size_ - pos : 0;
    }

    bool read_u8(uint8_t& value) {
        char byte = 0;
        if (!input_.read(&byte, 1)) return false;
        value = static_cast<uint8_t>(static_cast<unsigned char>(byte));
        return true;
    }

    bool read_u32(uint32_t& value) {
        std::array<uint8_t, 4> bytes{};
        for (auto& byte : bytes) if (!read_u8(byte)) return false;
        value = static_cast<uint32_t>(bytes[0]) |
                (static_cast<uint32_t>(bytes[1]) << 8) |
                (static_cast<uint32_t>(bytes[2]) << 16) |
                (static_cast<uint32_t>(bytes[3]) << 24);
        return true;
    }

    bool read_u64(uint64_t& value) {
        std::array<uint8_t, 8> bytes{};
        for (auto& byte : bytes) if (!read_u8(byte)) return false;
        value = 0;
        for (uint32_t i = 0; i < 8; ++i) value |= static_cast<uint64_t>(bytes[i]) << (i * 8);
        return true;
    }

    bool read_string(std::string& value) {
        uint64_t length = 0;
        if (!read_u64(length) || length > remaining() || length > (1ULL << 30) ||
            length > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) return false;
        value.resize(static_cast<size_t>(length));
        return length == 0 || static_cast<bool>(input_.read(value.data(), static_cast<std::streamsize>(length)));
    }

    bool skip(uint64_t count) {
        if (count > remaining() || count > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) return false;
        input_.seekg(static_cast<std::streamoff>(count), std::ios::cur);
        return static_cast<bool>(input_);
    }

    bool skip_value(uint32_t type, uint32_t depth = 0) {
        if (depth > 8) return false;
        uint32_t count32 = 0;
        uint64_t count64 = 0;
        std::string ignored;
        switch (type) {
            case 0: case 1: case 7: return skip(1);
            case 2: case 3: return skip(2);
            case 4: case 5: case 6: return skip(4);
            case 8: return read_string(ignored);
            case 9:
                if (!read_u32(count32) || !read_u64(count64) || count64 > remaining() + 1) return false;
                for (uint64_t i = 0; i < count64; ++i) {
                    if (!skip_value(count32, depth + 1)) return false;
                }
                return true;
            case 10: case 11: case 12: return skip(8);
            default: return false;
        }
    }

private:
    std::ifstream& input_;
    uint64_t file_size_;
};

bool align_up(uint64_t value, uint32_t alignment, uint64_t& out) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) return false;
    const uint64_t mask = static_cast<uint64_t>(alignment - 1);
    if (value > std::numeric_limits<uint64_t>::max() - mask) return false;
    out = (value + mask) & ~mask;
    return true;
}

bool read_metadata_value(Reader& reader, const std::string& key, uint32_t type, GgufMetadataValue& out) {
    uint8_t u8 = 0;
    uint32_t u32 = 0;
    uint64_t u64 = 0;
    std::string string_value;
    switch (type) {
        case 0:
            if (!reader.read_u8(u8)) return false;
            out = static_cast<uint64_t>(u8); return true;
        case 1:
            if (!reader.read_u8(u8)) return false;
            out = static_cast<int64_t>(static_cast<int8_t>(u8)); return true;
        case 2: {
            uint8_t lo = 0, hi = 0;
            if (!reader.read_u8(lo) || !reader.read_u8(hi)) return false;
            out = static_cast<uint64_t>(lo | (static_cast<uint16_t>(hi) << 8)); return true;
        }
        case 3: {
            uint8_t lo = 0, hi = 0;
            if (!reader.read_u8(lo) || !reader.read_u8(hi)) return false;
            out = static_cast<int64_t>(static_cast<int16_t>(lo | (static_cast<uint16_t>(hi) << 8))); return true;
        }
        case 4:
            if (!reader.read_u32(u32)) return false;
            out = static_cast<uint64_t>(u32); return true;
        case 5:
            if (!reader.read_u32(u32)) return false;
            out = static_cast<int64_t>(static_cast<int32_t>(u32)); return true;
        case 6:
            if (!reader.read_u32(u32)) return false;
            out = static_cast<double>([] (uint32_t bits) { float value = 0; std::memcpy(&value, &bits, sizeof(value)); return value; }(u32));
            return true;
        case 7:
            if (!reader.read_u8(u8) || u8 > 1) return false;
            out = (u8 != 0); return true;
        case 8:
            if (!reader.read_string(string_value)) return false;
            out = std::move(string_value); return true;
        case 9: {
            uint32_t element_type = 0;
            uint64_t count = 0;
            if (!reader.read_u32(element_type) || !reader.read_u64(count) ||
                count > reader.remaining() + 1 || count > 4'000'000) return false;

            if ((key == "tokenizer.ggml.tokens" || key == "tokenizer.ggml.merges") && element_type == 8) {
                std::vector<std::string> values;
                values.reserve(static_cast<size_t>(count));
                for (uint64_t i = 0; i < count; ++i) {
                    std::string value;
                    if (!reader.read_string(value)) return false;
                    values.push_back(std::move(value));
                }
                out = std::move(values);
                return true;
            }
            if (key == "tokenizer.ggml.scores" && element_type == 6) {
                std::vector<double> values;
                values.reserve(static_cast<size_t>(count));
                for (uint64_t i = 0; i < count; ++i) {
                    uint32_t bits = 0;
                    if (!reader.read_u32(bits)) return false;
                    float value = 0;
                    std::memcpy(&value, &bits, sizeof(value));
                    values.push_back(value);
                }
                out = std::move(values);
                return true;
            }
            if (key == "tokenizer.ggml.token_type" && (element_type == 4 || element_type == 5)) {
                std::vector<int64_t> values;
                values.reserve(static_cast<size_t>(count));
                for (uint64_t i = 0; i < count; ++i) {
                    uint32_t value = 0;
                    if (!reader.read_u32(value)) return false;
                    values.push_back(element_type == 5
                        ? static_cast<int64_t>(static_cast<int32_t>(value))
                        : static_cast<int64_t>(value));
                }
                out = std::move(values);
                return true;
            }
            // Preserve all scalar array values. Architecture-specific arrays
            // (for example RoPE scaling sections) can affect execution semantics.
            auto read_array = [&](auto& values) {
                for (uint64_t i = 0; i < count; ++i) {
                    GgufMetadataValue item;
                    if (!read_metadata_value(reader, key, element_type, item)) return false;
                    using Item = typename std::decay_t<decltype(values)>::value_type;
                    const auto* typed = std::get_if<Item>(&item);
                    if (!typed) return false;
                    values.push_back(*typed);
                }
                return true;
            };
            GgufMetadataArray array;
            array.element_type = element_type;
            switch (element_type) {
                case 0: case 2: case 4: case 10: {
                    std::vector<uint64_t> values;
                    values.reserve(static_cast<size_t>(count));
                    if (!read_array(values)) return false;
                    array.values = std::move(values);
                    break;
                }
                case 1: case 3: case 5: case 11: {
                    std::vector<int64_t> values;
                    values.reserve(static_cast<size_t>(count));
                    if (!read_array(values)) return false;
                    array.values = std::move(values);
                    break;
                }
                case 6: case 12: {
                    std::vector<double> values;
                    values.reserve(static_cast<size_t>(count));
                    if (!read_array(values)) return false;
                    array.values = std::move(values);
                    break;
                }
                case 7: {
                    std::vector<bool> values;
                    values.reserve(static_cast<size_t>(count));
                    if (!read_array(values)) return false;
                    array.values = std::move(values);
                    break;
                }
                case 8: {
                    std::vector<std::string> values;
                    values.reserve(static_cast<size_t>(count));
                    if (!read_array(values)) return false;
                    array.values = std::move(values);
                    break;
                }
                default:
                    return false;
            }
            out = std::move(array);
            return true;
        }
        case 10:
            if (!reader.read_u64(u64)) return false;
            out = u64; return true;
        case 11:
            if (!reader.read_u64(u64)) return false;
            out = static_cast<int64_t>(u64); return true;
        case 12:
            if (!reader.read_u64(u64)) return false;
            out = [] (uint64_t bits) { double value = 0; std::memcpy(&value, &bits, sizeof(value)); return value; }(u64);
            return true;
        default:
            return false;
    }
}

template <typename T>
T metadata_as(const GgufFileInfo& info, const std::string& key, T fallback) {
    const auto found = info.metadata.find(key);
    if (found == info.metadata.end()) return fallback;
    if (const auto* value = std::get_if<T>(&found->second)) return *value;
    return fallback;
}

int64_t metadata_as_i64(const GgufFileInfo& info, const std::string& key, int64_t fallback) {
    const auto found = info.metadata.find(key);
    if (found == info.metadata.end()) return fallback;
    if (const auto* value = std::get_if<int64_t>(&found->second)) return *value;
    if (const auto* value = std::get_if<uint64_t>(&found->second)) {
        return *value <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max())
            ? static_cast<int64_t>(*value) : fallback;
    }
    return fallback;
}

bool checked_product(const std::vector<uint64_t>& dimensions, uint64_t& out) {
    out = 1;
    for (uint64_t dimension : dimensions) {
        if (dimension == 0 || out > std::numeric_limits<uint64_t>::max() / dimension) return false;
        out *= dimension;
    }
    return true;
}

} // namespace

GgufTensorTypeInfo GgufReader::tensor_type_info(uint32_t type_id) {
    // Block storage sizes from the GGUF tensor type definitions. These describe
    // file layout only; they do not imply a validated GPU quantized decoder.
    switch (type_id) {
        case 0: return {0, "F32", 1, 4, false, true, true};
        case 1: return {1, "F16", 1, 2, false, true, true};
        case 2: return {2, "Q4_0", 32, 18, true, true, false};
        case 3: return {3, "Q4_1", 32, 20, true, true, false};
        case 6: return {6, "Q5_0", 32, 22, true, true, false};
        case 7: return {7, "Q5_1", 32, 24, true, true, false};
        case 8: return {8, "Q8_0", 32, 34, true, true, false};
        case 9: return {9, "Q8_1", 32, 40, true, true, false};
        case 10: return {10, "Q2_K", 256, 84, true, true, false};
        case 11: return {11, "Q3_K", 256, 110, true, true, false};
        case 12: return {12, "Q4_K", 256, 144, true, true, false};
        case 13: return {13, "Q5_K", 256, 176, true, true, false};
        case 14: return {14, "Q6_K", 256, 210, true, true, false};
        case 15: return {15, "Q8_K", 256, 292, true, true, false};
        case 16: return {16, "IQ2_XXS", 256, 66, true, true, false};
        case 17: return {17, "IQ2_XS", 256, 74, true, true, false};
        case 18: return {18, "IQ3_XXS", 256, 98, true, true, false};
        case 19: return {19, "IQ1_S", 256, 50, true, true, false};
        case 20: return {20, "IQ4_NL", 32, 18, true, true, false};
        case 21: return {21, "IQ3_S", 256, 110, true, true, false};
        case 22: return {22, "IQ2_S", 256, 82, true, true, false};
        case 23: return {23, "IQ4_XS", 256, 136, true, true, false};
        case 24: return {24, "I8", 1, 1, false, true, true};
        case 25: return {25, "I16", 1, 2, false, true, true};
        case 26: return {26, "I32", 1, 4, false, true, true};
        case 27: return {27, "I64", 1, 8, false, true, true};
        case 28: return {28, "F64", 1, 8, false, true, true};
        case 29: return {29, "IQ1_M", 256, 56, true, true, false};
        case 30: return {30, "BF16", 1, 2, false, true, true};
        case 31: return {31, "REMOVED_Q4_0_4_4", 0, 0, false, false, false};
        case 32: return {32, "REMOVED_Q4_0_4_8", 0, 0, false, false, false};
        case 33: return {33, "REMOVED_Q4_0_8_8", 0, 0, false, false, false};
        case 34: return {34, "TQ1_0", 256, 54, true, true, false};
        case 35: return {35, "TQ2_0", 256, 66, true, true, false};
        case 36: return {36, "REMOVED_IQ4_NL_4_4", 0, 0, false, false, false};
        case 37: return {37, "REMOVED_IQ4_NL_4_8", 0, 0, false, false, false};
        case 38: return {38, "REMOVED_IQ4_NL_8_8", 0, 0, false, false, false};
        case 39: return {39, "MXFP4", 32, 17, true, true, false};
        case 40: return {40, "NVFP4", 64, 36, true, true, false};
        case 41: return {41, "Q1_0", 128, 18, true, true, false};
        case 42: return {42, "Q2_0", 64, 18, true, true, false};
        case 43: return {43, "F8_E4M3", 1, 1, false, true, false};
        default: return {type_id, "UNKNOWN", 0, 0, false, false, false};
    }
}

Status GgufReader::open(const std::string& path, GgufReader& out) {
    out = GgufReader{};
    if (path.empty()) return {StatusCode::InvalidArgument, "GGUF path is empty"};
    auto stream = std::make_shared<GgufFileStream>();
    stream->input.open(path, std::ios::binary);
    if (!stream->input) return {StatusCode::InvalidArgument, "GGUF file cannot be opened"};
    std::ifstream& input = stream->input;
    input.seekg(0, std::ios::end);
    const auto end = input.tellg();
    if (end < 0) return {StatusCode::InvalidArgument, "GGUF file size cannot be determined"};
    const uint64_t file_size = static_cast<uint64_t>(end);
    input.seekg(0, std::ios::beg);
    if (file_size < 24) return {StatusCode::InvalidArgument, "GGUF file is shorter than its header"};

    Reader reader(input, file_size);
    std::array<char, 4> magic{};
    if (!input.read(magic.data(), 4) || magic != std::array<char, 4>{'G','G','U','F'}) {
        return {StatusCode::Unsupported, "File does not have the GGUF magic"};
    }

    GgufFileInfo info;
    info.file_size_bytes = file_size;
    if (!reader.read_u32(info.version) || info.version < 1 || info.version > 3 ||
        !reader.read_u64(info.tensor_count) || !reader.read_u64(info.metadata_count)) {
        return {StatusCode::Unsupported, "Unsupported or truncated GGUF header"};
    }
    if (info.tensor_count == 0 || info.tensor_count > 1'000'000 ||
        info.metadata_count > 1'000'000) {
        return {StatusCode::InvalidArgument, "GGUF header counts exceed safety limits"};
    }

    for (uint64_t i = 0; i < info.metadata_count; ++i) {
        std::string key;
        uint32_t type = 0;
        GgufMetadataValue value;
        if (!reader.read_string(key) || key.empty() || !reader.read_u32(type) ||
            !read_metadata_value(reader, key, type, value)) {
            return {StatusCode::InvalidArgument, "Malformed GGUF metadata entry"};
        }
        if (!info.metadata.emplace(std::move(key), std::move(value)).second) {
            return {StatusCode::InvalidArgument, "GGUF metadata contains a duplicate key"};
        }
    }

    const uint64_t alignment_value = metadata_as<uint64_t>(info, "general.alignment", 32);
    if (alignment_value == 0 || alignment_value > 4096 ||
        (alignment_value & (alignment_value - 1)) != 0) {
        return {StatusCode::InvalidArgument, "GGUF general.alignment must be a power of two up to 4096"};
    }
    info.alignment = static_cast<uint32_t>(alignment_value);
    info.architecture = metadata_as<std::string>(info, "general.architecture", "unknown");
    info.model_name = metadata_as<std::string>(info, "general.name", "unknown");
    info.tokenizer.model = metadata_as<std::string>(info, "tokenizer.ggml.model", "unknown");
    info.tokenizer.pre_tokenizer = metadata_as<std::string>(info, "tokenizer.ggml.pre", "unknown");
    info.tokenizer.bos_token_id = metadata_as_i64(info, "tokenizer.ggml.bos_token_id", -1);
    info.tokenizer.eos_token_id = metadata_as_i64(info, "tokenizer.ggml.eos_token_id", -1);
    info.tokenizer.unknown_token_id = metadata_as_i64(info, "tokenizer.ggml.unknown_token_id", -1);
    info.tokenizer.padding_token_id = metadata_as_i64(info, "tokenizer.ggml.padding_token_id", -1);
    info.tokenizer.add_bos_token = metadata_as<bool>(info, "tokenizer.ggml.add_bos_token", false);
    info.tokenizer.add_eos_token = metadata_as<bool>(info, "tokenizer.ggml.add_eos_token", false);
    info.tokenizer.add_space_prefix = metadata_as<bool>(info, "tokenizer.ggml.add_space_prefix", false);
    info.tokenizer.remove_extra_whitespaces = metadata_as<bool>(info, "tokenizer.ggml.remove_extra_whitespaces", false);
    info.tokenizer.escape_whitespaces = metadata_as<bool>(info, "tokenizer.ggml.escape_whitespaces", true);
    info.tokenizer.byte_fallback = metadata_as<bool>(info, "tokenizer.ggml.byte_fallback", false);
    if (const auto found = info.metadata.find("tokenizer.ggml.tokens"); found != info.metadata.end()) {
        if (const auto* values = std::get_if<std::vector<std::string>>(&found->second)) info.tokenizer.tokens = *values;
    }
    if (const auto found = info.metadata.find("tokenizer.ggml.merges"); found != info.metadata.end()) {
        if (const auto* values = std::get_if<std::vector<std::string>>(&found->second)) info.tokenizer.merges = *values;
    }
    if (const auto found = info.metadata.find("tokenizer.ggml.scores"); found != info.metadata.end()) {
        if (const auto* values = std::get_if<std::vector<double>>(&found->second)) info.tokenizer.scores = *values;
    }
    if (const auto found = info.metadata.find("tokenizer.ggml.token_type"); found != info.metadata.end()) {
        if (const auto* values = std::get_if<std::vector<int64_t>>(&found->second)) info.tokenizer.token_types = *values;
    }
    if ((!info.tokenizer.scores.empty() && info.tokenizer.scores.size() != info.tokenizer.tokens.size()) ||
        (!info.tokenizer.token_types.empty() && info.tokenizer.token_types.size() != info.tokenizer.tokens.size())) {
        return {StatusCode::InvalidArgument, "GGUF tokenizer metadata arrays have inconsistent vocabulary lengths"};
    }
    for (double score : info.tokenizer.scores) {
        if (!std::isfinite(score)) return {StatusCode::InvalidArgument, "GGUF tokenizer contains a non-finite token score"};
    }
    const auto valid_token_id = [&info](int64_t id) {
        return id == -1 || (id >= 0 && static_cast<uint64_t>(id) < info.tokenizer.tokens.size());
    };
    if (!valid_token_id(info.tokenizer.bos_token_id) ||
        !valid_token_id(info.tokenizer.eos_token_id) ||
        !valid_token_id(info.tokenizer.unknown_token_id) ||
        !valid_token_id(info.tokenizer.padding_token_id)) {
        return {StatusCode::InvalidArgument, "GGUF tokenizer special token id is outside the vocabulary"};
    }

    std::set<std::string> tensor_names;
    info.tensors.reserve(static_cast<size_t>(info.tensor_count));
    for (uint64_t i = 0; i < info.tensor_count; ++i) {
        GgufTensorInfo tensor;
        uint32_t dimensions = 0;
        if (!reader.read_string(tensor.name) || tensor.name.empty() ||
            !tensor_names.insert(tensor.name).second ||
            !reader.read_u32(dimensions) || dimensions == 0 || dimensions > 8) {
            return {StatusCode::InvalidArgument, "Malformed GGUF tensor descriptor"};
        }
        tensor.dimensions.resize(dimensions);
        for (uint64_t& dimension : tensor.dimensions) {
            if (!reader.read_u64(dimension) || dimension == 0) {
                return {StatusCode::InvalidArgument, "GGUF tensor contains an invalid dimension"};
            }
        }
        if (!reader.read_u32(tensor.type_id) || !reader.read_u64(tensor.relative_offset)) {
            return {StatusCode::InvalidArgument, "GGUF tensor descriptor is truncated"};
        }
        tensor.type = tensor_type_info(tensor.type_id);
        info.tensors.push_back(std::move(tensor));
    }

    if (!align_up(reader.position(), info.alignment, info.data_section_offset) ||
        info.data_section_offset > file_size) {
        return {StatusCode::InvalidArgument, "GGUF tensor data section offset is invalid"};
    }
    const uint64_t data_bytes = file_size - info.data_section_offset;
    std::vector<size_t> offset_order(info.tensors.size());
    for (size_t i = 0; i < offset_order.size(); ++i) offset_order[i] = i;
    std::sort(offset_order.begin(), offset_order.end(), [&](size_t a, size_t b) {
        return info.tensors[a].relative_offset < info.tensors[b].relative_offset;
    });

    uint64_t previous_end = 0;
    for (size_t sorted_index : offset_order) {
        GgufTensorInfo& tensor = info.tensors[sorted_index];
        if (tensor.relative_offset > data_bytes ||
            tensor.relative_offset > std::numeric_limits<uint64_t>::max() - info.data_section_offset ||
            (tensor.relative_offset % info.alignment) != 0) {
            return {StatusCode::InvalidArgument, "GGUF tensor offset is outside the data section or violates alignment"};
        }
        tensor.absolute_offset = info.data_section_offset + tensor.relative_offset;

        uint64_t elements = 0;
        if (!checked_product(tensor.dimensions, elements)) {
            return {StatusCode::InvalidArgument, "GGUF tensor element count overflows"};
        }
        if (tensor.type.storage_size_known) {
            if (elements % tensor.type.block_elements != 0) {
                return {StatusCode::InvalidArgument, "GGUF tensor element count is not divisible by its storage block"};
            }
            const uint64_t blocks = elements / tensor.type.block_elements;
            if (blocks > std::numeric_limits<uint64_t>::max() / tensor.type.bytes_per_block) {
                return {StatusCode::InvalidArgument, "GGUF tensor byte size overflows"};
            }
            tensor.byte_size = blocks * tensor.type.bytes_per_block;
            tensor.byte_size_known = true;
            if (tensor.byte_size > data_bytes - tensor.relative_offset) {
                return {StatusCode::InvalidArgument, "GGUF tensor storage extends beyond end of file"};
            }
            if (tensor.relative_offset < previous_end) {
                return {StatusCode::InvalidArgument, "GGUF tensor storage ranges overlap"};
            }
            previous_end = tensor.relative_offset + tensor.byte_size;
        }
    }

    out.path_ = path;
    out.info_ = std::move(info);
    out.file_stream_ = std::move(stream);
    out.tensor_indices_.reserve(out.info_.tensors.size());
    for (size_t index = 0; index < out.info_.tensors.size(); ++index)
        out.tensor_indices_.emplace(out.info_.tensors[index].name, index);
    return Status::success();
}

const GgufTensorInfo* GgufReader::find_tensor(const std::string& name) const noexcept {
    const auto found = tensor_indices_.find(name);
    if (found == tensor_indices_.end() || found->second >= info_.tensors.size()) return nullptr;
    return &info_.tensors[found->second];
}

Status GgufReader::read_tensor_chunk(const GgufTensorInfo& tensor, uint64_t tensor_offset,
                                     void* destination, uint64_t size_bytes) const {
    const GgufTensorInfo* canonical = find_tensor(tensor.name);
    if (path_.empty() || !canonical || canonical->absolute_offset != tensor.absolute_offset ||
        canonical->relative_offset != tensor.relative_offset ||
        canonical->byte_size != tensor.byte_size || !canonical->byte_size_known ||
        !destination || size_bytes == 0 || tensor_offset > canonical->byte_size ||
        size_bytes > canonical->byte_size - tensor_offset) {
        return {StatusCode::InvalidArgument, "Requested GGUF tensor chunk is invalid, foreign, or has unknown storage size"};
    }
    if (tensor.absolute_offset > std::numeric_limits<uint64_t>::max() - tensor_offset) {
        return {StatusCode::InvalidArgument, "GGUF tensor read offset overflows"};
    }
    const uint64_t absolute = canonical->absolute_offset + tensor_offset;
    if (absolute > info_.file_size_bytes || size_bytes > info_.file_size_bytes - absolute ||
        absolute > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max()) ||
        size_bytes > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
        return {StatusCode::InvalidArgument, "GGUF tensor read range exceeds file bounds"};
    }
    if (!file_stream_)
        return {StatusCode::NotReady, "GGUF reader has no retained file stream"};
    std::lock_guard<std::mutex> lock(file_stream_->mutex);
    std::ifstream& input = file_stream_->input;
    input.clear();
    input.seekg(static_cast<std::streamoff>(absolute), std::ios::beg);
    if (!input || !input.read(static_cast<char*>(destination), static_cast<std::streamsize>(size_bytes))) {
        return {StatusCode::BackendFailure, "GGUF tensor chunk read failed"};
    }
    return Status::success();
}

} // namespace ai_chat::gpu_engine
