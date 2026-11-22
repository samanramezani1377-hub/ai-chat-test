#pragma once

#include "model_registry.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace ai_chat::gpu_engine {

struct TokenizerDescriptor {
    std::string model;
    std::string pre_tokenizer;
    std::string algorithm;
    SupportState validation = SupportState::Unverified;
};

struct TokenizerPreflightReport {
    bool metadata_valid = false;
    bool adapter_registered = false;
    bool adapter_validated = false;
    bool ready = false;
    std::vector<std::string> blockers;
};

class TokenizerAdapter {
public:
    virtual ~TokenizerAdapter() = default;
    virtual const TokenizerDescriptor& descriptor() const noexcept = 0;
    virtual Status validate_metadata(const GgufTokenizerInfo& metadata) const = 0;
    virtual Status encode(const GgufTokenizerInfo& metadata, const std::string& text,
                          std::vector<uint32_t>& token_ids) const = 0;
    virtual Status decode(const GgufTokenizerInfo& metadata,
                          const std::vector<uint32_t>& token_ids,
                          std::string& text) const = 0;
};

class TokenizerRegistry {
public:
    Status register_adapter(std::shared_ptr<TokenizerAdapter> adapter);
    std::shared_ptr<const TokenizerAdapter> find(const GgufTokenizerInfo& metadata) const;
    std::vector<TokenizerDescriptor> registered_adapters() const;
private:
    static std::string make_key(const std::string& model, const std::string& pre_tokenizer);
    std::unordered_map<std::string, std::shared_ptr<TokenizerAdapter>> adapters_;
};

TokenizerPreflightReport evaluate_tokenizer_preflight(
    const GgufTokenizerInfo& metadata, const TokenizerRegistry& registry);
// Stable, delimiter-safe metadata summary for device-side diagnostics. This does
// not imply that an adapter is supported or validated.
std::string describe_tokenizer_metadata(
    const GgufTokenizerInfo& metadata, const TokenizerRegistry& registry);
// Optional exact Unicode property classifications supplied by the Android ICU-backed
// Java Character API for Qwen3.5's Unicode regex pre-tokenizer. Values map to
// LetterMark=0, Number=1, Space=2, Newline=3, Other=4.
using Qwen35UnicodeCategories = std::unordered_map<uint32_t, uint8_t>;
void set_qwen35_unicode_categories(const Qwen35UnicodeCategories* categories) noexcept;

Status register_builtin_tokenizer_adapters(TokenizerRegistry& registry);

class TokenizerEngine {
public:
    explicit TokenizerEngine(const TokenizerRegistry& registry) : registry_(registry) {}
    Status encode(const GgufTokenizerInfo& metadata, const std::string& text,
                  std::vector<uint32_t>& token_ids) const;
    Status decode(const GgufTokenizerInfo& metadata,
                  const std::vector<uint32_t>& token_ids, std::string& text) const;
private:
    Status ensure_ready(const GgufTokenizerInfo& metadata) const;
    const TokenizerRegistry& registry_;
    // A model's GGUF tokenizer metadata is immutable for the lifetime of a
    // TokenizerEngine. Cache its expensive structural preflight across decode
    // steps instead of rescanning a 248k-token vocabulary for every output token.
    mutable const GgufTokenizerInfo* validated_metadata_ = nullptr;
    mutable bool metadata_ready_ = false;
};

} // namespace ai_chat::gpu_engine
