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

class TokenizerEngine {
public:
    explicit TokenizerEngine(const TokenizerRegistry& registry) : registry_(registry) {}
    Status encode(const GgufTokenizerInfo& metadata, const std::string& text,
                  std::vector<uint32_t>& token_ids) const;
    Status decode(const GgufTokenizerInfo& metadata,
                  const std::vector<uint32_t>& token_ids, std::string& text) const;
private:
    const TokenizerRegistry& registry_;
};

} // namespace ai_chat::gpu_engine
