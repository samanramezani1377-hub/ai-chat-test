#include "tokenizer.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace ai_chat::gpu_engine {

std::string TokenizerRegistry::make_key(const std::string& model,
                                        const std::string& pre_tokenizer) {
    return model + "\n" + pre_tokenizer;
}

Status TokenizerRegistry::register_adapter(std::shared_ptr<TokenizerAdapter> adapter) {
    if (!adapter) return {StatusCode::InvalidArgument, "Cannot register a null tokenizer adapter"};
    const auto& descriptor = adapter->descriptor();
    if (descriptor.model.empty() || descriptor.pre_tokenizer.empty() ||
        descriptor.algorithm.empty()) {
        return {StatusCode::InvalidArgument, "Tokenizer adapter requires model, pre-tokenizer and algorithm identities"};
    }
    const std::string key = make_key(descriptor.model, descriptor.pre_tokenizer);
    if (adapters_.find(key) != adapters_.end())
        return {StatusCode::InvalidArgument, "Tokenizer adapter is already registered for this model/pre-tokenizer pair"};
    adapters_.emplace(key, std::move(adapter));
    return Status::success();
}

std::shared_ptr<const TokenizerAdapter> TokenizerRegistry::find(
        const GgufTokenizerInfo& metadata) const {
    const auto found = adapters_.find(make_key(metadata.model, metadata.pre_tokenizer));
    return found == adapters_.end() ? nullptr : found->second;
}

std::vector<TokenizerDescriptor> TokenizerRegistry::registered_adapters() const {
    std::vector<TokenizerDescriptor> result;
    result.reserve(adapters_.size());
    for (const auto& entry : adapters_) result.push_back(entry.second->descriptor());
    std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
        if (a.model != b.model) return a.model < b.model;
        return a.pre_tokenizer < b.pre_tokenizer;
    });
    return result;
}

TokenizerPreflightReport evaluate_tokenizer_preflight(
        const GgufTokenizerInfo& metadata, const TokenizerRegistry& registry) {
    TokenizerPreflightReport report;
    const bool model_known = !metadata.model.empty() && metadata.model != "unknown";
    const bool pretokenizer_known = !metadata.pre_tokenizer.empty() &&
                                    metadata.pre_tokenizer != "unknown";
    if (!model_known) report.blockers.emplace_back("TOKENIZER_MODEL_UNKNOWN");
    if (!pretokenizer_known) report.blockers.emplace_back("TOKENIZER_PRETOKENIZER_UNKNOWN");
    if (metadata.tokens.empty()) report.blockers.emplace_back("TOKENIZER_VOCABULARY_EMPTY");
    if (metadata.tokens.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max()))
        report.blockers.emplace_back("TOKENIZER_VOCABULARY_EXCEEDS_TOKEN_ID_RANGE");
    if (!metadata.scores.empty() && metadata.scores.size() != metadata.tokens.size())
        report.blockers.emplace_back("TOKENIZER_SCORE_COUNT_MISMATCH");
    if (!metadata.token_types.empty() && metadata.token_types.size() != metadata.tokens.size())
        report.blockers.emplace_back("TOKENIZER_TOKEN_TYPE_COUNT_MISMATCH");

    const auto valid_special_id = [&metadata](int64_t id) {
        return id == -1 || (id >= 0 && static_cast<uint64_t>(id) < metadata.tokens.size());
    };
    if (!valid_special_id(metadata.bos_token_id) ||
        !valid_special_id(metadata.eos_token_id) ||
        !valid_special_id(metadata.unknown_token_id) ||
        !valid_special_id(metadata.padding_token_id)) {
        report.blockers.emplace_back("TOKENIZER_SPECIAL_TOKEN_ID_OUT_OF_RANGE");
    }
    report.metadata_valid = report.blockers.empty();

    const auto adapter = registry.find(metadata);
    report.adapter_registered = static_cast<bool>(adapter);
    if (!adapter) {
        report.blockers.emplace_back("TOKENIZER_ADAPTER_NOT_REGISTERED");
        return report;
    }
    report.adapter_validated = adapter->descriptor().validation == SupportState::Validated;
    if (!report.adapter_validated)
        report.blockers.emplace_back("TOKENIZER_ADAPTER_UNVERIFIED");
    if (report.metadata_valid) {
        const Status status = adapter->validate_metadata(metadata);
        if (!status.ok()) report.blockers.emplace_back("TOKENIZER_ADAPTER_REJECTED_METADATA");
    }
    report.ready = report.metadata_valid && report.adapter_validated &&
                   report.blockers.empty();
    return report;
}

Status TokenizerEngine::encode(const GgufTokenizerInfo& metadata, const std::string& text,
                               std::vector<uint32_t>& token_ids) const {
    token_ids.clear();
    const auto report = evaluate_tokenizer_preflight(metadata, registry_);
    if (!report.ready)
        return {StatusCode::Unsupported, "Tokenizer is not validated for this GGUF model/pre-tokenizer"};
    const auto adapter = registry_.find(metadata);
    Status status = adapter->encode(metadata, text, token_ids);
    if (!status.ok()) token_ids.clear();
    if (status.ok()) {
        for (uint32_t id : token_ids) {
            if (id >= metadata.tokens.size()) {
                token_ids.clear();
                return {StatusCode::NumericalFailure, "Tokenizer adapter emitted an out-of-vocabulary token id"};
            }
        }
    }
    return status;
}

Status TokenizerEngine::decode(const GgufTokenizerInfo& metadata,
                               const std::vector<uint32_t>& token_ids,
                               std::string& text) const {
    text.clear();
    const auto report = evaluate_tokenizer_preflight(metadata, registry_);
    if (!report.ready)
        return {StatusCode::Unsupported, "Tokenizer is not validated for this GGUF model/pre-tokenizer"};
    for (uint32_t id : token_ids) {
        if (id >= metadata.tokens.size())
            return {StatusCode::InvalidArgument, "Decode token id is outside the GGUF vocabulary"};
    }
    return registry_.find(metadata)->decode(metadata, token_ids, text);
}

} // namespace ai_chat::gpu_engine
