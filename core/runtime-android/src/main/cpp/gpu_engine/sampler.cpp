#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>

namespace ai_chat::gpu_engine {
namespace {

bool valid_config(const SamplingConfig& config) {
    return std::isfinite(config.temperature) && config.temperature >= 0.0f &&
           std::isfinite(config.top_p) && config.top_p > 0.0f && config.top_p <= 1.0f &&
           std::isfinite(config.repetition_penalty) && config.repetition_penalty > 0.0f;
}

} // namespace

Sampler::Sampler(SamplingConfig config) : config_(config), random_(config.seed) {}

Status Sampler::set_config(SamplingConfig config) {
    if (!valid_config(config)) return {StatusCode::InvalidArgument, "Invalid sampling configuration"};
    config_ = config;
    random_.seed(config.seed);
    return Status::success();
}

Status Sampler::sample(const std::vector<float>& logits,
                       const std::vector<int32_t>& history,
                       int32_t& out_token) {
    out_token = -1;
    if (logits.empty() || logits.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        return {StatusCode::InvalidArgument, "Sampler received an empty or oversized logits vector"};
    }
    if (!valid_config(config_)) return {StatusCode::InvalidArgument, "Sampling configuration is invalid"};
    for (float value : logits) {
        if (!std::isfinite(value)) return {StatusCode::NumericalFailure, "Logits contain NaN or infinity"};
    }

    std::vector<float> adjusted = logits;
    if (config_.repetition_penalty != 1.0f) {
        std::vector<uint8_t> seen(logits.size(), 0);
        for (int32_t token : history) {
            if (token < 0 || static_cast<size_t>(token) >= adjusted.size() || seen[static_cast<size_t>(token)]) continue;
            seen[static_cast<size_t>(token)] = 1;
            float& value = adjusted[static_cast<size_t>(token)];
            value = value < 0.0f ? value * config_.repetition_penalty
                                 : value / config_.repetition_penalty;
        }
    }

    if (config_.temperature == 0.0f || config_.top_k == 1) {
        const auto best = std::max_element(adjusted.begin(), adjusted.end());
        out_token = static_cast<int32_t>(std::distance(adjusted.begin(), best));
        return Status::success();
    }

    std::vector<size_t> order(adjusted.size());
    std::iota(order.begin(), order.end(), 0);
    if (config_.top_k > 0 && config_.top_k < order.size()) {
        std::partial_sort(order.begin(), order.begin() + config_.top_k, order.end(),
            [&adjusted](size_t a, size_t b) { return adjusted[a] > adjusted[b]; });
        order.resize(config_.top_k);
    }

    const float maximum = *std::max_element(order.begin(), order.end(),
        [&adjusted](size_t a, size_t b) { return adjusted[a] < adjusted[b]; }) < adjusted.size()
        ? adjusted[*std::max_element(order.begin(), order.end(),
            [&adjusted](size_t a, size_t b) { return adjusted[a] < adjusted[b]; })] : 0.0f;

    std::vector<double> probabilities(logits.size(), 0.0);
    double total = 0.0;
    for (size_t index : order) {
        const double probability = std::exp(static_cast<double>((adjusted[index] - maximum) / config_.temperature));
        probabilities[index] = probability;
        total += probability;
    }
    if (!std::isfinite(total) || total <= 0.0) {
        return {StatusCode::NumericalFailure, "Softmax probability normalization failed"};
    }
    for (double& probability : probabilities) probability /= total;

    if (config_.top_p < 1.0f) {
        std::vector<size_t> ranked = order;
        std::sort(ranked.begin(), ranked.end(),
            [&probabilities](size_t a, size_t b) { return probabilities[a] > probabilities[b]; });
        double cumulative = 0.0;
        std::vector<uint8_t> keep(probabilities.size(), 0);
        for (size_t index : ranked) {
            keep[index] = 1;
            cumulative += probabilities[index];
            if (cumulative >= config_.top_p) break;
        }
        total = 0.0;
        for (size_t i = 0; i < probabilities.size(); ++i) {
            if (!keep[i]) probabilities[i] = 0.0;
            total += probabilities[i];
        }
        if (total <= 0.0 || !std::isfinite(total)) {
            return {StatusCode::NumericalFailure, "Top-p filtering removed all candidate tokens"};
        }
        for (double& probability : probabilities) probability /= total;
    }

    std::discrete_distribution<size_t> distribution(probabilities.begin(), probabilities.end());
    out_token = static_cast<int32_t>(distribution(random_));
    return Status::success();
}

StopController::StopController(int32_t eos_token_id,
                               std::vector<std::vector<int32_t>> stop_sequences)
    : eos_token_id_(eos_token_id), stop_sequences_(std::move(stop_sequences)) {
    for (const auto& sequence : stop_sequences_) {
        maximum_stop_length_ = std::max(maximum_stop_length_, sequence.size());
    }
}

void StopController::reset() { generated_.clear(); }

bool StopController::push(int32_t token) {
    generated_.push_back(token);
    if (eos_token_id_ >= 0 && token == eos_token_id_) return true;
    if (maximum_stop_length_ == 0) return false;
    const size_t available = std::min(maximum_stop_length_, generated_.size());
    for (const auto& sequence : stop_sequences_) {
        if (sequence.empty() || sequence.size() > available) continue;
        if (std::equal(sequence.rbegin(), sequence.rend(), generated_.rbegin())) return true;
    }
    return false;
}

const std::vector<int32_t>& StopController::generated_tokens() const noexcept { return generated_; }

} // namespace ai_chat::gpu_engine
