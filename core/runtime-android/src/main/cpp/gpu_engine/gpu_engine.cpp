#include "gpu_engine.h"

#include <algorithm>
#include <utility>

namespace ai_chat::gpu_engine {

Status BackendRegistry::register_backend(std::shared_ptr<Backend> backend) {
    if (!backend) {
        return {StatusCode::InvalidArgument, "Cannot register a null GPU backend"};
    }
    const BackendKind kind = backend->kind();
    if (kind == BackendKind::Unknown) {
        return {StatusCode::InvalidArgument, "Cannot register a backend with unknown kind"};
    }
    const auto duplicate = std::find_if(backends_.begin(), backends_.end(),
        [kind](const std::shared_ptr<Backend>& current) {
            return current && current->kind() == kind;
        });
    if (duplicate != backends_.end()) {
        return {StatusCode::InvalidArgument, "A backend of this kind is already registered"};
    }
    backends_.push_back(std::move(backend));
    return Status::success();
}

std::vector<BackendKind> BackendRegistry::registered_backends() const {
    std::vector<BackendKind> result;
    result.reserve(backends_.size());
    for (const auto& backend : backends_) {
        if (backend) result.push_back(backend->kind());
    }
    return result;
}

std::shared_ptr<Backend> BackendRegistry::find(BackendKind kind) const {
    const auto found = std::find_if(backends_.begin(), backends_.end(),
        [kind](const std::shared_ptr<Backend>& backend) {
            return backend && backend->kind() == kind;
        });
    return found == backends_.end() ? nullptr : *found;
}

} // namespace ai_chat::gpu_engine
