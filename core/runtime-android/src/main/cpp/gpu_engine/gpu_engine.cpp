#include "gpu_engine.h"

#include <algorithm>
#include <utility>
#include "execution_engine.h"

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


Status GpuEngine::register_backend(std::shared_ptr<Backend> backend) {
    if (active_backend_) return {StatusCode::NotReady, "Cannot change backend registry while the engine is initialized"};
    return registry_.register_backend(std::move(backend));
}

Status GpuEngine::initialize(BackendKind preferred_backend) {
    if (active_backend_) return Status::success();
    active_device_ = DeviceInfo{};
    diagnostics_ = RuntimeDiagnostics{};

    active_backend_ = registry_.find(preferred_backend);
    if (!active_backend_) {
        diagnostics_.last_error = "Requested GPU backend is not registered";
        return {StatusCode::Unsupported, diagnostics_.last_error};
    }

    std::vector<DeviceInfo> devices;
    Status status = active_backend_->enumerate_devices(devices);
    if (!status.ok()) {
        diagnostics_.last_error = status.message;
        active_backend_->shutdown();
        active_backend_.reset();
        return status;
    }
    const auto selected = std::find_if(devices.begin(), devices.end(),
        [](const DeviceInfo& device) { return device.compute_available; });
    if (selected == devices.end()) {
        diagnostics_.last_error = "No compute-capable device was discovered by the selected backend";
        active_backend_->shutdown();
        active_backend_.reset();
        return {StatusCode::Unsupported, diagnostics_.last_error};
    }

    active_device_ = *selected;
    status = active_backend_->initialize(active_device_);
    if (!status.ok() || !active_backend_->healthy()) {
        diagnostics_.last_error = status.ok() ? "Backend did not remain healthy after initialization" : status.message;
        active_backend_->shutdown();
        active_backend_.reset();
        return status.ok() ? Status{StatusCode::DeviceLost, diagnostics_.last_error} : status;
    }

    active_device_.state = CapabilityState::BackendReady;
    active_device_.inference_ready = false;
    diagnostics_.selected_backend = active_device_.backend;
    diagnostics_.capability_state = active_device_.state;
    diagnostics_.device_name = active_device_.name.empty() ? "unknown" : active_device_.name;
    diagnostics_.last_error.clear();
    return Status::success();
}

Status GpuEngine::initialize_with_fallback(const std::vector<BackendKind>& preference_order) {
    if (preference_order.empty())
        return {StatusCode::InvalidArgument, "GPU backend preference order is empty"};
    if (active_backend_) return Status::success();
    std::vector<std::string> failures;
    Status last_status{StatusCode::Unsupported, "No GPU backend preference was attempted"};
    for (BackendKind kind : preference_order) {
        last_status = initialize(kind);
        if (last_status.ok()) {
            if (!failures.empty()) {
                diagnostics_.fallback_used = true;
                diagnostics_.fallback_reason = failures.front();
            }
            return Status::success();
        }
        failures.emplace_back((kind == BackendKind::DawnWebGpu ? "DawnWebGPU" : "OpenGLES") + std::string(": ") + last_status.message);
    }
    diagnostics_.last_error = "No preferred GPU backend initialized";
    for (const auto& failure : failures) diagnostics_.last_error += " | " + failure;
    return {last_status.code, diagnostics_.last_error};
}
Status GpuEngine::execute(const ComputeGraph& graph, const std::atomic<bool>* cancellation) {
    if (!active_backend_) return {StatusCode::NotReady, "GPU engine is not initialized"};
    ExecutionScheduler scheduler;
    Status status = scheduler.execute(*active_backend_, graph, cancellation);
    if (!status.ok()) diagnostics_.last_error = status.message;
    return status;
}

Status GpuEngine::shutdown() {
    if (!active_backend_) return Status::success();
    Status status = active_backend_->shutdown();
    active_backend_.reset();
    active_device_ = DeviceInfo{};
    diagnostics_.selected_backend = BackendKind::Unknown;
    diagnostics_.capability_state = CapabilityState::Unknown;
    diagnostics_.device_name = "unknown";
    if (!status.ok()) diagnostics_.last_error = status.message;
    return status;
}

bool GpuEngine::ready() const noexcept {
    return active_backend_ && active_backend_->healthy();
}

const DeviceInfo* GpuEngine::active_device() const noexcept {
    return active_backend_ ? &active_device_ : nullptr;
}

const RuntimeDiagnostics& GpuEngine::diagnostics() const noexcept {
    return diagnostics_;
}

} // namespace ai_chat::gpu_engine
