#include "universal_adapter_bridge.h"

#include <sstream>

namespace ai_chat::gpu_engine {

DeviceInfo device_info_from_universal_adapter(
        const ai_chat::gpu::UniversalGpuCapabilities& capabilities) {
    DeviceInfo result;
    result.backend = BackendKind::OpenGles;
    result.name = capabilities.renderer.empty() ? "unknown" : capabilities.renderer;
    result.vendor = capabilities.vendor.empty() ? "unknown" : capabilities.vendor;
    result.api_version = capabilities.version.empty() ? "unknown" : capabilities.version;
    result.compute_available = capabilities.gles_compute_supported;
    result.compute_self_test_passed = capabilities.gles_compute_self_test_passed;

    // A probe/self-test is deliberately not promoted to backend or inference readiness.
    result.inference_ready = false;
    result.state = capabilities.gles_compute_self_test_passed
        ? CapabilityState::SelfTested
        : (capabilities.egl_context_created ? CapabilityState::Discovered
                                            : CapabilityState::Unknown);

    result.limits.max_storage_buffer_binding_size =
        capabilities.max_shader_storage_block_bytes > 0
            ? static_cast<uint64_t>(capabilities.max_shader_storage_block_bytes) : 0;
    result.limits.max_compute_workgroup_size_x =
        capabilities.max_compute_workgroup_size_x > 0
            ? static_cast<uint32_t>(capabilities.max_compute_workgroup_size_x) : 0;
    result.limits.max_compute_workgroup_size_y =
        capabilities.max_compute_workgroup_size_y > 0
            ? static_cast<uint32_t>(capabilities.max_compute_workgroup_size_y) : 0;
    result.limits.max_compute_workgroup_size_z =
        capabilities.max_compute_workgroup_size_z > 0
            ? static_cast<uint32_t>(capabilities.max_compute_workgroup_size_z) : 0;
    result.limits.max_compute_workgroup_invocations =
        capabilities.max_compute_invocations > 0
            ? static_cast<uint32_t>(capabilities.max_compute_invocations) : 0;
    result.limits.max_compute_workgroup_storage_bytes =
        capabilities.max_compute_shared_memory_bytes > 0
            ? static_cast<uint32_t>(capabilities.max_compute_shared_memory_bytes) : 0;

    std::ostringstream diagnostic;
    diagnostic << "source=UniversalGpuAdapter"
               << " eglContext=" << (capabilities.egl_context_created ? "1" : "0")
               << " glesCompute=" << (capabilities.gles_compute_supported ? "1" : "0")
               << " selfTest=" << (capabilities.gles_compute_self_test_passed ? "passed" : "not_passed")
               << " inferenceReady=0"
               << " reason=capability_probe_only";
    if (!capabilities.failure_reason.empty()) {
        diagnostic << " failure=" << capabilities.failure_reason;
    }
    result.diagnostic = diagnostic.str();
    return result;
}

} // namespace ai_chat::gpu_engine
