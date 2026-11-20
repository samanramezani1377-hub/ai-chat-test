#pragma once

#include <cstdint>
#include <string>

namespace ai_chat::gpu {

// Runtime-only, vendor-neutral OpenGL ES capability snapshot.
// This probe is deliberately independent from the UI and from GGML backend
// registration. A successful probe means GLES compute is available; it does
// not claim that every GGML operation or data type has a compatible kernel.
struct UniversalGpuCapabilities {
    bool egl_display_available = false;
    bool egl_context_created = false;
    bool gles_compute_supported = false;
    bool gles_fp16_shader_hint = false;
    int gles_major = 0;
    int gles_minor = 0;
    int egl_major = 0;
    int egl_minor = 0;
    int extension_count = 0;
    int max_compute_invocations = 0;
    int max_compute_shared_memory_bytes = 0;
    int max_compute_workgroup_count_x = 0;
    int max_compute_workgroup_count_y = 0;
    int max_compute_workgroup_count_z = 0;
    int max_compute_workgroup_size_x = 0;
    int max_compute_workgroup_size_y = 0;
    int max_compute_workgroup_size_z = 0;
    int64_t max_shader_storage_block_bytes = 0;
    std::string vendor;
    std::string renderer;
    std::string version;
    std::string shading_language_version;
    std::string egl_vendor;
    std::string egl_version;
    std::string failure_reason;

    // A single-line, log-safe report suitable for the existing native trace.
    std::string diagnostic_line() const;
};

// Creates a temporary 1x1 EGL pbuffer + OpenGL ES context, reads real driver
// capabilities, then releases every EGL/GL object before returning.
UniversalGpuCapabilities probe_universal_gpu_adapter();

} // namespace ai_chat::gpu
