#pragma once

#include <cstdint>
#include <string>
#include <functional>

namespace ai_chat::gpu {

// Runtime-only, vendor-neutral OpenGL ES capability snapshot.
// This probe is deliberately independent from the UI and from GGML backend
// registration. A successful probe means GLES compute is available; it does
// not claim that every GGML operation or data type has a compatible kernel.
struct UniversalGpuCapabilities {
    bool egl_display_available = false;
    bool egl_context_created = false;
    bool gles_compute_supported = false;
    // True only after a real GLES compute shader dispatch and SSBO readback pass.
    bool gles_compute_self_test_passed = false;
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

// Starts a process-lifetime GLES compute context on a dedicated owner thread.
// The context remains current on that thread after the capability/self-test pass;
// no GL call is made from the caller or Android UI/rendering thread.
UniversalGpuCapabilities probe_universal_gpu_adapter();

// Explicitly stops the dedicated context thread. Normally process lifetime owns
// this engine; call only when the native backend itself is being permanently torn down.
// Executes GLES work on the dedicated thread that owns the persistent EGL context.
// The callback must not retain GL thread-local assumptions or recursively enqueue work.
bool run_universal_gpu_task(const std::function<void()>& task, std::string& error);

void shutdown_universal_gpu_adapter();

} // namespace ai_chat::gpu
