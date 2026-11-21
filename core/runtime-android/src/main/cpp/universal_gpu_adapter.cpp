#include "universal_gpu_adapter.h"

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <GLES2/gl2ext.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <condition_variable>
#include <deque>
#include <future>
#include <functional>
#include <mutex>
#include <thread>
#include <exception>
#include <sstream>
#include <string>

namespace ai_chat::gpu {
namespace {

std::string egl_error_name(EGLint error) {
    switch (error) {
        case EGL_SUCCESS: return "EGL_SUCCESS";
        case EGL_NOT_INITIALIZED: return "EGL_NOT_INITIALIZED";
        case EGL_BAD_ACCESS: return "EGL_BAD_ACCESS";
        case EGL_BAD_ALLOC: return "EGL_BAD_ALLOC";
        case EGL_BAD_ATTRIBUTE: return "EGL_BAD_ATTRIBUTE";
        case EGL_BAD_CONFIG: return "EGL_BAD_CONFIG";
        case EGL_BAD_CONTEXT: return "EGL_BAD_CONTEXT";
        case EGL_BAD_CURRENT_SURFACE: return "EGL_BAD_CURRENT_SURFACE";
        case EGL_BAD_DISPLAY: return "EGL_BAD_DISPLAY";
        case EGL_BAD_MATCH: return "EGL_BAD_MATCH";
        case EGL_BAD_NATIVE_PIXMAP: return "EGL_BAD_NATIVE_PIXMAP";
        case EGL_BAD_NATIVE_WINDOW: return "EGL_BAD_NATIVE_WINDOW";
        case EGL_BAD_PARAMETER: return "EGL_BAD_PARAMETER";
        case EGL_BAD_SURFACE: return "EGL_BAD_SURFACE";
        default: {
            std::ostringstream out;
            out << "EGL_ERROR_0x" << std::hex << error;
            return out.str();
        }
    }
}

std::string egl_string(EGLDisplay display, EGLint name) {
    const char *value = eglQueryString(display, name);
    return value ? value : "";
}

std::string gl_string(GLenum name) {
    const GLubyte *value = glGetString(name);
    return value ? reinterpret_cast<const char *>(value) : "";
}

// OpenGL ES 3.x requires indexed extension enumeration. GL_EXTENSIONS passed
// to glGetString is not a valid extension-query path on ES 3.0+ contexts.
std::string gl_extensions(int &count) {
    count = 0;
    GLint reported_count = 0;
    glGetIntegerv(GL_NUM_EXTENSIONS, &reported_count);
    if (glGetError() != GL_NO_ERROR || reported_count < 0) return {};

    std::string extensions;
    for (GLint i = 0; i < reported_count; ++i) {
        const GLubyte *value = glGetStringi(GL_EXTENSIONS, static_cast<GLuint>(i));
        if (!value) continue;
        if (!extensions.empty()) extensions.push_back(' ');
        extensions.append(reinterpret_cast<const char *>(value));
        ++count;
    }
    return extensions;
}

bool has_extension(const std::string &extensions, const char *name) {
    if (extensions.empty() || !name || !*name) return false;
    const std::string needle(name);
    size_t pos = 0;
    while ((pos = extensions.find(needle, pos)) != std::string::npos) {
        const bool left = pos == 0 || extensions[pos - 1] == ' ';
        const size_t end = pos + needle.size();
        const bool right = end == extensions.size() || extensions[end] == ' ';
        if (left && right) return true;
        pos = end;
    }
    return false;
}

std::string safe_token(std::string value) {
    for (char &c : value) {
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '=') c = '_';
    }
    return value.empty() ? "unknown" : value;
}

bool run_gles_compute_self_test(std::string &failure_reason) {
    // A minimal integer-only compute kernel proves shader compilation, SSBO
    // binding, dispatch, barriers and GPU-to-CPU readback without assuming FP16.
    static constexpr char shader_source[] = R"glsl(#version 310 es
precision highp int;
layout(local_size_x = 4, local_size_y = 1, local_size_z = 1) in;
layout(std430, binding = 0) buffer SelfTestData {
    uint values[];
};
void main() {
    uint index = gl_GlobalInvocationID.x;
    values[index] = values[index] * 3u + 7u;
}
)glsl";

    GLuint shader = 0;
    GLuint program = 0;
    GLuint buffer = 0;
    auto cleanup = [&]() {
        if (buffer != 0) glDeleteBuffers(1, &buffer);
        if (program != 0) glDeleteProgram(program);
        if (shader != 0) glDeleteShader(shader);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        glUseProgram(0);
    };

    while (glGetError() != GL_NO_ERROR) {}
    shader = glCreateShader(GL_COMPUTE_SHADER);
    if (shader == 0) {
        failure_reason = "compute_self_test_shader_create_failed";
        cleanup();
        return false;
    }
    const GLchar *source = shader_source;
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    GLint compiled = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (compiled != GL_TRUE) {
        GLint log_length = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_length);
        std::string log(static_cast<size_t>(std::max(1, log_length)), '\0');
        GLsizei written = 0;
        glGetShaderInfoLog(shader, static_cast<GLsizei>(log.size()), &written, log.data());
        failure_reason = "compute_self_test_shader_compile_failed";
        if (written > 0) failure_reason += "_" + log.substr(0, static_cast<size_t>(written));
        cleanup();
        return false;
    }

    program = glCreateProgram();
    if (program == 0) {
        failure_reason = "compute_self_test_program_create_failed";
        cleanup();
        return false;
    }
    glAttachShader(program, shader);
    glLinkProgram(program);
    GLint linked = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (linked != GL_TRUE) {
        GLint log_length = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &log_length);
        std::string log(static_cast<size_t>(std::max(1, log_length)), '\0');
        GLsizei written = 0;
        glGetProgramInfoLog(program, static_cast<GLsizei>(log.size()), &written, log.data());
        failure_reason = "compute_self_test_program_link_failed";
        if (written > 0) failure_reason += "_" + log.substr(0, static_cast<size_t>(written));
        cleanup();
        return false;
    }

    constexpr std::array<uint32_t, 4> input = {1u, 2u, 3u, 4u};
    constexpr std::array<uint32_t, 4> expected = {10u, 13u, 16u, 19u};
    glGenBuffers(1, &buffer);
    if (buffer == 0) {
        failure_reason = "compute_self_test_ssbo_create_failed";
        cleanup();
        return false;
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(sizeof(input)),
        input.data(), GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffer);
    const GLenum setup_error = glGetError();
    if (setup_error != GL_NO_ERROR) {
        failure_reason = "compute_self_test_ssbo_setup_failed";
        cleanup();
        return false;
    }
    glUseProgram(program);
    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);

    const GLenum dispatch_error = glGetError();
    if (dispatch_error != GL_NO_ERROR) {
        failure_reason = "compute_self_test_dispatch_failed";
        cleanup();
        return false;
    }

    void *mapped = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0,
        static_cast<GLsizeiptr>(sizeof(input)), GL_MAP_READ_BIT);
    if (!mapped) {
        failure_reason = "compute_self_test_readback_map_failed";
        cleanup();
        return false;
    }
    std::array<uint32_t, 4> output{};
    std::memcpy(output.data(), mapped, sizeof(output));
    const GLboolean unmapped = glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    if (unmapped != GL_TRUE) {
        failure_reason = "compute_self_test_readback_unmap_failed";
        cleanup();
        return false;
    }
    cleanup();

    if (output != expected) {
        failure_reason = "compute_self_test_result_mismatch";
        return false;
    }
    const GLenum cleanup_error = glGetError();
    if (cleanup_error != GL_NO_ERROR) {
        failure_reason = "compute_self_test_cleanup_gl_error";
        return false;
    }
    failure_reason.clear();
    return true;
}

void read_compute_limits(UniversalGpuCapabilities &caps) {
    glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &caps.max_compute_invocations);
    glGetIntegerv(GL_MAX_COMPUTE_SHARED_MEMORY_SIZE, &caps.max_compute_shared_memory_bytes);
    glGetInteger64v(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &caps.max_shader_storage_block_bytes);

    GLint values[3] = {0, 0, 0};
    for (GLuint i = 0; i < 3; ++i) {
        glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, i, &values[i]);
    }
    caps.max_compute_workgroup_count_x = values[0];
    caps.max_compute_workgroup_count_y = values[1];
    caps.max_compute_workgroup_count_z = values[2];

    for (GLuint i = 0; i < 3; ++i) values[i] = 0;
    for (GLuint i = 0; i < 3; ++i) {
        glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, i, &values[i]);
    }
    caps.max_compute_workgroup_size_x = values[0];
    caps.max_compute_workgroup_size_y = values[1];
    caps.max_compute_workgroup_size_z = values[2];
}


class PersistentGlesProbe {
public:
    PersistentGlesProbe() = default;
    PersistentGlesProbe(const PersistentGlesProbe &) = delete;
    PersistentGlesProbe & operator=(const PersistentGlesProbe &) = delete;
    ~PersistentGlesProbe() { shutdown(); }

    UniversalGpuCapabilities initialize() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!worker_.joinable()) {
            stopping_ = false;
            startup_complete_ = false;
            try {
                worker_ = std::thread(&PersistentGlesProbe::run, this);
            } catch (const std::exception &) {
                capabilities_.failure_reason = "gles_owner_thread_start_failed";
                startup_complete_ = true;
            } catch (...) {
                capabilities_.failure_reason = "gles_owner_thread_start_failed_unknown";
                startup_complete_ = true;
            }
        }
        startup_cv_.wait(lock, [this] { return startup_complete_; });
        return capabilities_;
    }

    bool run_task(const std::function<void()>& task, std::string& error) {
        error.clear();
        if (!task) { error = "gles_task_callback_is_empty"; return false; }
        const UniversalGpuCapabilities caps = initialize();
        if (!caps.egl_context_created || !caps.gles_compute_supported ||
            !caps.gles_compute_self_test_passed) {
            error = caps.failure_reason.empty() ? "gles_context_not_compute_ready" : caps.failure_reason;
            return false;
        }
        auto completion = std::make_shared<std::promise<void>>();
        std::future<void> finished = completion->get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!worker_.joinable() || stopping_) {
                error = "gles_owner_thread_is_stopping";
                return false;
            }
            tasks_.emplace_back([task, completion]() {
                try { task(); completion->set_value(); }
                catch (...) { completion->set_exception(std::current_exception()); }
            });
        }
        task_cv_.notify_one();
        try { finished.get(); return true; }
        catch (const std::exception& exception) {
            error = std::string("gles_task_exception_") + exception.what();
        } catch (...) { error = "gles_task_exception_unknown"; }
        return false;
    }

    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!worker_.joinable()) return;
            stopping_ = true;
        }
        task_cv_.notify_one();
        worker_.join();
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = false;
        startup_complete_ = false;
        capabilities_ = {};
        tasks_.clear();
    }

private:
    void publish_startup(const UniversalGpuCapabilities &caps) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            capabilities_ = caps;
            startup_complete_ = true;
        }
        startup_cv_.notify_all();
    }

    void process_tasks() {
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                task_cv_.wait(lock, [this] { return stopping_ || !tasks_.empty(); });
                if (stopping_ && tasks_.empty()) return;
                task = std::move(tasks_.front());
                tasks_.pop_front();
            }
            task();
        }
    }

    void run() {
        UniversalGpuCapabilities caps;
        EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        EGLSurface surface = EGL_NO_SURFACE;
        EGLContext context = EGL_NO_CONTEXT;
        bool context_current = false;

        if (display == EGL_NO_DISPLAY) {
            caps.failure_reason = "egl_get_display_failed_" + egl_error_name(eglGetError());
        } else {
            EGLint egl_major = 0;
            EGLint egl_minor = 0;
            if (eglInitialize(display, &egl_major, &egl_minor) != EGL_TRUE) {
                caps.failure_reason = "egl_initialize_failed_" + egl_error_name(eglGetError());
            } else {
                caps.egl_display_available = true;
                caps.egl_major = egl_major;
                caps.egl_minor = egl_minor;
                caps.egl_version = egl_string(display, EGL_VERSION);
                caps.egl_vendor = egl_string(display, EGL_VENDOR);

                if (eglBindAPI(EGL_OPENGL_ES_API) != EGL_TRUE) {
                    caps.failure_reason = "egl_bind_opengles_api_failed_" + egl_error_name(eglGetError());
                } else {
                    EGLint config_attributes[] = {
                        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR,
                        EGL_RED_SIZE, 8,
                        EGL_GREEN_SIZE, 8,
                        EGL_BLUE_SIZE, 8,
                        EGL_NONE
                    };
                    EGLConfig config = nullptr;
                    EGLint config_count = 0;
                    if (eglChooseConfig(display, config_attributes, &config, 1, &config_count) != EGL_TRUE ||
                        config_count < 1) {
                        caps.failure_reason = "egl_choose_es3_pbuffer_config_failed_" + egl_error_name(eglGetError());
                    } else {
                        EGLint pbuffer_attributes[] = { EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE };
                        surface = eglCreatePbufferSurface(display, config, pbuffer_attributes);
                        if (surface == EGL_NO_SURFACE) {
                            caps.failure_reason = "egl_create_pbuffer_failed_" + egl_error_name(eglGetError());
                        } else {
                            EGLint context_attributes[] = { EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE };
                            context = eglCreateContext(display, config, EGL_NO_CONTEXT, context_attributes);
                            if (context == EGL_NO_CONTEXT) {
                                caps.failure_reason = "egl_create_gles3_context_failed_" + egl_error_name(eglGetError());
                            } else if (eglMakeCurrent(display, surface, surface, context) != EGL_TRUE) {
                                caps.failure_reason = "egl_make_current_failed_" + egl_error_name(eglGetError());
                            } else {
                                context_current = true;
                                caps.egl_context_created = true;
                            }
                        }
                    }
                }
            }
        }

        if (context_current) {
            caps.vendor = gl_string(GL_VENDOR);
            caps.renderer = gl_string(GL_RENDERER);
            caps.version = gl_string(GL_VERSION);
            caps.shading_language_version = gl_string(GL_SHADING_LANGUAGE_VERSION);

            GLint major = 0;
            GLint minor = 0;
            glGetIntegerv(GL_MAJOR_VERSION, &major);
            glGetIntegerv(GL_MINOR_VERSION, &minor);
            caps.gles_major = major;
            caps.gles_minor = minor;

            const std::string extensions = gl_extensions(caps.extension_count);
            caps.gles_compute_supported = major > 3 || (major == 3 && minor >= 1);
            caps.gles_fp16_shader_hint =
                has_extension(extensions, "GL_EXT_shader_explicit_arithmetic_types_float16") ||
                has_extension(extensions, "GL_AMD_gpu_shader_half_float") ||
                has_extension(extensions, "GL_NV_gpu_shader5");

            if (caps.gles_compute_supported) {
                read_compute_limits(caps);
                const GLenum error = glGetError();
                if (error != GL_NO_ERROR) {
                    caps.failure_reason = "gles_compute_limit_query_failed";
                    caps.gles_compute_supported = false;
                } else {
                    std::string self_test_failure;
                    caps.gles_compute_self_test_passed = run_gles_compute_self_test(self_test_failure);
                    if (!caps.gles_compute_self_test_passed && caps.failure_reason.empty()) {
                        caps.failure_reason = self_test_failure.empty()
                            ? "gles_compute_self_test_failed" : self_test_failure;
                    }
                }
            } else if (caps.failure_reason.empty()) {
                caps.failure_reason = "opengles_3_1_compute_not_supported";
            }
        }

        // Publish even on failure so the caller never waits indefinitely.
        publish_startup(caps);
        process_tasks();

        // All EGL/GLES teardown occurs on the same thread that owns the context.
        if (context_current) {
            eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        }
        if (context != EGL_NO_CONTEXT) eglDestroyContext(display, context);
        if (surface != EGL_NO_SURFACE) eglDestroySurface(display, surface);
        // Do not eglTerminate the process-wide display: Android UI and other
        // native libraries may share it independently.
    }

    std::mutex mutex_;
    std::condition_variable startup_cv_;
    std::condition_variable task_cv_;
    std::thread worker_;
    std::deque<std::function<void()>> tasks_;
    UniversalGpuCapabilities capabilities_;
    bool startup_complete_ = false;
    bool stopping_ = false;
};

PersistentGlesProbe & persistent_gles_probe() {
    static PersistentGlesProbe instance;
    return instance;
}

} // namespace

std::string UniversalGpuCapabilities::diagnostic_line() const {
    std::ostringstream out;
    out << "UNIVERSAL_GPU_ADAPTER"
        << " eglDisplay=" << (egl_display_available ? 1 : 0)
        << " eglContext=" << (egl_context_created ? 1 : 0)
        << " eglVersion=" << (egl_version.empty() ? "unknown" : safe_token(egl_version))
        << " eglVendor=" << safe_token(egl_vendor)
        << " glesVersion=" << gles_major << "." << gles_minor
        << " glesCompute=" << (gles_compute_supported ? 1 : 0)
        << " glesComputeSelfTest=" << (gles_compute_self_test_passed ? 1 : 0)
        << " glesFp16ShaderHint=" << (gles_fp16_shader_hint ? 1 : 0)
        << " vendor=" << safe_token(vendor)
        << " renderer=" << safe_token(renderer)
        << " glVersion=" << safe_token(version)
        << " glslVersion=" << safe_token(shading_language_version)
        << " extensionCount=" << extension_count
        << " maxComputeInvocations=" << max_compute_invocations
        << " maxComputeSharedBytes=" << max_compute_shared_memory_bytes
        << " maxSsboBytes=" << max_shader_storage_block_bytes
        << " maxWorkgroupCount=" << max_compute_workgroup_count_x << "x"
        << max_compute_workgroup_count_y << "x" << max_compute_workgroup_count_z
        << " maxWorkgroupSize=" << max_compute_workgroup_size_x << "x"
        << max_compute_workgroup_size_y << "x" << max_compute_workgroup_size_z
        << " failureReason=" << (failure_reason.empty() ? "none" : safe_token(failure_reason));
    return out.str();
}

UniversalGpuCapabilities probe_universal_gpu_adapter() {
    return persistent_gles_probe().initialize();
}

bool run_universal_gpu_task(const std::function<void()>& task, std::string& error) {
    return persistent_gles_probe().run_task(task, error);
}

void shutdown_universal_gpu_adapter() {
    persistent_gles_probe().shutdown();
}

} // namespace ai_chat::gpu