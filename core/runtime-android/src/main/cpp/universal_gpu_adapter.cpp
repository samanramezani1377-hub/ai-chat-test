#include "universal_gpu_adapter.h"

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <GLES2/gl2ext.h>

#include <algorithm>
#include <cstdio>
#include <sstream>
#include <string>

namespace ai_chat::gpu {
namespace {

class EglCurrentContext {
public:
    EglCurrentContext()
        : previous_api_(eglQueryAPI()),
          previous_display_(eglGetCurrentDisplay()),
          previous_draw_(eglGetCurrentSurface(EGL_DRAW)),
          previous_read_(eglGetCurrentSurface(EGL_READ)),
          previous_context_(eglGetCurrentContext()) {}
    EglCurrentContext(const EglCurrentContext &) = delete;
    EglCurrentContext & operator=(const EglCurrentContext &) = delete;

    ~EglCurrentContext() { restore(); }

    bool bind(EGLDisplay display, EGLSurface surface, EGLContext context) {
        if (eglMakeCurrent(display, surface, surface, context) != EGL_TRUE) return false;
        active_display_ = display;
        changed_ = true;
        return true;
    }

    bool restore() {
        if (!changed_) return true;
        EGLBoolean result = EGL_FALSE;
        if (previous_context_ != EGL_NO_CONTEXT && previous_display_ != EGL_NO_DISPLAY) {
            // eglMakeCurrent validates the context against the thread's bound
            // client API. Restore that API before rebinding an application
            // context that may have been created for desktop OpenGL.
            if (eglBindAPI(previous_api_) != EGL_TRUE) {
                last_restore_error_ = eglGetError();
                eglMakeCurrent(active_display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
                changed_ = false;
                return false;
            }
            result = eglMakeCurrent(previous_display_, previous_draw_, previous_read_, previous_context_);
        } else {
            // Unbind the temporary GLES context while GLES is still the bound API.
            result = eglMakeCurrent(active_display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
            if (result == EGL_TRUE && eglBindAPI(previous_api_) != EGL_TRUE) {
                last_restore_error_ = eglGetError();
                changed_ = false;
                return false;
            }
        }
        const EGLint restore_error = result == EGL_TRUE ? EGL_SUCCESS : eglGetError();
        if (result != EGL_TRUE) {
            // Best effort: do not leave the temporary probe context current if
            // restoring an application-owned context failed.
            eglMakeCurrent(active_display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        }
        changed_ = false;
        last_restore_error_ = restore_error;
        return result == EGL_TRUE;
    }

    EGLint last_restore_error() const { return last_restore_error_; }

private:
    EGLenum previous_api_ = EGL_OPENGL_ES_API;
    EGLDisplay previous_display_ = EGL_NO_DISPLAY;
    EGLSurface previous_draw_ = EGL_NO_SURFACE;
    EGLSurface previous_read_ = EGL_NO_SURFACE;
    EGLContext previous_context_ = EGL_NO_CONTEXT;
    EGLDisplay active_display_ = EGL_NO_DISPLAY;
    EGLint last_restore_error_ = EGL_SUCCESS;
    bool changed_ = false;
};

class EglApiBinding {
public:
    EglApiBinding() : previous_api_(eglQueryAPI()) {}
    EglApiBinding(const EglApiBinding &) = delete;
    EglApiBinding & operator=(const EglApiBinding &) = delete;
    ~EglApiBinding() {
        if (bound_) eglBindAPI(previous_api_);
    }
    bool bind_opengles() {
        if (eglBindAPI(EGL_OPENGL_ES_API) != EGL_TRUE) return false;
        bound_ = true;
        return true;
    }

private:
    EGLenum previous_api_;
    bool bound_ = false;
};

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
    UniversalGpuCapabilities caps;
    EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (display == EGL_NO_DISPLAY) {
        caps.failure_reason = "egl_get_display_failed_" + egl_error_name(eglGetError());
        return caps;
    }

    EGLint egl_major = 0;
    EGLint egl_minor = 0;
    if (eglInitialize(display, &egl_major, &egl_minor) != EGL_TRUE) {
        caps.failure_reason = "egl_initialize_failed_" + egl_error_name(eglGetError());
        return caps;
    }
    caps.egl_display_available = true;
    caps.egl_major = egl_major;
    caps.egl_minor = egl_minor;
    caps.egl_version = egl_string(display, EGL_VERSION);
    caps.egl_vendor = egl_string(display, EGL_VENDOR);

    EglApiBinding api_binding;
    if (!api_binding.bind_opengles()) {
        caps.failure_reason = "egl_bind_opengles_api_failed_" + egl_error_name(eglGetError());
        return caps;
    }

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
        return caps;
    }

    EGLint pbuffer_attributes[] = { EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE };
    EGLSurface surface = eglCreatePbufferSurface(display, config, pbuffer_attributes);
    if (surface == EGL_NO_SURFACE) {
        caps.failure_reason = "egl_create_pbuffer_failed_" + egl_error_name(eglGetError());
        return caps;
    }

    EGLint context_attributes[] = { EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE };
    EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT, context_attributes);
    if (context == EGL_NO_CONTEXT) {
        caps.failure_reason = "egl_create_gles3_context_failed_" + egl_error_name(eglGetError());
        eglDestroySurface(display, surface);
        return caps;
    }

    EglCurrentContext current_context;
    if (!current_context.bind(display, surface, context)) {
        caps.failure_reason = "egl_make_current_failed_" + egl_error_name(eglGetError());
        eglDestroyContext(display, context);
        eglDestroySurface(display, surface);
        return caps;
    }
    caps.egl_context_created = true;

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

    const std::string extensions = gl_string(GL_EXTENSIONS);
    size_t extension_pos = 0;
    while ((extension_pos = extensions.find(' ', extension_pos)) != std::string::npos) {
        ++caps.extension_count;
        ++extension_pos;
    }
    if (!extensions.empty()) ++caps.extension_count;

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
        }
    } else if (caps.failure_reason.empty()) {
        caps.failure_reason = "opengles_3_1_compute_not_supported";
    }

    // Restore the thread's previous EGL context/surfaces before destroying
    // the temporary probe context. The probe may run on a thread that already
    // owns an application graphics context; unbinding it would break rendering.
    if (!current_context.restore() && caps.failure_reason.empty()) {
        caps.failure_reason = "egl_restore_previous_context_failed_" +
            egl_error_name(current_context.last_restore_error());
    }
    eglDestroyContext(display, context);
    eglDestroySurface(display, surface);
    return caps;
}

} // namespace ai_chat::gpu
