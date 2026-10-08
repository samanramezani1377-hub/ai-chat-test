#include "opengles_runtime.h"

#include <EGL/egl.h>
#include <EGL/eglext.h>
#ifndef EGL_OPENGL_ES3_BIT_KHR
#define EGL_OPENGL_ES3_BIT_KHR 0x0040
#endif
#include <GLES3/gl31.h>

#include <mutex>
#include <sstream>
#include <cstdio>
#include <thread>
#include <unordered_map>

namespace {
EGLDisplay g_display = EGL_NO_DISPLAY;
EGLContext g_context = EGL_NO_CONTEXT;
EGLSurface g_surface = EGL_NO_SURFACE;
EGLConfig g_config = nullptr;
std::thread::id g_root_thread;
struct ThreadContext {
    EGLContext context = EGL_NO_CONTEXT;
    EGLSurface surface = EGL_NO_SURFACE;
};
std::unordered_map<std::thread::id, ThreadContext> g_thread_contexts;
OpenGLESRuntimeInfo g_info;
std::mutex g_mutex;

static void shutdown_locked() {
    if (g_display == EGL_NO_DISPLAY) return;
    eglMakeCurrent(g_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    for (auto & entry : g_thread_contexts) {
        if (entry.second.surface != EGL_NO_SURFACE) eglDestroySurface(g_display, entry.second.surface);
        if (entry.second.context != EGL_NO_CONTEXT) eglDestroyContext(g_display, entry.second.context);
    }
    g_thread_contexts.clear();
    if (g_surface != EGL_NO_SURFACE) eglDestroySurface(g_display, g_surface);
    if (g_context != EGL_NO_CONTEXT) eglDestroyContext(g_display, g_context);
    eglTerminate(g_display);
    g_surface = EGL_NO_SURFACE;
    g_context = EGL_NO_CONTEXT;
    g_config = nullptr;
    g_display = EGL_NO_DISPLAY;
}

static void fail(const char * reason) {
    g_info.available = false;
    g_info.failure = reason ? reason : "unknown";
}

static bool parse_version(const char * value) {
    if (!value) return false;
    int major = 0;
    int minor = 0;
    if (sscanf(value, "OpenGL ES %d.%d", &major, &minor) != 2) {
        return false;
    }
    g_info.major = major;
    g_info.minor = minor;
    return major > 3 || (major == 3 && minor >= 1);
}

static std::string safe_string(GLenum name) {
    const GLubyte * value = glGetString(name);
    return value ? reinterpret_cast<const char *>(value) : std::string();
}
}

bool opengles_runtime_init() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_info.available) {
        const std::thread::id tid = std::this_thread::get_id();
        const EGLContext current = eglGetCurrentContext();
        if (current == g_context) return true;
        auto found = g_thread_contexts.find(tid);
        if (found != g_thread_contexts.end()) {
            if (current == found->second.context) return true;
            if (eglMakeCurrent(g_display, found->second.surface, found->second.surface,
                               found->second.context)) return true;
            g_info.failure = "EGL_SHARED_THREAD_CONTEXT_MAKE_CURRENT_FAILED";
            return false;
        }
        if (tid == g_root_thread) {
            if (eglMakeCurrent(g_display, g_surface, g_surface, g_context)) return true;
            g_info.failure = "EGL_ROOT_CONTEXT_MAKE_CURRENT_FAILED";
            return false;
        }

        // EGL contexts are thread-current, not process-current. llama.cpp may
        // load model weights and initialize the inference context on different
        // executor threads. Give each thread a context in the same share group
        // so shared programs/buffers remain visible without moving a live
        // context between threads.
        if (!eglBindAPI(EGL_OPENGL_ES_API)) {
            g_info.failure = "EGL_BIND_SHARED_CONTEXT_API_FAILED";
            return false;
        }
        const EGLint attrs[] = { EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE };
        ThreadContext shared;
        shared.context = eglCreateContext(g_display, g_config, g_context, attrs);
        if (shared.context == EGL_NO_CONTEXT) {
            g_info.failure = "EGL_CREATE_SHARED_THREAD_CONTEXT_FAILED";
            return false;
        }
        const EGLint surface_attrs[] = { EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE };
        shared.surface = eglCreatePbufferSurface(g_display, g_config, surface_attrs);
        if (shared.surface == EGL_NO_SURFACE ||
            !eglMakeCurrent(g_display, shared.surface, shared.surface, shared.context)) {
            if (shared.surface != EGL_NO_SURFACE) eglDestroySurface(g_display, shared.surface);
            eglDestroyContext(g_display, shared.context);
            g_info.failure = "EGL_SHARED_THREAD_CONTEXT_MAKE_CURRENT_FAILED";
            return false;
        }
        g_thread_contexts.emplace(tid, shared);
        g_info.failure.clear();
        return true;
    }
    if (g_display != EGL_NO_DISPLAY) return false;

    g_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (g_display == EGL_NO_DISPLAY) {
        fail("EGL_DEFAULT_DISPLAY_UNAVAILABLE");
        return false;
    }

    EGLint major = 0;
    EGLint minor = 0;
    if (!eglInitialize(g_display, &major, &minor)) {
        fail("EGL_INITIALIZE_FAILED");
        g_display = EGL_NO_DISPLAY;
        return false;
    }

    const EGLint config_attribs[] = {
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR,
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_NONE
    };

    EGLint config_count = 0;
    if (!eglChooseConfig(g_display, config_attribs, &g_config, 1, &config_count) ||
        config_count != 1) {
        fail("EGL_ES3_CONFIG_UNAVAILABLE");
        eglTerminate(g_display);
        g_display = EGL_NO_DISPLAY;
        return false;
    }

    if (!eglBindAPI(EGL_OPENGL_ES_API)) {
        fail("EGL_BIND_OPENGL_ES_FAILED");
        eglTerminate(g_display);
        g_display = EGL_NO_DISPLAY;
        return false;
    }

    const EGLint context_attribs[] = {
        EGL_CONTEXT_CLIENT_VERSION, 3,
        EGL_NONE
    };
    g_context = eglCreateContext(g_display, g_config, EGL_NO_CONTEXT, context_attribs);
    if (g_context == EGL_NO_CONTEXT) {
        fail("EGL_CREATE_CONTEXT_FAILED");
        eglTerminate(g_display);
        g_display = EGL_NO_DISPLAY;
        return false;
    }

    const EGLint surface_attribs[] = {
        EGL_WIDTH, 1,
        EGL_HEIGHT, 1,
        EGL_NONE
    };
    g_surface = eglCreatePbufferSurface(g_display, g_config, surface_attribs);
    if (g_surface == EGL_NO_SURFACE ||
        !eglMakeCurrent(g_display, g_surface, g_surface, g_context)) {
        fail("EGL_MAKE_CURRENT_FAILED");
        eglMakeCurrent(g_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (g_surface != EGL_NO_SURFACE) eglDestroySurface(g_display, g_surface);
        if (g_context != EGL_NO_CONTEXT) eglDestroyContext(g_display, g_context);
        eglTerminate(g_display);
        g_surface = EGL_NO_SURFACE;
        g_context = EGL_NO_CONTEXT;
        g_display = EGL_NO_DISPLAY;
        return false;
    }

    g_info.vendor = safe_string(GL_VENDOR);
    g_info.renderer = safe_string(GL_RENDERER);
    g_info.version = safe_string(GL_VERSION);
    g_info.glsl_version = safe_string(GL_SHADING_LANGUAGE_VERSION);
    g_info.extensions = safe_string(GL_EXTENSIONS);

    GLint ssbo_blocks = 0;
    GLint workgroup_x = 0;
    GLint64 ssbo_block_size = 0;
    glGetIntegerv(GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS, &ssbo_blocks);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &workgroup_x);
    glGetInteger64v(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &ssbo_block_size);
    if (glGetError() != GL_NO_ERROR || ssbo_blocks <= 0 || workgroup_x <= 0 || ssbo_block_size <= 0) {
        fail("OPENGL_ES_COMPUTE_LIMIT_QUERY_FAILED");
        shutdown_locked();
        return false;
    }
    g_info.max_compute_ssbo_blocks = ssbo_blocks;
    g_info.max_workgroup_size_x = workgroup_x;
    g_info.max_ssbo_block_size = ssbo_block_size;

    if (!parse_version(g_info.version.c_str())) {
        fail("OPENGL_ES_31_COMPUTE_UNAVAILABLE");
        opengles_runtime_shutdown();
        return false;
    }

    // A successful ES 3.1 context guarantees the core compute-shader API.
    // Probe one minimal shader object so broken/partial driver stacks are rejected
    // before the inference backend is selected.
    const GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    const GLenum error = glGetError();
    if (!shader || error != GL_NO_ERROR) {
        fail("OPENGL_ES_COMPUTE_SHADER_UNAVAILABLE");
        opengles_runtime_shutdown();
        return false;
    }
    glDeleteShader(shader);
    if (glGetError() != GL_NO_ERROR) {
        fail("OPENGL_ES_COMPUTE_SHADER_DELETE_FAILED");
        opengles_runtime_shutdown();
        return false;
    }

    g_root_thread = std::this_thread::get_id();
    g_info.available = true;
    g_info.failure.clear();
    return true;
}

void opengles_runtime_shutdown() {
    std::lock_guard<std::mutex> lock(g_mutex);
    shutdown_locked();
    g_info.available = false;
    g_info.failure.clear();
}

bool opengles_runtime_available() {
    return opengles_runtime_init();
}

const OpenGLESRuntimeInfo & opengles_runtime_info() {
    opengles_runtime_init();
    return g_info;
}
