#include "opengles_runtime.h"

#include <EGL/egl.h>
#include <GLES3/gl31.h>

#include <mutex>
#include <sstream>
#include <cstdio>

namespace {
EGLDisplay g_display = EGL_NO_DISPLAY;
EGLContext g_context = EGL_NO_CONTEXT;
EGLSurface g_surface = EGL_NO_SURFACE;
OpenGLESRuntimeInfo g_info;
std::mutex g_mutex;

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
    if (g_info.available) return true;
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

    EGLConfig config = nullptr;
    EGLint config_count = 0;
    if (!eglChooseConfig(g_display, config_attribs, &config, 1, &config_count) ||
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
    g_context = eglCreateContext(g_display, config, EGL_NO_CONTEXT, context_attribs);
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
    g_surface = eglCreatePbufferSurface(g_display, config, surface_attribs);
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
        opengles_runtime_shutdown();
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

    g_info.available = true;
    g_info.failure.clear();
    return true;
}

void opengles_runtime_shutdown() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_display == EGL_NO_DISPLAY) return;

    eglMakeCurrent(g_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    if (g_surface != EGL_NO_SURFACE) eglDestroySurface(g_display, g_surface);
    if (g_context != EGL_NO_CONTEXT) eglDestroyContext(g_display, g_context);
    eglTerminate(g_display);

    g_surface = EGL_NO_SURFACE;
    g_context = EGL_NO_CONTEXT;
    g_display = EGL_NO_DISPLAY;
}

bool opengles_runtime_available() {
    return opengles_runtime_init();
}

const OpenGLESRuntimeInfo & opengles_runtime_info() {
    opengles_runtime_init();
    return g_info;
}
