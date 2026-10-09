#include "ggml-opengles.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

namespace {
struct OpenGLESBackendRegistrar {
    OpenGLESBackendRegistrar() {
        // Registration is intentionally static: Android disables GGML dynamic backend
        // loading, so the OpenGL ES backend must be present in the global registry.
        ggml_backend_register(ggml_backend_opengles_reg());
    }
};
static OpenGLESBackendRegistrar g_registrar;
}
