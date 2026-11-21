#include "android_backends.h"

namespace ai_chat::gpu_engine {

Status register_android_gpu_backends(GpuEngine& engine) {
    Status status = engine.register_backend(create_dawn_backend());
    if (!status.ok()) return status;
    return engine.register_backend(create_universal_gles_backend());
}

Status initialize_android_gpu_engine(GpuEngine& engine) {
    Status status = register_android_gpu_backends(engine);
    if (!status.ok()) return status;
    return engine.initialize_with_fallback({BackendKind::DawnWebGpu, BackendKind::OpenGles});
}

Status initialize_android_gpu_engine(GpuEngine& engine) {
    Status status = register_android_gpu_backends(engine);
    if (!status.ok()) return status;
    return engine.initialize_with_fallback({BackendKind::DawnWebGpu, BackendKind::OpenGles});
}
} // namespace ai_chat::gpu_engine
