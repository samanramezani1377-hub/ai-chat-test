#include "android_backends.h"

namespace ai_chat::gpu_engine {

Status register_android_gpu_backends(GpuEngine& engine) {
    Status status = engine.register_backend(create_dawn_backend());
    if (!status.ok()) return status;
    return engine.register_backend(create_universal_gles_backend());
}

} // namespace ai_chat::gpu_engine
