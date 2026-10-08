#pragma once

#include <string>

struct OpenGLESRuntimeInfo {
    bool available = false;
    int major = 0;
    int minor = 0;
    std::string vendor;
    std::string renderer;
    std::string version;
    std::string glsl_version;
    std::string extensions;
    int max_compute_ssbo_blocks = 0;
    int max_workgroup_size_x = 0;
    int64_t max_ssbo_block_size = 0;
    std::string failure;
};

bool opengles_runtime_init();
void opengles_runtime_shutdown();
bool opengles_runtime_available();
const OpenGLESRuntimeInfo & opengles_runtime_info();
