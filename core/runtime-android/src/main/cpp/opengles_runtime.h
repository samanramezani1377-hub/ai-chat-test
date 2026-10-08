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
    std::string failure;
};

bool opengles_runtime_init();
void opengles_runtime_shutdown();
bool opengles_runtime_available();
const OpenGLESRuntimeInfo & opengles_runtime_info();
