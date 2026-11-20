# AndroidX Jetpack WebGPU publishes Dawn as a Prefab package.
# GGML's upstream WebGPU CMake expects the canonical dawn::webgpu_dawn target.
find_package(webgpu_c_bundled CONFIG REQUIRED)

if(TARGET webgpu_c_bundled::webgpu_c_bundled)
    set(_ai_chat_dawn_target webgpu_c_bundled::webgpu_c_bundled)
elseif(TARGET webgpu_c_bundled)
    set(_ai_chat_dawn_target webgpu_c_bundled)
else()
    message(FATAL_ERROR "AndroidX WebGPU Prefab package did not export webgpu_c_bundled target")
endif()

if(NOT TARGET dawn::webgpu_dawn)
    add_library(dawn::webgpu_dawn INTERFACE IMPORTED)
    set_target_properties(dawn::webgpu_dawn PROPERTIES
        INTERFACE_LINK_LIBRARIES "${_ai_chat_dawn_target}"
    )
endif()

set(Dawn_FOUND TRUE)
set(Dawn_VERSION "AndroidX-Jetpack-WebGPU-1.0.0-alpha06")
