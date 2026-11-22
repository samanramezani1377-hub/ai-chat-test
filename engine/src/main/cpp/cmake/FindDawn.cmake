# Build the native Dawn implementation from the exact source revision
# recorded by the AndroidX WebGPU AAR. The AAR's bundled JNI library does not
# export the WebGPU C ABI consumed by ggml-webgpu.
if (NOT DEFINED AI_CHAT_DAWN_METADATA_FILE OR NOT EXISTS "${AI_CHAT_DAWN_METADATA_FILE}")
    message(FATAL_ERROR
        "AI_CHAT_DAWN_METADATA_FILE is missing. Extract assets/dawn_build_metadata.json "
        "from the pinned AndroidX WebGPU AAR before CMake configuration.")
endif()

file(READ "${AI_CHAT_DAWN_METADATA_FILE}" _ai_chat_dawn_metadata)
string(JSON AI_CHAT_DAWN_SOURCE_COMMIT
    ERROR_VARIABLE _ai_chat_dawn_metadata_error
    GET "${_ai_chat_dawn_metadata}" dawn_source_commit_sha)
if (NOT _ai_chat_dawn_metadata_error STREQUAL "NOTFOUND")
    message(FATAL_ERROR
        "Could not read dawn_source_commit_sha from AndroidX metadata: ${_ai_chat_dawn_metadata_error}")
endif()
string(LENGTH "${AI_CHAT_DAWN_SOURCE_COMMIT}" _ai_chat_dawn_sha_length)
if (NOT _ai_chat_dawn_sha_length EQUAL 40 OR
    NOT AI_CHAT_DAWN_SOURCE_COMMIT MATCHES "^[0-9a-fA-F]+$")
    message(FATAL_ERROR "AndroidX AAR contains an invalid Dawn source SHA: ${AI_CHAT_DAWN_SOURCE_COMMIT}")
endif()

include(FetchContent)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Dawn's CMake build requires C++20. Restore the parent project's language
# standard immediately after creating Dawn targets so GGML keeps its own policy.
set(_ai_chat_saved_cxx_standard "${CMAKE_CXX_STANDARD}")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(DAWN_FETCH_DEPENDENCIES ON CACHE BOOL "Fetch Dawn's pinned third-party dependencies" FORCE)
set(DAWN_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(DAWN_ENABLE_OPENGLES ON CACHE BOOL "Build Dawn's OpenGL ES backend" FORCE)
set(DAWN_ENABLE_VULKAN OFF CACHE BOOL "Vulkan is intentionally disabled" FORCE)
set(DAWN_ENABLE_DESKTOP_GL OFF CACHE BOOL "" FORCE)
set(DAWN_ENABLE_D3D11 OFF CACHE BOOL "" FORCE)
set(DAWN_ENABLE_D3D12 OFF CACHE BOOL "" FORCE)
set(DAWN_ENABLE_METAL OFF CACHE BOOL "" FORCE)
set(DAWN_ENABLE_NULL OFF CACHE BOOL "" FORCE)
set(DAWN_BUILD_MONOLITHIC_LIBRARY STATIC CACHE STRING "Build Dawn as one static native library" FORCE)
set(DAWN_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
set(DAWN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(DAWN_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(DAWN_BUILD_FUZZERS OFF CACHE BOOL "" FORCE)
set(DAWN_BUILD_NODE_BINDINGS OFF CACHE BOOL "" FORCE)
set(DAWN_BUILD_PROTOBUF OFF CACHE BOOL "" FORCE)
set(DAWN_ENABLE_SPIRV_VALIDATION OFF CACHE BOOL "" FORCE)
set(TINT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(TINT_BUILD_CMD_TOOLS OFF CACHE BOOL "" FORCE)
set(TINT_BUILD_SPV_READER OFF CACHE BOOL "" FORCE)
set(TINT_BUILD_SPV_WRITER OFF CACHE BOOL "" FORCE)
set(TINT_BUILD_GLSL_WRITER ON CACHE BOOL "" FORCE)
set(TINT_BUILD_WGSL_READER ON CACHE BOOL "" FORCE)
# Dawn OpenGL ES compute-pipeline validation uses Tint's null writer to
# reflect workgroup metadata. This is a Tint compiler component, not Dawn's
# optional Null GPU backend; keep the backend itself disabled.
set(TINT_BUILD_NULL_WRITER ON CACHE BOOL "Build Tint null writer required by Dawn OpenGL ES" FORCE)

FetchContent_Declare(ai_chat_dawn
    GIT_REPOSITORY https://dawn.googlesource.com/dawn.git
    GIT_TAG "${AI_CHAT_DAWN_SOURCE_COMMIT}"
    GIT_SHALLOW FALSE
    # Dawn's repository contains optional Git submodules for test suites,
    # alternate backends and tooling. They are not part of this Android
    # OpenGL ES build; recursively cloning them pulls inaccessible/private
    # Chromium test repositories and wastes most of the CI job before compile.
    # DAWN_FETCH_DEPENDENCIES below still fetches Dawn's declared build deps.
    GIT_SUBMODULES ""
    GIT_SUBMODULES_RECURSE FALSE
)
FetchContent_MakeAvailable(ai_chat_dawn)

set(CMAKE_CXX_STANDARD "${_ai_chat_saved_cxx_standard}")
unset(_ai_chat_saved_cxx_standard)

if (NOT TARGET dawn::webgpu_dawn)
    message(FATAL_ERROR
        "The pinned Dawn source did not create dawn::webgpu_dawn; "
        "the native WebGPU C ABI cannot be linked.")
endif()

set(Dawn_FOUND TRUE)
set(Dawn_VERSION "Dawn-${AI_CHAT_DAWN_SOURCE_COMMIT}")
