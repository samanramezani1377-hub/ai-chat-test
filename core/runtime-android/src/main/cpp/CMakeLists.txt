cmake_minimum_required(VERSION 3.22)
project(ai_chat_runtime LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
set(LLAMA_CPP_TAG "9871df5911a03a813518bcd623ee2eba91bf32aa" CACHE STRING "Pinned llama.cpp revision for the Android OpenGL ES runtime")
set(FETCHCONTENT_QUIET OFF)
set(FETCHCONTENT_BASE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../.cxx/fetch-content" CACHE PATH "Stable FetchContent base directory")
set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL "Do not update pinned FetchContent dependencies")

FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG ${LLAMA_CPP_TAG}
    GIT_SHALLOW TRUE
)
# OpenGL ES is provided by Android's system EGL/GLES implementation.
# No vendor-specific GPU SDK is bundled in the APK.
set(GGML_BACKEND_DL OFF CACHE BOOL "Disable dynamic ggml backend loading on Android" FORCE)
set(GGML_BACKEND_DIR "" CACHE STRING "No dynamic backend directory" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_UI OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(llama_cpp)

# Android GPU-only contract: keep llama.cpp's CPU backend available as an internal
# helper for host-side tensor utilities, but exclude it from the graph scheduler.
# This prevents unsupported GGML nodes from silently being reassigned to CPU.
set(_llama_context_cpp "${llama_cpp_SOURCE_DIR}/src/llama-context.cpp")
file(READ "${_llama_context_cpp}" _llama_context_source)
if(NOT _llama_context_source MATCHES "AI_CHAT_GPU_ONLY_SCHEDULER_PATCH")
    string(REPLACE
        "            auto backend_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));

            if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU && !model.devices.empty()) {"
        "            auto backend_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));

            // AI_CHAT_GPU_ONLY_SCHEDULER_PATCH: CPU remains initialized as a host helper,
            // but it is never inserted into the graph scheduler backend list.
            if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
                continue;
            }

            if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU && !model.devices.empty()) {"
        _llama_context_source)
    string(REPLACE
        "    ggml_opt_free(opt_ctx);
}"
        "    ggml_opt_free(opt_ctx);
    if (backend_cpu) {
        ggml_backend_free(backend_cpu);
        backend_cpu = nullptr;
    }
}"
        _llama_context_source)
    file(WRITE "${_llama_context_cpp}" "${_llama_context_source}")
endif()

# Fail configuration if the pinned llama.cpp source could not be patched. In GPU-only
# mode a silent miss here would re-enable the CPU scheduler and violate the runtime contract.
file(READ "${_llama_context_cpp}" _llama_context_verify)
if(NOT _llama_context_verify MATCHES "AI_CHAT_GPU_ONLY_SCHEDULER_PATCH")
    message(FATAL_ERROR "AI_CHAT_GPU_ONLY_SCHEDULER_PATCH was not applied to pinned llama-context.cpp; refusing to build a CPU-fallback runtime")
endif()

add_library(ai_chat_runtime SHARED native_runtime_android_safe.cpp opengles_runtime.cpp ggml-opengles/ggml-opengles.cpp ggml-opengles/ggml-opengles-registration.cpp)

# native_runtime_android_safe.cpp intentionally includes llama.cpp/ggml internal
# headers for strict backend and tensor-residency validation. Keep both
# internal source directories on the include path; the public ggml include
# directory alone is insufficient for ggml-backend-impl.h on this pinned revision.
target_include_directories(ai_chat_runtime PRIVATE
    ${llama_cpp_SOURCE_DIR}/include
    ${llama_cpp_SOURCE_DIR}/src
    ${llama_cpp_SOURCE_DIR}/ggml/include
    ${llama_cpp_SOURCE_DIR}/ggml/src
    ${CMAKE_CURRENT_LIST_DIR}/ggml-opengles
)

# The runtime is a single arm64 production native library. Strip unused sections
# and symbols at link time to reduce the installed footprint without changing inference.
target_compile_options(ai_chat_runtime PRIVATE
    -Os
    -ffunction-sections
    -fdata-sections
)
target_link_options(ai_chat_runtime PRIVATE
    -Wl,--gc-sections
    -Wl,--strip-all
)

target_link_libraries(ai_chat_runtime PRIVATE llama llama-common android log EGL GLESv3)

target_compile_definitions(ai_chat_runtime PRIVATE
    AI_CHAT_LLAMA_CPP_SHA="${LLAMA_CPP_TAG}"
    AI_CHAT_GPU_BACKEND_OPENGL=1
    AI_CHAT_REQUIRE_OPENGL_GPU=1
)

if (ANDROID)
    set_target_properties(ai_chat_runtime PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
    )
endif()
