cmake_minimum_required(VERSION 3.22)
project(ai_chat_runtime LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
set(LLAMA_CPP_TAG "2e7c58c" CACHE STRING "Pinned llama.cpp revision with Android OpenCL support")
set(FETCHCONTENT_QUIET OFF)
set(FETCHCONTENT_BASE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../.cxx/fetch-content" CACHE PATH "Stable FetchContent base directory")
set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL "Do not update pinned FetchContent dependencies")

FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG ${LLAMA_CPP_TAG}
    GIT_SHALLOW TRUE
)
FetchContent_Declare(
    opencl_headers
    GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-Headers.git
    GIT_TAG v2025.07.22
    GIT_SHALLOW TRUE
)
FetchContent_Declare(
    opencl_icd_loader
    GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-ICD-Loader.git
    GIT_TAG v2025.07.22
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(opencl_headers)
set(OPENCL_ICD_LOADER_HEADERS_DIR "${opencl_headers_SOURCE_DIR}" CACHE PATH "OpenCL headers for the ICD loader" FORCE)
set(OPENCL_ICD_LOADER_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(ENABLE_OPENCL_LAYER OFF CACHE BOOL "" FORCE)
set(ENABLE_OPENCL_ICD_LOADER ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(opencl_icd_loader)

# Android's CMake toolchain does not expose a linkable system OpenCL library.
# The ICD loader target above is our portable link-time OpenCL implementation;
# the real vendor OpenCL implementation is discovered dynamically at runtime.
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

set(GGML_OPENCL ON CACHE BOOL "Use OpenCL GPU backend" FORCE)
set(GGML_OPENCL_USE_ADRENO_KERNELS OFF CACHE BOOL "Use portable OpenCL kernels, not Adreno-specific kernels" FORCE)
set(GGML_OPENCL_USE_ADRENO_BIN_KERNELS OFF CACHE BOOL "Do not use vendor-specific Adreno binary kernels" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "Vulkan is intentionally disabled" FORCE)
set(GGML_OPENCL_USE_SVM OFF CACHE BOOL "Avoid vendor SVM driver instability on Android" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
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

add_library(ai_chat_runtime SHARED native_runtime_android_safe.cpp)

target_include_directories(ai_chat_runtime PRIVATE
    ${llama_cpp_SOURCE_DIR}/include
    ${llama_cpp_SOURCE_DIR}/ggml/include
    ${opencl_headers_SOURCE_DIR}
)

target_link_libraries(ai_chat_runtime llama android log OpenCL)

target_compile_definitions(ai_chat_runtime PRIVATE
    AI_CHAT_LLAMA_CPP_SHA="${LLAMA_CPP_TAG}"
    AI_CHAT_GPU_BACKEND_OPENCL=1
    AI_CHAT_GPU_BACKEND_VULKAN=0
)

if (ANDROID)
    set_target_properties(ai_chat_runtime PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
    )
endif()
