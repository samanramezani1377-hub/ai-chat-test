cmake_minimum_required(VERSION 3.22)
project(ai_chat_runtime LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
set(LLAMA_CPP_TAG "9871df5911a03a813518bcd623ee2eba91bf32aa" CACHE STRING "Pinned llama.cpp revision with Qwen3.x and Vulkan backend support")
set(FETCHCONTENT_QUIET OFF)
set(FETCHCONTENT_BASE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../.cxx/fetch-content" CACHE PATH "Stable FetchContent base directory")
set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL "Do not update pinned FetchContent dependencies")

# Vulkan is supplied by the Android system loader at runtime. CI/local native builds
# need glslc and SPIR-V headers from the LunarG SDK to compile llama.cpp's shaders.
if (ANDROID)
    if (DEFINED ENV{VULKAN_SDK})
        set(Vulkan_INCLUDE_DIR "$ENV{VULKAN_SDK}/include" CACHE PATH "Vulkan headers" FORCE)
        set(Vulkan_GLSLC_EXECUTABLE "$ENV{VULKAN_SDK}/bin/glslc" CACHE FILEPATH "Vulkan shader compiler" FORCE)
        set(SPIRV-Headers_DIR "$ENV{VULKAN_SDK}/share/cmake/SPIRV-Headers" CACHE PATH "SPIRV-Headers CMake package" FORCE)
        set(CMAKE_PREFIX_PATH "$ENV{VULKAN_SDK};${CMAKE_PREFIX_PATH}" CACHE STRING "Vulkan package search path" FORCE)
    endif()
    set(Vulkan_LIBRARY "${CMAKE_ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/29/libvulkan.so" CACHE FILEPATH "Android Vulkan loader" FORCE)
endif()

if (NOT DEFINED Vulkan_GLSLC_EXECUTABLE AND NOT DEFINED ENV{VULKAN_SDK})
    message(FATAL_ERROR "Vulkan SDK is required to compile Vulkan shaders. Set VULKAN_SDK or run ci/prepare-vulkan.sh.")
endif()

FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG ${LLAMA_CPP_TAG}
    GIT_SHALLOW TRUE
)

# GPU-only Android runtime: no OpenCL, CPU fallback, or dynamic backend discovery.
set(GGML_VULKAN ON CACHE BOOL "Use Vulkan GPU backend" FORCE)
set(GGML_OPENCL OFF CACHE BOOL "OpenCL backend removed on the Vulkan branch" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "Statically link the Vulkan backend on Android" FORCE)
set(GGML_BACKEND_DIR "" CACHE STRING "No dynamic backend directory" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "Portable Android native build" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "Avoid OpenMP runtime dependency on Android" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "Disable llamafile on Android" FORCE)
set(GGML_VULKAN_RUN_TESTS OFF CACHE BOOL "Do not build Vulkan backend tests" FORCE)
set(GGML_VULKAN_VALIDATE OFF CACHE BOOL "Do not enable validation layers in production" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Static llama.cpp dependencies" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_UI OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(llama_cpp)

if (NOT TARGET ggml-vulkan)
    message(FATAL_ERROR "GPU-only runtime requires the statically linked ggml-vulkan target")
endif()

add_library(ai_chat_runtime SHARED native_runtime_android_safe.cpp)

target_include_directories(ai_chat_runtime PRIVATE
    ${llama_cpp_SOURCE_DIR}/include
    ${llama_cpp_SOURCE_DIR}/src
    ${llama_cpp_SOURCE_DIR}/ggml/include
    ${llama_cpp_SOURCE_DIR}/ggml/src
)

target_compile_options(ai_chat_runtime PRIVATE
    -Os
    -ffunction-sections
    -fdata-sections
)
target_link_options(ai_chat_runtime PRIVATE
    -Wl,--gc-sections
    -Wl,--strip-all
)

target_link_libraries(ai_chat_runtime PRIVATE
    llama
    llama-common
    ggml-vulkan
    android
    log
    "${Vulkan_LIBRARY}"
)

target_compile_definitions(ai_chat_runtime PRIVATE
    AI_CHAT_LLAMA_CPP_SHA="${LLAMA_CPP_TAG}"
    AI_CHAT_GPU_BACKEND_VULKAN=1
    AI_CHAT_REQUIRE_VULKAN_GPU=1
)

if (ANDROID)
    set_target_properties(ai_chat_runtime PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
    )
endif()
