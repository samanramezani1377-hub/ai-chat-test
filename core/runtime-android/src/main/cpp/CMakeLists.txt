cmake_minimum_required(VERSION 3.22)
project(ai_chat_runtime LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(AI_CHAT_NATIVE_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}")
include("${AI_CHAT_NATIVE_CMAKE_DIR}/cmake/AIChatPatchHelpers.cmake")



include(FetchContent)
set(LLAMA_CPP_TAG "9871df5911a03a813518bcd623ee2eba91bf32aa" CACHE STRING "Pinned llama.cpp revision with Qwen3.x/OpenCL/Q6_K support")
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

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

set(GGML_OPENCL ON CACHE BOOL "Keep legacy OpenCL backend available for diagnostics" FORCE)
# Android inference uses upstream GGML WebGPU linked to a native Dawn build.
# AndroidX supplies only the pinned Dawn source revision metadata; its bundled
# JNI library is not a public WebGPU C ABI library for native GGML consumers.
set(GGML_WEBGPU ON CACHE BOOL "Enable GGML WebGPU backend through native Dawn" FORCE)
if (NOT DEFINED AI_CHAT_DAWN_METADATA_FILE OR NOT EXISTS "${AI_CHAT_DAWN_METADATA_FILE}")
    message(FATAL_ERROR "AndroidX WebGPU Dawn build metadata was not extracted before CMake configuration")
endif()
# Android runtime links the OpenCL backend statically when available. Dynamic backend
# loading is unreliable on Android because ggml_backend_load_all() cannot discover
# MODULE libraries from the APK's native library namespace.
set(GGML_BACKEND_DL OFF CACHE BOOL "Disable dynamic ggml backend loading on Android" FORCE)
set(GGML_BACKEND_DIR "" CACHE STRING "No dynamic backend directory" FORCE)
# Disable Adreno-only optimized kernels. This does NOT make the pinned upstream
# backend vendor-neutral: its device gate and several kernel dispatch paths still
# target supported Adreno/Intel families. Do not bypass that gate for Mali or any
# unknown family until a real, tested generic kernel/dispatch path exists.
set(GGML_OPENCL_USE_ADRENO_KERNELS OFF CACHE BOOL "Disable Adreno-specific OpenCL kernels" FORCE)
set(GGML_OPENCL_USE_ADRENO_BIN_KERNELS OFF CACHE BOOL "Do not use vendor-specific Adreno binary kernels" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "Vulkan is intentionally disabled" FORCE)
set(GGML_OPENCL_USE_SVM OFF CACHE BOOL "Avoid vendor SVM driver instability on Android" FORCE)

# Profiling is enabled by the Android release configuration so the diagnostics screen
# can expose real OpenCL kernel timing. Compare with profiling disabled only after
# correctness and GPU registration are verified.
option(AI_CHAT_OPENCL_PROFILING "Enable app-visible OpenCL GPU profiling" OFF)
if (AI_CHAT_OPENCL_PROFILING)
    set(GGML_OPENCL_PROFILING ON CACHE BOOL "Enable OpenCL event profiling" FORCE)
endif()
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD ON CACHE BOOL "Build libmtmd for local image understanding" FORCE)
set(MTMD_VIDEO OFF CACHE BOOL "No video helper binaries are needed for screen images" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_UI OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
# Keep the generic ARM64 backend portable. Never compile the whole ggml CPU
# backend for the build host or force newer ISA features: phones in the same
# arm64-v8a ABI can differ in DotProd/i8mm support. KleidiAI builds its optimized
# microkernels separately and uses its compatible dispatch paths at runtime.
set(GGML_NATIVE OFF CACHE BOOL "Keep the generic CPU backend ISA-portable" FORCE)
if (ANDROID)
    if (NOT CMAKE_ANDROID_ARCH_ABI STREQUAL "arm64-v8a")
        message(FATAL_ERROR "AI Chat Test native runtime supports arm64-v8a only; got '${CMAKE_ANDROID_ARCH_ABI}'")
    endif()
    # Pin the generic backend to the ARMv8-A baseline. Do not raise this to
    # armv8.2-a/i8mm or a device-specific -mcpu: that would risk SIGILL on
    # otherwise supported arm64 phones. KleidiAI's own kernels remain enabled.
    set(GGML_CPU_ARM_ARCH "armv8-a" CACHE STRING "Portable generic ARM64 CPU baseline" FORCE)
    message(STATUS "AI_CHAT_CPU_ARCH_POLICY baseline=armv8-a native=OFF kleidiai=ON runtime_compatible_dispatch=ON")
endif()
set(GGML_CPU_KLEIDIAI ON CACHE BOOL "Enable portable ARM KleidiAI CPU microkernels" FORCE)

# OpenMP is enabled for the CPU backend. Android's NDK Clang requires static
# OpenMP runtime linkage so the APK does not depend on a system libomp.so.
set(GGML_OPENMP ON CACHE BOOL "Enable GGML OpenMP CPU parallelism" FORCE)
if (ANDROID)
    set(OpenMP_C_FLAGS "-fopenmp -static-openmp" CACHE STRING "Android static OpenMP C flags" FORCE)
    set(OpenMP_CXX_FLAGS "-fopenmp -static-openmp" CACHE STRING "Android static OpenMP C++ flags" FORCE)
endif()
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(llama_cpp)

include("${AI_CHAT_NATIVE_CMAKE_DIR}/cmake/patch_webgpu_backend.cmake")

# Trace and validate the per-layer recurrent state cache on the selected backend.
include("${CMAKE_CURRENT_LIST_DIR}/cmake/patch_recurrent_memory_diagnostics.cmake")

# Do not silently ship a build where OpenMP was requested but FindOpenMP could
# not enable it. This also catches NDK/runtime-linking regressions during CI.
if (NOT GGML_OPENMP_ENABLED)
    message(FATAL_ERROR "GGML_OPENMP=ON was requested, but the pinned GGML build did not enable OpenMP")
endif()
if (NOT GGML_CPU_KLEIDIAI)
    message(FATAL_ERROR "GGML_CPU_KLEIDIAI=ON was requested, but KleidiAI is disabled")
endif()

# Add one-shot per-node CPU/OpenCL backend assignment logs for performance diagnostics.
include("${CMAKE_CURRENT_LIST_DIR}/cmake/patch_backend_assignment_trace.cmake")

# Add bounded diagnostics around graph reservation and Dawn buffer allocation.
include("${CMAKE_CURRENT_LIST_DIR}/cmake/patch_graph_reservation_diagnostics.cmake")

# Prevent unbounded virtual chunks and make Dawn allocation failures recoverable.
include("${CMAKE_CURRENT_LIST_DIR}/cmake/patch_bounded_buffer_allocation.cmake")

# Convert Dawn callback/wait failures into recoverable GGML backend errors.
include("${CMAKE_CURRENT_LIST_DIR}/cmake/patch_webgpu_failure_handling.cmake")

include("${AI_CHAT_NATIVE_CMAKE_DIR}/cmake/patch_android_opencl.cmake")

# Backend-neutral GPU engine contracts are intentionally isolated from llama.cpp/GGML.
# This target may be reused by future backends and must not link their execution APIs.
add_library(ai_chat_gpu_engine STATIC
    gpu_engine/gpu_engine.cpp
    gpu_engine/universal_adapter_bridge.cpp
)
target_include_directories(ai_chat_gpu_engine PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
)
target_compile_features(ai_chat_gpu_engine PUBLIC cxx_std_17)

# Dawn is isolated in its own backend target. The backend-neutral engine core
# has no Dawn, llama.cpp or GGML dependency.
if (NOT TARGET dawn::webgpu_dawn)
    message(FATAL_ERROR "The standalone GPU engine requires the pinned Dawn target")
endif()
add_library(ai_chat_gpu_backend_dawn STATIC gpu_engine/dawn_backend.cpp)
target_include_directories(ai_chat_gpu_backend_dawn PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
)
target_compile_features(ai_chat_gpu_backend_dawn PUBLIC cxx_std_17)
target_link_libraries(ai_chat_gpu_backend_dawn PUBLIC ai_chat_gpu_engine PRIVATE dawn::webgpu_dawn)

add_library(ai_chat_runtime SHARED native_runtime_android_safe.cpp vision_runtime_android.cpp universal_gpu_adapter.cpp)

# native_runtime.cpp intentionally includes llama.cpp/ggml internal headers for
# strict backend and tensor-residency validation. Keep both internal source
# directories on the include path; the public ggml include directory alone is
# insufficient for ggml-backend-impl.h and ggml-opencl.h on this pinned revision.
target_include_directories(ai_chat_runtime PRIVATE
    ${llama_cpp_SOURCE_DIR}/include
    ${llama_cpp_SOURCE_DIR}/src
    ${llama_cpp_SOURCE_DIR}/ggml/include
    ${llama_cpp_SOURCE_DIR}/ggml/src
    ${llama_cpp_SOURCE_DIR}/ggml/src/ggml-opencl
    ${llama_cpp_SOURCE_DIR}/ggml/src/ggml-webgpu
    ${llama_cpp_SOURCE_DIR}/tools/mtmd
    ${opencl_headers_SOURCE_DIR}
)

# The runtime is a single arm64 native library. Keep line-table symbols and a
# stable ELF Build ID so Android tombstones can be symbolized against the exact
# shipped binary. Section GC still removes unreachable code; do not use --strip-all
# here because it makes native backend aborts impossible to diagnose by source line.
target_compile_options(ai_chat_runtime PRIVATE
    -Os
    -gline-tables-only
    -ffunction-sections
    -fdata-sections
)
target_link_options(ai_chat_runtime PRIVATE
    -Wl,--gc-sections
    -Wl,--build-id=sha1
)

if (NOT TARGET ggml-webgpu)
    message(FATAL_ERROR "GGML_WEBGPU=ON was requested, but ggml-webgpu target was not created")
endif()
target_link_libraries(ai_chat_runtime PRIVATE ai_chat_gpu_engine ai_chat_gpu_backend_dawn llama llama-common mtmd ggml-opencl ggml-webgpu android log OpenCL EGL GLESv3)

target_compile_definitions(ai_chat_runtime PRIVATE
    AI_CHAT_LLAMA_CPP_SHA="${LLAMA_CPP_TAG}"
    AI_CHAT_GPU_BACKEND_OPENCL=1
    AI_CHAT_GPU_BACKEND_WEBGPU=1
    AI_CHAT_GPU_BACKEND_VULKAN=0
    AI_CHAT_GPU_ADAPTER_OPENGL_ES=1
    AI_CHAT_GGML_OPENMP_ENABLED=1
    AI_CHAT_GGML_CPU_KLEIDIAI_ENABLED=1
    AI_CHAT_CPU_ARM_BASELINE="armv8-a"
)

if (ANDROID)
    set_target_properties(ai_chat_runtime PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
    )
endif()
