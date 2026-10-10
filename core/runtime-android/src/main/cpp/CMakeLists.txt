cmake_minimum_required(VERSION 3.22)
project(ai_chat_runtime LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

set(GGML_OPENCL ON CACHE BOOL "Use OpenCL GPU backend" FORCE)
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
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_UI OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
# Keep a universally safe ARM baseline and explicitly retain CPU weight-repack support.
# Feature-specific kernels must not be forced globally onto devices that lack their ISA.
set(GGML_CPU_ARM_ARCH "armv8-a" CACHE STRING "Portable Android ARM64 CPU baseline" FORCE)
set(GGML_CPU_REPACK ON CACHE BOOL "Enable supported CPU weight repacking" FORCE)
# KleidiAI does not provide the Q6_K decode kernel that dominates the target model.
# Leave it off until a measured compatible path exists for this pinned llama.cpp revision.
set(GGML_CPU_KLEIDIAI OFF CACHE BOOL "Enable KleidiAI only when useful for supported model kernels" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(llama_cpp)

# Add one-shot per-node CPU/OpenCL backend assignment logs for performance diagnostics.
include("${CMAKE_CURRENT_LIST_DIR}/cmake/patch_backend_assignment_trace.cmake")

# Android OpenCL vendor libraries are often complete OpenCL implementations but
# not ICDs. Patch only the pinned ggml OpenCL call sites to use our direct API
# dispatcher; the Khronos loader remains linked for ABI compatibility but is not
# used for driver dispatch. This avoids fabricating the ICD extension entrypoint.
set(AI_CHAT_OPENCL_SOURCE_FILES
    "${llama_cpp_SOURCE_DIR}/ggml/src/ggml-opencl/ggml-opencl.cpp"
    "${llama_cpp_SOURCE_DIR}/ggml/src/ggml-opencl/cl-program-cache.cpp"
)
foreach(AI_CHAT_OPENCL_SOURCE IN LISTS AI_CHAT_OPENCL_SOURCE_FILES)
    file(READ "${AI_CHAT_OPENCL_SOURCE}" AI_CHAT_OPENCL_SOURCE_TEXT)
    string(FIND "${AI_CHAT_OPENCL_SOURCE_TEXT}" "android_opencl_dispatch.h" AI_CHAT_OPENCL_ALREADY_PATCHED)
    if (AI_CHAT_OPENCL_ALREADY_PATCHED EQUAL -1)
        if (AI_CHAT_OPENCL_SOURCE MATCHES "ggml-opencl\\.cpp$")
            string(FIND "${AI_CHAT_OPENCL_SOURCE_TEXT}" "#include <CL/cl.h>" AI_CHAT_OPENCL_INCLUDE_POS)
            if (AI_CHAT_OPENCL_INCLUDE_POS EQUAL -1)
                message(FATAL_ERROR "Could not patch direct Android OpenCL dispatch in ${AI_CHAT_OPENCL_SOURCE}")
            endif()
            string(REPLACE "#include <CL/cl.h>"
                "#include <CL/cl.h>\n#include \"android_opencl_dispatch.h\""
                AI_CHAT_OPENCL_SOURCE_TEXT "${AI_CHAT_OPENCL_SOURCE_TEXT}")
        else()
            string(FIND "${AI_CHAT_OPENCL_SOURCE_TEXT}" "#include \"cl-program-cache.h\"" AI_CHAT_OPENCL_INCLUDE_POS)
            if (AI_CHAT_OPENCL_INCLUDE_POS EQUAL -1)
                message(FATAL_ERROR "Could not patch direct Android OpenCL dispatch in ${AI_CHAT_OPENCL_SOURCE}")
            endif()
            string(REPLACE "#include \"cl-program-cache.h\""
                "#include \"cl-program-cache.h\"\n#include \"android_opencl_dispatch.h\""
                AI_CHAT_OPENCL_SOURCE_TEXT "${AI_CHAT_OPENCL_SOURCE_TEXT}")
        endif()
        file(WRITE "${AI_CHAT_OPENCL_SOURCE}" "${AI_CHAT_OPENCL_SOURCE_TEXT}")
    endif()
endforeach()

# Do not spoof an unsupported GPU family by matching device-name substrings. The pinned
# upstream OpenCL backend contains family-specific kernels and subgroup assumptions.
# Device-agnostic support must be added through runtime capability checks and genuinely
# generic kernels, not by relabeling Mali/unknown devices as Adreno or Intel.
target_include_directories(ggml-opencl PRIVATE "${CMAKE_CURRENT_LIST_DIR}")

if (NOT TARGET ggml-opencl)
    message(FATAL_ERROR "OpenCL-preferred runtime requires the statically linked ggml-opencl target")
endif()
target_include_directories(ggml-opencl PRIVATE "${CMAKE_CURRENT_LIST_DIR}")

# Static backend hardening: make the OpenCL registration visible to the final
# runtime target as a real link dependency. This prevents linker GC from
# discarding the backend registration object and makes a missing OpenCL target
# a build-time error instead of a silent CPU-only runtime.
if (NOT TARGET ggml-opencl)
    message(FATAL_ERROR "OpenCL-preferred runtime requires the statically linked ggml-opencl target")
endif()
if (TARGET ggml)
    target_compile_definitions(ggml PUBLIC GGML_USE_OPENCL=1)
endif()

if (AI_CHAT_OPENCL_PROFILING)
    include("${CMAKE_CURRENT_LIST_DIR}/cmake/patch_opencl_profiling.cmake")
endif()

add_library(ai_chat_runtime SHARED native_runtime_android_safe.cpp)

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
    ${opencl_headers_SOURCE_DIR}
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

target_link_libraries(ai_chat_runtime PRIVATE llama llama-common ggml-opencl android log OpenCL)

target_compile_definitions(ai_chat_runtime PRIVATE
    AI_CHAT_LLAMA_CPP_SHA="${LLAMA_CPP_TAG}"
    AI_CHAT_GPU_BACKEND_OPENCL=1
    AI_CHAT_GPU_BACKEND_VULKAN=0
    AI_CHAT_CPU_BASELINE_ARM_ARCH=\\"${GGML_CPU_ARM_ARCH}\\"
    AI_CHAT_CPU_REPACK=1
    AI_CHAT_CPU_KLEIDIAI=0
    AI_CHAT_CPU_DOTPROD_KERNELS=0
    AI_CHAT_CPU_I8MM_KERNELS=0
)

if (ANDROID)
    set_target_properties(ai_chat_runtime PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
    )
endif()
