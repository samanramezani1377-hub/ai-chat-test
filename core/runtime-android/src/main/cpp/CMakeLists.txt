cmake_minimum_required(VERSION 3.22)
project(ai_chat_runtime LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
set(LLAMA_CPP_TAG "c5fc7e34885ba31217e330809437afa993d27745" CACHE STRING "Pinned llama.cpp revision")
set(FETCHCONTENT_QUIET OFF)

# Keep FetchContent outside CMake's per-configuration _deps directory.
# The Android Gradle plugin uses changing .cxx/<build-hash>/... directories;
# placing the dependency source in this stable module-local directory lets the
# CI .cxx cache preserve the llama.cpp checkout across native source changes.
set(FETCHCONTENT_BASE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../.cxx/fetch-content" CACHE PATH "Stable FetchContent base directory")
set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL "Do not update pinned FetchContent dependencies")

# Direct llama.cpp integration: no llama.kt/llama.rn bootstrap layer.
if (ANDROID AND DEFINED ENV{VULKAN_SDK})
    set(Vulkan_INCLUDE_DIR "$ENV{VULKAN_SDK}/include" CACHE PATH "Vulkan headers" FORCE)
    set(Vulkan_GLSLC_EXECUTABLE "$ENV{VULKAN_SDK}/bin/glslc" CACHE FILEPATH "Vulkan shader compiler" FORCE)
    set(Vulkan_LIBRARY "${CMAKE_ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/29/libvulkan.so" CACHE FILEPATH "Android Vulkan loader" FORCE)

    set(SPIRV-Headers_DIR "$ENV{VULKAN_SDK}/share/cmake/SPIRV-Headers" CACHE PATH "SPIRV-Headers CMake package" FORCE)
    set(CMAKE_PREFIX_PATH "$ENV{VULKAN_SDK};${CMAKE_PREFIX_PATH}" CACHE STRING "CMake package search path" FORCE)
endif()

FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG ${LLAMA_CPP_TAG}
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(llama_cpp)

# Downstream fix for Android model loading: this is the upstream llama.cpp
# change from 5ec4eab69edbfaa4b01bcf1ee1123bb885b8d892.
set(LLAMA_MODEL_LOADING_PATCH "${CMAKE_CURRENT_LIST_DIR}/patches/llama-model-loading-memory.patch")
if (EXISTS "${LLAMA_MODEL_LOADING_PATCH}")
    execute_process(
        COMMAND git -C "${llama_cpp_SOURCE_DIR}" apply --check "${LLAMA_MODEL_LOADING_PATCH}"
        RESULT_VARIABLE LLAMA_PATCH_CHECK_RESULT
        OUTPUT_QUIET
        ERROR_QUIET
    )
    if (LLAMA_PATCH_CHECK_RESULT EQUAL 0)
        execute_process(
            COMMAND git -C "${llama_cpp_SOURCE_DIR}" apply "${LLAMA_MODEL_LOADING_PATCH}"
            RESULT_VARIABLE LLAMA_PATCH_APPLY_RESULT
        )
        if (NOT LLAMA_PATCH_APPLY_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to apply llama.cpp model-loading memory patch")
        endif()
        message(STATUS "Applied llama.cpp Android model-loading memory patch")
    else()
        execute_process(
            COMMAND git -C "${llama_cpp_SOURCE_DIR}" apply --reverse --check "${LLAMA_MODEL_LOADING_PATCH}"
            RESULT_VARIABLE LLAMA_PATCH_REVERSE_CHECK_RESULT
            OUTPUT_QUIET
            ERROR_QUIET
        )
        if (LLAMA_PATCH_REVERSE_CHECK_RESULT EQUAL 0)
            message(STATUS "llama.cpp model-loading memory patch already applied")
        else()
            message(FATAL_ERROR "llama.cpp source is neither clean nor patched with the expected model-loading fix")
        endif()
    endif()
else()
    message(FATAL_ERROR "Missing llama.cpp model-loading patch: ${LLAMA_MODEL_LOADING_PATCH}")
endif()

# Diagnostic instrumentation is deliberately a separate downstream patch.
# It adds stage markers around model creation, device preparation, hparams,
# vocabulary and tensor loading so the next device run identifies the exact
# failing stage instead of relying on truncated llama.cpp logs.
set(LLAMA_MODEL_LOADING_DIAGNOSTICS_PATCH "${CMAKE_CURRENT_LIST_DIR}/patches/llama-model-loading-diagnostics.patch")
if (EXISTS "${LLAMA_MODEL_LOADING_DIAGNOSTICS_PATCH}")
    execute_process(
        COMMAND git -C "${llama_cpp_SOURCE_DIR}" apply --check "${LLAMA_MODEL_LOADING_DIAGNOSTICS_PATCH}"
        RESULT_VARIABLE LLAMA_DIAG_PATCH_CHECK_RESULT
        OUTPUT_QUIET
        ERROR_QUIET
    )
    if (LLAMA_DIAG_PATCH_CHECK_RESULT EQUAL 0)
        execute_process(
            COMMAND git -C "${llama_cpp_SOURCE_DIR}" apply "${LLAMA_MODEL_LOADING_DIAGNOSTICS_PATCH}"
            RESULT_VARIABLE LLAMA_DIAG_PATCH_APPLY_RESULT
        )
        if (NOT LLAMA_DIAG_PATCH_APPLY_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to apply llama.cpp model-loading diagnostics patch")
        endif()
        message(STATUS "Applied llama.cpp Android model-loading diagnostics patch")
    else()
        execute_process(
            COMMAND git -C "${llama_cpp_SOURCE_DIR}" apply --reverse --check "${LLAMA_MODEL_LOADING_DIAGNOSTICS_PATCH}"
            RESULT_VARIABLE LLAMA_DIAG_PATCH_REVERSE_CHECK_RESULT
            OUTPUT_QUIET
            ERROR_QUIET
        )
        if (LLAMA_DIAG_PATCH_REVERSE_CHECK_RESULT EQUAL 0)
            message(STATUS "llama.cpp model-loading diagnostics patch already applied")
        else()
            message(FATAL_ERROR "llama.cpp source is neither clean nor instrumented with the expected diagnostics patch")
        endif()
    endif()
else()
    message(FATAL_ERROR "Missing llama.cpp model-loading diagnostics patch: ${LLAMA_MODEL_LOADING_DIAGNOSTICS_PATCH}")
endif()

add_library(ai_chat_runtime SHARED native_runtime_android_safe.cpp)

target_include_directories(ai_chat_runtime PRIVATE
    ${llama_cpp_SOURCE_DIR}/include
    ${llama_cpp_SOURCE_DIR}/ggml/include
)

target_link_libraries(ai_chat_runtime llama android log)

target_compile_definitions(ai_chat_runtime PRIVATE AI_CHAT_LLAMA_CPP_SHA="${LLAMA_CPP_TAG}")

if (ANDROID)
    set_target_properties(ai_chat_runtime PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
    )
endif()
