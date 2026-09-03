cmake_minimum_required(VERSION 3.22)
project(ai_chat_runtime LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
set(LLAMA_CPP_TAG "c5fc7e34885ba31217e330809437afa993d27745" CACHE STRING "Pinned llama.cpp revision")
set(FETCHCONTENT_QUIET OFF)

# Keep the dependency in a stable location, but never rely on a cached patched
# tree. CMake always resets the checkout to the exact pinned revision first.
set(FETCHCONTENT_BASE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../.cxx/fetch-content" CACHE PATH "Stable FetchContent base directory")
set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL "Do not update pinned FetchContent dependencies")

# Vulkan is a build-time capability. Runtime CPU mode never calls the Vulkan
# backend loader; GPU mode explicitly loads it from native_runtime.cpp.
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

execute_process(
    COMMAND git -C "${llama_cpp_SOURCE_DIR}" reset --hard ${LLAMA_CPP_TAG}
    RESULT_VARIABLE LLAMA_RESET_RESULT
    OUTPUT_VARIABLE LLAMA_RESET_OUTPUT
    ERROR_VARIABLE LLAMA_RESET_ERROR
)
if (NOT LLAMA_RESET_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to reset cached llama.cpp source to ${LLAMA_CPP_TAG}: ${LLAMA_RESET_ERROR}")
endif()

execute_process(
    COMMAND git -C "${llama_cpp_SOURCE_DIR}" clean -fd
    RESULT_VARIABLE LLAMA_CLEAN_RESULT
    OUTPUT_VARIABLE LLAMA_CLEAN_OUTPUT
    ERROR_VARIABLE LLAMA_CLEAN_ERROR
)
if (NOT LLAMA_CLEAN_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to clean cached llama.cpp source: ${LLAMA_CLEAN_ERROR}")
endif()

message(STATUS "llama.cpp source reset to pinned revision ${LLAMA_CPP_TAG}")

# Android memory/staging fix. This patch is known to apply to the pinned
# revision. Keep exactly one downstream llama.cpp patch in the build path.
set(LLAMA_MODEL_LOADING_PATCH "${CMAKE_CURRENT_LIST_DIR}/patches/llama-model-loading-memory.patch")
if (EXISTS "${LLAMA_MODEL_LOADING_PATCH}")
    execute_process(
        COMMAND git -C "${llama_cpp_SOURCE_DIR}" apply --check "${LLAMA_MODEL_LOADING_PATCH}"
        RESULT_VARIABLE LLAMA_PATCH_CHECK_RESULT
        OUTPUT_QUIET
        ERROR_VARIABLE LLAMA_PATCH_CHECK_ERROR
    )
    if (NOT LLAMA_PATCH_CHECK_RESULT EQUAL 0)
        message(FATAL_ERROR "Pinned llama.cpp source does not accept the Android model-loading memory patch: ${LLAMA_PATCH_CHECK_ERROR}")
    endif()

    execute_process(
        COMMAND git -C "${llama_cpp_SOURCE_DIR}" apply "${LLAMA_MODEL_LOADING_PATCH}"
        RESULT_VARIABLE LLAMA_PATCH_APPLY_RESULT
        ERROR_VARIABLE LLAMA_PATCH_APPLY_ERROR
    )
    if (NOT LLAMA_PATCH_APPLY_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to apply llama.cpp model-loading memory patch: ${LLAMA_PATCH_APPLY_ERROR}")
    endif()
    message(STATUS "Applied llama.cpp Android model-loading memory patch")
else()
    message(FATAL_ERROR "Missing llama.cpp model-loading patch: ${LLAMA_MODEL_LOADING_PATCH}")
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
