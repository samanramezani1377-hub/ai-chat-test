cmake_minimum_required(VERSION 3.22)
project(ai_chat_runtime LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
set(LLAMA_CPP_TAG "c5fc7e34885ba31217e330809437afa993d27745" CACHE STRING "Pinned llama.cpp revision")
set(FETCHCONTENT_QUIET OFF)

set(FETCHCONTENT_BASE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../.cxx/fetch-content" CACHE PATH "Stable FetchContent base directory")
set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL "Do not update pinned FetchContent dependencies")

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

set(LLAMA_MODEL_LOADING_PATCH "${CMAKE_CURRENT_LIST_DIR}/patches/llama-model-loading-memory.patch")
set(LLAMA_BACKEND_DIAGNOSTICS_PATCH "${CMAKE_CURRENT_LIST_DIR}/patches/llama-backend-vulkan-diagnostics.patch")
set(LLAMA_VULKAN_MICRO_DIAGNOSTICS_PATCH "${CMAKE_CURRENT_LIST_DIR}/patches/llama-vulkan-micro-diagnostics.patch")
foreach(LLAMA_PATCH IN ITEMS "${LLAMA_MODEL_LOADING_PATCH}" "${LLAMA_BACKEND_DIAGNOSTICS_PATCH}" "${LLAMA_VULKAN_MICRO_DIAGNOSTICS_PATCH}")
    if (NOT EXISTS "${LLAMA_PATCH}")
        message(FATAL_ERROR "Missing llama.cpp downstream patch: ${LLAMA_PATCH}")
    endif()
endforeach()

file(SHA256 "${LLAMA_MODEL_LOADING_PATCH}" LLAMA_MODEL_LOADING_PATCH_SHA256)
file(SHA256 "${LLAMA_BACKEND_DIAGNOSTICS_PATCH}" LLAMA_BACKEND_DIAGNOSTICS_PATCH_SHA256)
file(SHA256 "${LLAMA_VULKAN_MICRO_DIAGNOSTICS_PATCH}" LLAMA_VULKAN_MICRO_DIAGNOSTICS_PATCH_SHA256)
set(LLAMA_PREPARED_MARKER "${FETCHCONTENT_BASE_DIR}/.ai-chat-test-llama-prepared-${LLAMA_CPP_TAG}")
set(LLAMA_PREPARED_EXPECTED "${LLAMA_CPP_TAG}\n${LLAMA_MODEL_LOADING_PATCH_SHA256}\n${LLAMA_BACKEND_DIAGNOSTICS_PATCH_SHA256}\n${LLAMA_VULKAN_MICRO_DIAGNOSTICS_PATCH_SHA256}\n")
set(LLAMA_SOURCE_PREPARED FALSE)

if (EXISTS "${LLAMA_PREPARED_MARKER}")
    file(READ "${LLAMA_PREPARED_MARKER}" LLAMA_PREPARED_CONTENT)
    execute_process(
        COMMAND git -C "${llama_cpp_SOURCE_DIR}" rev-parse HEAD
        RESULT_VARIABLE LLAMA_HEAD_RESULT
        OUTPUT_VARIABLE LLAMA_HEAD
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if (LLAMA_HEAD_RESULT EQUAL 0 AND LLAMA_HEAD STREQUAL LLAMA_CPP_TAG AND LLAMA_PREPARED_CONTENT STREQUAL LLAMA_PREPARED_EXPECTED)
        execute_process(COMMAND git -C "${llama_cpp_SOURCE_DIR}" apply --reverse --check "${LLAMA_MODEL_LOADING_PATCH}" RESULT_VARIABLE LLAMA_MODEL_REVERSE_RESULT OUTPUT_QUIET ERROR_QUIET)
        execute_process(COMMAND git -C "${llama_cpp_SOURCE_DIR}" apply --reverse --check "${LLAMA_BACKEND_DIAGNOSTICS_PATCH}" RESULT_VARIABLE LLAMA_BACKEND_REVERSE_RESULT OUTPUT_QUIET ERROR_QUIET)
        execute_process(COMMAND git -C "${llama_cpp_SOURCE_DIR}" apply --reverse --check "${LLAMA_VULKAN_MICRO_DIAGNOSTICS_PATCH}" RESULT_VARIABLE LLAMA_MICRO_REVERSE_RESULT OUTPUT_QUIET ERROR_QUIET)
        if (LLAMA_MODEL_REVERSE_RESULT EQUAL 0 AND LLAMA_BACKEND_REVERSE_RESULT EQUAL 0 AND LLAMA_MICRO_REVERSE_RESULT EQUAL 0)
            set(LLAMA_SOURCE_PREPARED TRUE)
            message(STATUS "llama.cpp source preparation already complete; skipping reset/clean/reapply")
        endif()
    endif()
endif()

if (NOT LLAMA_SOURCE_PREPARED)
    execute_process(COMMAND git -C "${llama_cpp_SOURCE_DIR}" reset --hard ${LLAMA_CPP_TAG} RESULT_VARIABLE LLAMA_RESET_RESULT ERROR_VARIABLE LLAMA_RESET_ERROR)
    if (NOT LLAMA_RESET_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to reset llama.cpp source to ${LLAMA_CPP_TAG}: ${LLAMA_RESET_ERROR}")
    endif()
    execute_process(COMMAND git -C "${llama_cpp_SOURCE_DIR}" clean -fd RESULT_VARIABLE LLAMA_CLEAN_RESULT ERROR_VARIABLE LLAMA_CLEAN_ERROR)
    if (NOT LLAMA_CLEAN_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to clean llama.cpp source: ${LLAMA_CLEAN_ERROR}")
    endif()
    foreach(LLAMA_PATCH IN ITEMS "${LLAMA_MODEL_LOADING_PATCH}" "${LLAMA_BACKEND_DIAGNOSTICS_PATCH}" "${LLAMA_VULKAN_MICRO_DIAGNOSTICS_PATCH}")
        execute_process(COMMAND git -C "${llama_cpp_SOURCE_DIR}" apply --check "${LLAMA_PATCH}" RESULT_VARIABLE LLAMA_PATCH_CHECK_RESULT OUTPUT_QUIET ERROR_VARIABLE LLAMA_PATCH_CHECK_ERROR)
        if (NOT LLAMA_PATCH_CHECK_RESULT EQUAL 0)
            message(FATAL_ERROR "Pinned llama.cpp source does not accept downstream patch ${LLAMA_PATCH}: ${LLAMA_PATCH_CHECK_ERROR}")
        endif()
        execute_process(COMMAND git -C "${llama_cpp_SOURCE_DIR}" apply "${LLAMA_PATCH}" RESULT_VARIABLE LLAMA_PATCH_APPLY_RESULT ERROR_VARIABLE LLAMA_PATCH_APPLY_ERROR)
        if (NOT LLAMA_PATCH_APPLY_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to apply llama.cpp downstream patch ${LLAMA_PATCH}: ${LLAMA_PATCH_APPLY_ERROR}")
        endif()
    endforeach()
    file(MAKE_DIRECTORY "${FETCHCONTENT_BASE_DIR}")
    file(WRITE "${LLAMA_PREPARED_MARKER}" "${LLAMA_PREPARED_EXPECTED}")
    message(STATUS "Prepared llama.cpp ${LLAMA_CPP_TAG} and applied Android model-loading/backend/micro diagnostics patches")
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
