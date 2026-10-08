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
# OpenGL ES is provided by Android's system EGL/GLES implementation. No
# No vendor-specific GPU SDK is bundled in the APK.
set(GGML_BACKEND_DL OFF CACHE BOOL "Disable dynamic ggml backend loading on Android" FORCE)
set(GGML_BACKEND_DIR "" CACHE STRING "No dynamic backend directory" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "Vulkan is intentionally disabled" FORCE)
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

add_library(ai_chat_runtime SHARED native_runtime_android_safe.cpp opengles_runtime.cpp ggml-opengles/ggml-opengles.cpp ggml-opengles/ggml-opengles-registration.cpp)

# native_runtime.cpp intentionally includes llama.cpp/ggml internal headers for
# strict backend and tensor-residency validation. Keep both internal source
# directories on the include path; the public ggml include directory alone is
# insufficient for ggml-backend-impl.h and ggml-opencl.h on this pinned revision.
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
    AI_CHAT_GPU_BACKEND_VULKAN=0
    AI_CHAT_REQUIRE_OPENGL_GPU=1
)

if (ANDROID)
    set_target_properties(ai_chat_runtime PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
    )
endif()
